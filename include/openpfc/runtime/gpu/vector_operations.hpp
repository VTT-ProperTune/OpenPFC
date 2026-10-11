// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/solvers/vector_operations.hpp>
#include <type_traits>

namespace pfc::solvers::gpu {
template <class Tag>
concept EnabledBackend = false
#if defined(OpenPFC_ENABLE_CUDA)
                         || std::is_same_v<Tag, pfc::backend::CUDATag>
#endif
#if defined(OpenPFC_ENABLE_HIP)
                         || std::is_same_v<Tag, pfc::backend::HIPTag>
#endif
    ;
namespace detail {
#if defined(OpenPFC_ENABLE_CUDA)
void combine_cuda(double *, double, const double *, double, std::size_t);
double dot_cuda(const double *, const double *, std::size_t, double *);
void mean_cuda(double *, const double *, std::size_t, int, double *, bool);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void combine_hip(double *, double, const double *, double, std::size_t);
double dot_hip(const double *, const double *, std::size_t, double *);
void mean_hip(double *, const double *, std::size_t, int, double *, bool);
#endif
inline void check(pfc::gpuError_t error) {
  if (error != pfc::gpuSuccess)
    throw std::runtime_error(pfc::gpuGetErrorString(error));
}
} // namespace detail
/// Resident vectors with device reductions and optional MPI scalar allreduce.
/// Host transfers consist of one double per dot product. Scalar control of the
/// algorithms stays on the host; no host mirror participates in field algebra.
template <EnabledBackend Tag> class VectorOperations {
  MPI_Comm comm_;
  pfc::core::DataBuffer<Tag, double> workspace_;
  void workspace(std::size_t n) {
    auto count =
        std::max<std::size_t>(256, std::min<std::size_t>(1024, (n + 255) / 256));
    if (workspace_.size() < count) workspace_.resize(count);
  }
  static void combine(double *y, double a, const double *x, double b,
                      std::size_t n) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::combine_cuda(y, a, x, b, n);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::combine_hip(y, a, x, b, n);
#endif
  }
  void mean(double *y, const double *x, std::size_t n, int components,
            bool subtract) {
    if (components < 1 || components > 256 || n % components)
      throw std::invalid_argument("invalid component mean layout");
    workspace(n);
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::mean_cuda(y, x, n, components, workspace_.data(), subtract);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::mean_hip(y, x, n, components, workspace_.data(), subtract);
#endif
  }

public:
  using Vector = pfc::core::DataBuffer<Tag, double>;
  ReductionMetrics reductions;
  std::size_t scalar_transfer_bytes = 0;
  explicit VectorOperations(MPI_Comm comm = MPI_COMM_NULL) : comm_(comm) {}
  Vector make(std::size_t n) const {
    Vector x(n);
    combine(x.data(), 0, nullptr, 0, n);
    return x;
  }
  void copy(const Vector &x, Vector &y) const {
    if (x.data() == y.data()) return;
    if (y.size() != x.size()) y.resize(x.size());
    if (x.size())
      detail::check(pfc::gpuMemcpyAsync(y.data(), x.data(),
                                        x.size() * sizeof(double),
                                        pfc::gpuMemcpyDeviceToDevice, nullptr));
  }
  void scale(Vector &x, double a) const {
    combine(x.data(), 0, nullptr, a, x.size());
  }
  void axpy(Vector &y, double a, const Vector &x) const {
    if (x.size() != y.size()) throw std::invalid_argument("vector size mismatch");
    combine(y.data(), a, x.data(), 1, y.size());
  }
  double dot(const Vector &x, const Vector &y) {
    if (x.size() != y.size()) throw std::invalid_argument("vector size mismatch");
    workspace(x.size());
    const auto start = std::chrono::steady_clock::now();
    double local = 0;
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      local = detail::dot_cuda(x.data(), y.data(), x.size(), workspace_.data());
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      local = detail::dot_hip(x.data(), y.data(), x.size(), workspace_.data());
#endif
    if (x.size()) scalar_transfer_bytes += sizeof(double);
    double result = local;
    if (comm_ != MPI_COMM_NULL) {
      int err = MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_SUM, comm_);
      if (err != MPI_SUCCESS)
        throw std::runtime_error("solver MPI reduction failed");
    }
    ++reductions.calls;
    reductions.seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return result;
  }
  MPI_Comm communicator() const { return comm_; }
  double norm(const Vector &x) { return std::sqrt(dot(x, x)); }
  /// Single-rank componentwise gauge, for interleaved vectors. Distributed
  /// gauges belong to the operator adapter and need communicator-wide means.
  void project(Vector &x, int components = 1) {
    mean(x.data(), x.data(), x.size(), components, true);
  }
  void add_mean(Vector &y, const Vector &x, int components = 1) {
    if (x.size() != y.size()) throw std::invalid_argument("gauge size mismatch");
    mean(y.data(), x.data(), x.size(), components, false);
  }
};
} // namespace pfc::solvers::gpu
