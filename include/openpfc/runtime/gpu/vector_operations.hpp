// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <chrono>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/solvers/vector_operations.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
/// One host copy of the resident Krylov state. `iterations` counts completed
/// steps. `flag` latches the first stop so later kernels leave the vectors
/// unchanged. `pending` marks an Arnoldi breakdown that still rotates the
/// column. Flags: 1 breakdown, 2 nonfinite abort, 3 counted nonfinite residual
/// (CG) or length abort (GMRES), 4 estimated residual reached.
struct ScalarPoll {
  double value = 0;
  int flag = 0;
  int iterations = 0;
  int columns = 0;
  int pending = 0;
};
namespace detail {
#if defined(OpenPFC_ENABLE_CUDA)
void combine_cuda(double *, double, const double *, double, std::size_t);
double dot_cuda(const double *, const double *, std::size_t, double *);
void mean_cuda(double *, const double *, std::size_t, int, double *, bool);
void dot_store_cuda(const double *, const double *, std::size_t, double *, double *,
                    const ScalarPoll *);
void combine_slot_cuda(double *, const double *, double, const double *, double,
                       std::size_t, const ScalarPoll *);
void scale_inv_cuda(double *, const double *, std::size_t, const ScalarPoll *);
void scale_mul_cuda(double *, const double *, std::size_t, const ScalarPoll *);
void scale_ratio_cuda(double *, const double *, const double *, std::size_t,
                      const ScalarPoll *);
void fill_cuda(double *, std::size_t, double);
void set_double_cuda(double *, double);
void copy_double_cuda(double *, const double *, const ScalarPoll *);
void sqrt_slot_cuda(double *, const ScalarPoll *);
void publish_sqrt_cuda(const double *, ScalarPoll *);
void latch_stop_cuda(ScalarPoll *, const double *);
void cg_alpha_cuda(const double *, const double *, double *, ScalarPoll *);
void inc_iter_cuda(ScalarPoll *);
void clear_poll_cuda(ScalarPoll *);
void clear_flag_cuda(ScalarPoll *);
void clear_restart_cuda(ScalarPoll *);
void add_h_cuda(double *, const double *, int, int, const ScalarPoll *);
void prepare_arnoldi_cuda(double *, const double *, const double *, int, int, double,
                          double *, ScalarPoll *);
void givens_cuda(double *, double *, double *, double *, int, int, const double *,
                 ScalarPoll *);
void backsolve_cuda(const double *, const double *, double *, int, int);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void combine_hip(double *, double, const double *, double, std::size_t);
double dot_hip(const double *, const double *, std::size_t, double *);
void mean_hip(double *, const double *, std::size_t, int, double *, bool);
void dot_store_hip(const double *, const double *, std::size_t, double *, double *,
                   const ScalarPoll *);
void combine_slot_hip(double *, const double *, double, const double *, double,
                      std::size_t, const ScalarPoll *);
void scale_inv_hip(double *, const double *, std::size_t, const ScalarPoll *);
void scale_mul_hip(double *, const double *, std::size_t, const ScalarPoll *);
void scale_ratio_hip(double *, const double *, const double *, std::size_t,
                     const ScalarPoll *);
void fill_hip(double *, std::size_t, double);
void set_double_hip(double *, double);
void copy_double_hip(double *, const double *, const ScalarPoll *);
void sqrt_slot_hip(double *, const ScalarPoll *);
void publish_sqrt_hip(const double *, ScalarPoll *);
void latch_stop_hip(ScalarPoll *, const double *);
void cg_alpha_hip(const double *, const double *, double *, ScalarPoll *);
void inc_iter_hip(ScalarPoll *);
void clear_poll_hip(ScalarPoll *);
void clear_flag_hip(ScalarPoll *);
void clear_restart_hip(ScalarPoll *);
void add_h_hip(double *, const double *, int, int, const ScalarPoll *);
void prepare_arnoldi_hip(double *, const double *, const double *, int, int, double,
                         double *, ScalarPoll *);
void givens_hip(double *, double *, double *, double *, int, int, const double *,
                ScalarPoll *);
void backsolve_hip(const double *, const double *, double *, int, int);
#endif
inline void check(pfc::gpuError_t error) {
  if (error != pfc::gpuSuccess)
    throw std::runtime_error(pfc::gpuGetErrorString(error));
}
} // namespace detail
/// Resident scalar kernels. `dot` / `norm` stay on the host path; these entry
/// points do not copy a reduction back.
template <EnabledBackend Tag> struct Resident {
  static void dot_store(const double *x, const double *y, std::size_t n,
                        double *partial, double *result, const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::dot_store_cuda(x, y, n, partial, result, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::dot_store_hip(x, y, n, partial, result, poll);
#endif
  }
  static void combine_slot(double *y, const double *a, double sign, const double *x,
                           double b, std::size_t n, const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::combine_slot_cuda(y, a, sign, x, b, n, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::combine_slot_hip(y, a, sign, x, b, n, poll);
#endif
  }
  static void scale_inv(double *y, const double *a, std::size_t n,
                        const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::scale_inv_cuda(y, a, n, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::scale_inv_hip(y, a, n, poll);
#endif
  }
  static void scale_mul(double *y, const double *a, std::size_t n,
                        const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::scale_mul_cuda(y, a, n, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::scale_mul_hip(y, a, n, poll);
#endif
  }
  static void scale_ratio(double *y, const double *num, const double *den,
                          std::size_t n, const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::scale_ratio_cuda(y, num, den, n, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::scale_ratio_hip(y, num, den, n, poll);
#endif
  }
  static void fill(double *p, std::size_t n, double value) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::fill_cuda(p, n, value);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::fill_hip(p, n, value);
#endif
  }
  static void set_double(double *p, double value) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::set_double_cuda(p, value);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::set_double_hip(p, value);
#endif
  }
  static void copy_double(double *dst, const double *src, const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::copy_double_cuda(dst, src, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::copy_double_hip(dst, src, poll);
#endif
  }
  static void sqrt_slot(double *p, const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::sqrt_slot_cuda(p, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::sqrt_slot_hip(p, poll);
#endif
  }
  static void publish_sqrt(const double *slot, ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::publish_sqrt_cuda(slot, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::publish_sqrt_hip(slot, poll);
#endif
  }
  static void latch_stop(ScalarPoll *poll, const double *stop) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::latch_stop_cuda(poll, stop);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::latch_stop_hip(poll, stop);
#endif
  }
  static void cg_alpha(const double *rho, const double *curv, double *alpha,
                       ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::cg_alpha_cuda(rho, curv, alpha, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::cg_alpha_hip(rho, curv, alpha, poll);
#endif
  }
  static void inc_iter(ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::inc_iter_cuda(poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::inc_iter_hip(poll);
#endif
  }
  static void clear_poll(ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::clear_poll_cuda(poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::clear_poll_hip(poll);
#endif
  }
  static void clear_flag(ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::clear_flag_cuda(poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::clear_flag_hip(poll);
#endif
  }
  static void clear_restart(ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::clear_restart_cuda(poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::clear_restart_hip(poll);
#endif
  }
  static void add_h(double *H, const double *coeff, int j, int ld,
                    const ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::add_h_cuda(H, coeff, j, ld, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::add_h_hip(H, coeff, j, ld, poll);
#endif
  }
  static void prepare_arnoldi(double *H, const double *dot, const double *before,
                              int j, int ld, double eps64, double *scale,
                              ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::prepare_arnoldi_cuda(H, dot, before, j, ld, eps64, scale, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::prepare_arnoldi_hip(H, dot, before, j, ld, eps64, scale, poll);
#endif
  }
  static void givens(double *H, double *cs, double *sn, double *g, int j, int ld,
                     const double *stop, ScalarPoll *poll) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::givens_cuda(H, cs, sn, g, j, ld, stop, poll);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::givens_hip(H, cs, sn, g, j, ld, stop, poll);
#endif
  }
  static void backsolve(const double *H, const double *g, double *y, int columns,
                        int ld) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::backsolve_cuda(H, g, y, columns, ld);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::backsolve_hip(H, g, y, columns, ld);
#endif
  }
};
/// Resident vectors with device reductions and optional MPI scalar allreduce.
/// `dot`, `dots` and `norm` copy one host double per product and remain the
/// reference. Resident Krylov uses `dot_store` / `dots_store` and does not
/// copy those coefficients when `comm` is `MPI_COMM_NULL`.
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
  void account(int count, bool nonempty, std::chrono::steady_clock::time_point start,
               double collective) {
    if (nonempty) reductions.scalars += static_cast<std::size_t>(count);
    ++reductions.calls;
    reductions.seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    reductions.collective_seconds += collective;
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
    double out = 0;
    dots(&x, 1, y, &out);
    return out;
  }
  /// Several inner products against one vector, then one MPI reduction.
  /// Each column is reduced on the device and copied back before that
  /// collective. Every coefficient uses y as it stands at entry.
  void dots(const Vector *columns, int count, const Vector &y, double *out) {
    if (count < 1 || columns == nullptr || out == nullptr)
      throw std::invalid_argument("empty inner-product batch");
    double stack[64];
    std::vector<double> heap;
    double *local = stack;
    if (count > 64) {
      heap.resize(count);
      local = heap.data();
    }
    workspace(y.size());
    const auto start = std::chrono::steady_clock::now();
    for (int column = 0; column < count; ++column) {
      if (columns[column].size() != y.size())
        throw std::invalid_argument("vector size mismatch");
      double value = 0;
#if defined(OpenPFC_ENABLE_CUDA)
      if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
        value = detail::dot_cuda(columns[column].data(), y.data(), y.size(),
                                 workspace_.data());
#endif
#if defined(OpenPFC_ENABLE_HIP)
      if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
        value = detail::dot_hip(columns[column].data(), y.data(), y.size(),
                                workspace_.data());
#endif
      local[column] = value;
    }
    if (y.size()) {
      scalar_transfer_bytes += sizeof(double) * static_cast<std::size_t>(count);
      reductions.scalars += static_cast<std::size_t>(count);
    }
    if (comm_ != MPI_COMM_NULL) {
      const auto collective = std::chrono::steady_clock::now();
      if (MPI_Allreduce(local, out, count, MPI_DOUBLE, MPI_SUM, comm_) !=
          MPI_SUCCESS)
        throw std::runtime_error("solver MPI reduction failed");
      reductions.collective_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        collective)
              .count();
    } else
      for (int column = 0; column < count; ++column) out[column] = local[column];
    ++reductions.calls;
    reductions.seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
  }
  /// Same batch as `dots`, stored in device slots. `MPI_COMM_NULL` does not
  /// copy the slots. A real communicator copies the batch once, reduces it
  /// and copies it back. A latched poll leaves the slots unchanged; the
  /// collective still runs so every rank agrees.
  void dots_store(const Vector *columns, int count, const Vector &y, double *slots,
                  const ScalarPoll *poll = nullptr) {
    if (count < 1 || columns == nullptr || slots == nullptr)
      throw std::invalid_argument("empty inner-product batch");
    for (int column = 0; column < count; ++column)
      if (columns[column].size() != y.size())
        throw std::invalid_argument("vector size mismatch");
    workspace(y.size());
    const auto start = std::chrono::steady_clock::now();
    for (int column = 0; column < count; ++column)
      Resident<Tag>::dot_store(columns[column].data(), y.data(), y.size(),
                               workspace_.data(), slots + column, poll);
    double collective = 0;
    if (comm_ != MPI_COMM_NULL) {
      std::vector<double> local(static_cast<std::size_t>(count));
      std::vector<double> reduced(local.size());
      detail::check(pfc::gpuMemcpy(local.data(), slots,
                                   sizeof(double) * local.size(),
                                   pfc::gpuMemcpyDeviceToHost));
      if (y.size())
        scalar_transfer_bytes += sizeof(double) * static_cast<std::size_t>(count);
      const auto began = std::chrono::steady_clock::now();
      if (MPI_Allreduce(local.data(), reduced.data(), count, MPI_DOUBLE, MPI_SUM,
                        comm_) != MPI_SUCCESS)
        throw std::runtime_error("solver MPI reduction failed");
      collective =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - began)
              .count();
      detail::check(pfc::gpuMemcpy(slots, reduced.data(),
                                   sizeof(double) * reduced.size(),
                                   pfc::gpuMemcpyHostToDevice));
    }
    account(count, y.size() != 0, start, collective);
  }
  void dot_store(const Vector &x, const Vector &y, double *slot,
                 const ScalarPoll *poll = nullptr) {
    if (x.size() != y.size()) throw std::invalid_argument("vector size mismatch");
    if (slot == nullptr) throw std::invalid_argument("missing dot slot");
    workspace(y.size());
    const auto start = std::chrono::steady_clock::now();
    Resident<Tag>::dot_store(x.data(), y.data(), y.size(), workspace_.data(), slot,
                             poll);
    account(1, y.size() != 0, start, 0);
  }
  void axpy_slot(Vector &y, const double *a, const Vector &x, double sign = 1,
                 const ScalarPoll *poll = nullptr) const {
    if (x.size() != y.size()) throw std::invalid_argument("vector size mismatch");
    Resident<Tag>::combine_slot(y.data(), a, sign, x.data(), 1, y.size(), poll);
  }
  void scale_inv_slot(Vector &x, const double *a,
                      const ScalarPoll *poll = nullptr) const {
    Resident<Tag>::scale_inv(x.data(), a, x.size(), poll);
  }
  void scale_mul_slot(Vector &x, const double *a,
                      const ScalarPoll *poll = nullptr) const {
    Resident<Tag>::scale_mul(x.data(), a, x.size(), poll);
  }
  void scale_ratio_slot(Vector &x, const double *num, const double *den,
                        const ScalarPoll *poll = nullptr) const {
    Resident<Tag>::scale_ratio(x.data(), num, den, x.size(), poll);
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
