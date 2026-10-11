// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <array>
#include <complex>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/vector_operations.hpp>
namespace pfc::finite_strain::gpu {
struct ProjectionGrid {
  int shape[3], low[3], extent[3];
  double spacing[3];
};
struct ComplexComponents {
  double *values[9];
};
#if defined(OpenPFC_ENABLE_CUDA)
void project_tensor_cuda(ComplexComponents, ProjectionGrid, std::size_t);
void apply_tangents_cuda(const double *, const double *, double *, std::size_t);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void project_tensor_hip(ComplexComponents, ProjectionGrid, std::size_t);
void apply_tangents_hip(const double *, const double *, double *, std::size_t);
#endif
/// Uses the existing FFT interface, including distributed HeFFTe device FFTs.
/// Odd grid only: identical compatibility and zero-mode convention to the host.
template <class MemorySpace> class CompatibilityProjector {
public:
  using FFT = fft::IDeviceFFT<MemorySpace>;
  using Tag = typename FFT::backend_tag;
  using Vector = typename FFT::RealBuffer;

private:
  FFT &fft_;
  ProjectionGrid grid_{};
  Vector real_;
  std::array<typename FFT::ComplexBuffer, 9> hat_;

public:
  CompatibilityProjector(FFT &fft, std::array<int, 3> shape,
                         std::array<double, 3> spacing)
      : fft_(fft), real_(fft.size_inbox()) {
    auto box = fft.get_outbox_bounds();
    for (int d = 0; d < 3; ++d) {
      if (shape[d] < 1 || shape[d] % 2 == 0 || !(spacing[d] > 0))
        throw std::invalid_argument("device compatibility requires odd grids");
      grid_.shape[d] = shape[d];
      grid_.spacing[d] = spacing[d];
      grid_.low[d] = box.low[d];
      grid_.extent[d] = box.size[d];
    }
    for (auto &h : hat_) h = typename FFT::ComplexBuffer(fft.size_outbox());
  }
  std::size_t count() const { return fft_.size_inbox(); }
  void apply(const Vector &input, Vector &out) {
    if (input.size() != 9 * count())
      throw std::invalid_argument("tensor field size mismatch");
    if (out.size() != input.size()) out.resize(input.size());
    auto copy = [](void *dst, const void *src, std::size_t bytes) {
      solvers::gpu::detail::check(pfc::gpuMemcpyAsync(
          dst, src, bytes, pfc::gpuMemcpyDeviceToDevice, nullptr));
    };
    for (int c = 0; c < 9; ++c) {
      copy(real_.data(), input.data() + c * count(), count() * sizeof(double));
      fft_.forward(real_, hat_[c]);
    }
    ComplexComponents fields;
    for (int c = 0; c < 9; ++c)
      fields.values[c] = reinterpret_cast<double *>(hat_[c].data());
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      project_tensor_cuda(fields, grid_, fft_.size_outbox());
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      project_tensor_hip(fields, grid_, fft_.size_outbox());
#endif
    for (int c = 0; c < 9; ++c) {
      fft_.backward(hat_[c], real_);
      copy(out.data() + c * count(), real_.data(), count() * sizeof(double));
    }
  }
  void tangent(const Vector &K, const Vector &v, Vector &w) {
    if (K.size() != 81 * count() || v.size() != 9 * count())
      throw std::invalid_argument("tangent field size mismatch");
    if (w.size() != v.size()) w.resize(v.size());
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      apply_tangents_cuda(K.data(), v.data(), w.data(), count());
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      apply_tangents_hip(K.data(), v.data(), w.data(), count());
#endif
    apply(w, w);
  }
};
} // namespace pfc::finite_strain::gpu
