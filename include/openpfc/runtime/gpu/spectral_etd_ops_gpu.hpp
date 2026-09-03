// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_etd_ops_gpu.hpp
 * @brief `SpectralETDOps` specializations for `CUDASpace` and `HIPSpace`.
 *
 * @details
 * Injects the device policy behind `pfc::sim::SpectralETDSystem<Physics,
 * CUDASpace|HIPSpace>`: k-space diagonals live in device `DataBuffer`s, the
 * complex×real multiply and the two-term ETD combine are the precompiled
 * elementwise kernels from `openpfc_gpu_kernels` / `openpfc_hip_kernels`,
 * transforms go through `IDeviceFFT`, and the pointwise nonlinearity is the
 * physics functor launched by `spectral_pointwise_apply` (which the consumer
 * instantiates in one device TU — see `spectral_pointwise_gpu.hpp`).
 *
 * Every operation stamps field residency; callers never call `note_*_write`
 * around these ops. Include this header from any TU that instantiates a
 * device `SpectralETDSystem`; kernel headers never include it.
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <complex>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

namespace pfc::sim {

namespace detail {

/**
 * @brief Shared device policy; `MemorySpace` selects the vendor kernel set.
 */
template <class MemorySpace> struct DeviceSpectralETDOps {
  using Complex = std::complex<double>;
  using FFT = fft::IDeviceFFT<MemorySpace>;
  using RealField = pfc::data::Field<double, MemorySpace>;
  using ComplexField = pfc::data::Field<Complex, MemorySpace>;
  using real_coeffs = typename FFT::RealBuffer;
  using complex_scratch = typename FFT::ComplexBuffer;

  static real_coeffs make_real(std::size_t n) { return real_coeffs(n); }
  static complex_scratch make_complex(std::size_t n) { return complex_scratch(n); }
  static void upload(real_coeffs &dst, std::span<const double> src) {
    dst.copy_from_host(src);
  }

  static void forward(FFT &fft, RealField &in, ComplexField &out) {
    in.sync_to_device();
    fft.forward(in.buffer(), out.buffer());
    out.note_device_write();
  }
  static void backward(FFT &fft, const ComplexField &in, RealField &out) {
    fft.backward(in.buffer(), out.buffer());
    out.note_device_write();
  }
  static void backward(FFT &fft, const complex_scratch &in, RealField &out) {
    fft.backward(in, out.buffer());
    out.note_device_write();
  }

  static void multiply(const ComplexField &in, const real_coeffs &w,
                       ComplexField &out) {
    multiply_raw(in.data(), w.data(), out.data(), in.size());
    out.note_device_write();
  }
  static void multiply(const ComplexField &in, const real_coeffs &w,
                       complex_scratch &out) {
    multiply_raw(in.data(), w.data(), out.data(), in.size());
  }

  static void combine(const ComplexField &u, const ComplexField &n_hat,
                      const real_coeffs &exp_Ldt, const real_coeffs &n_weight,
                      complex_scratch &out) {
    combine_raw(u.data(), n_hat.data(), exp_Ldt.data(), n_weight.data(), out.data(),
                u.size());
  }
  static void combine(const ComplexField &u, const complex_scratch &n_hat,
                      const real_coeffs &exp_Ldt, const real_coeffs &n_weight,
                      complex_scratch &out) {
    combine_raw(u.data(), n_hat.data(), exp_Ldt.data(), n_weight.data(), out.data(),
                u.size());
  }

  /// Commit: the candidate buffer becomes the field's device storage (O(1)).
  static void swap(ComplexField &field, complex_scratch &candidate) {
    std::swap(field.buffer(), candidate);
    field.note_device_write();
  }

  template <class F>
    requires SpectralPointwise<F>
  static void pointwise(const PointwiseGeometry &g, double t, RealField &psi,
                        RealField *psi_mf, RealField *p_star, RealField &n,
                        RealField *fe, const F &f) {
    psi.sync_to_device();
    if (psi_mf != nullptr) {
      psi_mf->sync_to_device();
    }
    if (p_star != nullptr) {
      p_star->sync_to_device();
    }
    pfc::sim::gpu::spectral_pointwise_apply<F>(
        g, t, psi.data(), psi_mf != nullptr ? psi_mf->data() : nullptr,
        p_star != nullptr ? p_star->data() : nullptr, n.data(),
        fe != nullptr ? fe->data() : nullptr, f, gpuStream_t{});
    n.note_device_write();
    if (fe != nullptr) {
      fe->note_device_write();
    }
  }

private:
  static void multiply_raw(const Complex *in, const double *w, Complex *out,
                           std::size_t n) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, pfc::CUDASpace>) {
      pfc::multiply_complex_real_cuda_impl(in, w, out, n);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, pfc::HIPSpace>) {
      pfc::multiply_complex_real_hip_impl(in, w, out, n);
      return;
    }
#endif
  }
  static void combine_raw(const Complex *u, const Complex *n_hat, const double *e,
                          const double *w, Complex *out, std::size_t n) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, pfc::CUDASpace>) {
      pfc::combine_two_term_cuda_impl(u, n_hat, e, w, out, n);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, pfc::HIPSpace>) {
      pfc::combine_two_term_hip_impl(u, n_hat, e, w, out, n);
      return;
    }
#endif
  }
};

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <>
struct SpectralETDOps<pfc::CUDASpace> : detail::DeviceSpectralETDOps<pfc::CUDASpace> {};
#endif
#if defined(OpenPFC_ENABLE_HIP)
template <>
struct SpectralETDOps<pfc::HIPSpace> : detail::DeviceSpectralETDOps<pfc::HIPSpace> {};
#endif

} // namespace pfc::sim

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
