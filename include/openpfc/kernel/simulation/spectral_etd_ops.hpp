// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_etd_ops.hpp
 * @brief Memory-space operations behind `SpectralETDSystem`.
 *
 * @details
 * `SpectralETDSystem<Physics, MemorySpace>` is written once against this
 * policy. The primary template is only declared; each memory space provides
 * a specialization:
 *
 * - `HostSpace` — defined here (std::vector coefficients, host loops,
 *   `IHostFFT`).
 * - `CUDASpace` / `HIPSpace` — `runtime/gpu/spectral_etd_ops_gpu.hpp`
 *   (device coefficient buffers, precompiled elementwise kernels,
 *   `IDeviceFFT`, and the device pointwise launcher).
 *
 * Every operation that writes a `Field` stamps its residency, so the system
 * never touches `note_*_write` by hand. This mirrors how `DataBuffer`
 * specializations are injected from `runtime/` (kernel never includes
 * runtime).
 */

#include <complex>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace pfc::sim {

/**
 * @brief Per-memory-space operations for the spectral ETD driver.
 *
 * Required interface (see the `HostSpace` specialization for reference):
 *
 * - `using FFT`, `using real_coeffs`, `using complex_scratch`;
 * - `make_real(n)`, `make_complex(n)`, `upload(real_coeffs&, span)`;
 * - `forward(fft, RealField&, ComplexField&)`, `backward(fft, ComplexField
 *   const&, RealField&)`, `backward(fft, complex_scratch const&, RealField&)`;
 * - `multiply(ComplexField const&, real_coeffs const&, ComplexField&)` and
 *   the `complex_scratch` output overload;
 * - `combine(ComplexField const& u, X const& n_hat, real_coeffs const& exp_Ldt,
 *   real_coeffs const& n_weight, complex_scratch& out)` for `X` in
 *   {`ComplexField`, `complex_scratch`};
 * - `swap(ComplexField&, complex_scratch&)`;
 * - `pointwise(geometry, t, psi, psi_mf*, p_star*, n, fe*, functor)`.
 */
template <class MemorySpace> struct SpectralETDOps;

template <> struct SpectralETDOps<pfc::HostSpace> {
  using Complex = std::complex<double>;
  using FFT = fft::IHostFFT;
  using RealField = pfc::data::Field<double, pfc::HostSpace>;
  using ComplexField = pfc::data::Field<Complex, pfc::HostSpace>;
  using real_coeffs = std::vector<double>;
  using complex_scratch = std::vector<Complex>;

  static real_coeffs make_real(std::size_t n) { return real_coeffs(n, 0.0); }
  static complex_scratch make_complex(std::size_t n) { return complex_scratch(n); }
  static void upload(real_coeffs &dst, std::span<const double> src) {
    dst.assign(src.begin(), src.end());
  }

  static void forward(FFT &fft, RealField &in, ComplexField &out) {
    fft.forward(in.vec(), out.vec());
    out.note_host_write();
  }
  static void backward(FFT &fft, const ComplexField &in, RealField &out) {
    fft.backward(in.vec(), out.vec());
    out.note_host_write();
  }
  static void backward(FFT &fft, const complex_scratch &in, RealField &out) {
    fft.backward(in, out.vec());
    out.note_host_write();
  }

  static void multiply(const ComplexField &in, const real_coeffs &w,
                       ComplexField &out) {
    multiply_raw(in.data(), w.data(), out.data(), in.size());
    out.note_host_write();
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

  /// Commit: the candidate becomes the field's storage (O(1)).
  static void swap(ComplexField &field, complex_scratch &candidate) {
    std::swap(field.vec(), candidate);
    field.note_host_write();
  }

  template <class F>
    requires SpectralPointwise<F>
  static void pointwise(const PointwiseGeometry &g, double t, RealField &psi,
                        RealField *psi_mf, RealField *p_star, RealField &n,
                        RealField *fe, const F &f) {
    for_each_spectral_cell(g, t, psi.data(),
                           psi_mf != nullptr ? psi_mf->data() : nullptr,
                           p_star != nullptr ? p_star->data() : nullptr, n.data(),
                           fe != nullptr ? fe->data() : nullptr, f);
    n.note_host_write();
    if (fe != nullptr) {
      fe->note_host_write();
    }
  }

private:
  static void multiply_raw(const Complex *in, const double *w, Complex *out,
                           std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = w[i] * in[i];
    }
  }
  static void combine_raw(const Complex *u, const Complex *n_hat, const double *e,
                          const double *w, Complex *out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = e[i] * u[i] + w[i] * n_hat[i];
    }
  }
};

} // namespace pfc::sim
