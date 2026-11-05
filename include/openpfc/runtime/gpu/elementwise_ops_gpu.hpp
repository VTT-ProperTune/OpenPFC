// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file elementwise_ops_gpu.hpp
 * @brief Generic device elementwise ops for CUDA and HIP (M3).
 *
 * Complex×real multiply, two-term diagonal combine
 * (`out = w0 * x0 + w1 * x1`), and axpy-style fill
 * (`out[i] = alpha * x[i] + beta`). Promoted from the mislabeled
 * Tungsten-specific kernels; used by M7's ETD skeleton.
 *
 * Compiled from `src/openpfc/runtime/gpu/elementwise_ops.cu` / `.hip`.
 *
 * @see runtime/gpu/spectral_etd_ops_gpu.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/field/grain_interface.hpp>

#include <complex>
#include <cstddef>

namespace pfc {

#if defined(OpenPFC_ENABLE_CUDA)
void multiply_complex_real_cuda_impl(const std::complex<double> *a, const double *b,
                                     std::complex<double> *out, std::size_t n);
void multiply_complex_real_cuda_impl(const std::complex<float> *a, const float *b,
                                     std::complex<float> *out, std::size_t n);
void combine_two_term_cuda_impl(const std::complex<double> *x0,
                                const std::complex<double> *x1, const double *w0,
                                const double *w1, std::complex<double> *out,
                                std::size_t n);
void combine_two_term_cuda_impl(const std::complex<double> *x0,
                                const std::complex<double> *x1,
                                const std::complex<double> *w0,
                                const std::complex<double> *w1,
                                std::complex<double> *out, std::size_t n);
void combine_two_term_cuda_impl(const std::complex<float> *x0,
                                const std::complex<float> *x1,
                                const std::complex<float> *w0,
                                const std::complex<float> *w1,
                                std::complex<float> *out, std::size_t n);
void combine_two_term_cuda_impl(const std::complex<float> *x0,
                                const std::complex<float> *x1, const float *w0,
                                const float *w1, std::complex<float> *out,
                                std::size_t n);
void combine_two_term_cuda_impl(const double *x0, const double *x1, const double *w0,
                                const double *w1, double *out, std::size_t n);
void combine_two_term_cuda_impl(const float *x0, const float *x1, const float *w0,
                                const float *w1, float *out, std::size_t n);
void axpy_fill_cuda_impl(double *out, const double *x, double alpha, double beta,
                         std::size_t n);
void axpy_fill_cuda_impl(float *out, const float *x, float alpha, float beta,
                         std::size_t n);
void amplitude_pair_etd1_cuda_impl(const std::complex<double> *re_hat,
                                   const std::complex<double> *im_hat,
                                   const std::complex<double> *nre_hat,
                                   const std::complex<double> *nim_hat,
                                   const double *exp_plus, const double *phi_plus,
                                   const double *exp_minus, const double *phi_minus,
                                   std::complex<double> *out_re,
                                   std::complex<double> *out_im, std::size_t n);
void level_set_rate_cuda_impl(const double *phi, const double *vx, const double *vy,
                              const double *vz, const double *vn, const double *phi0,
                              double *rate, int nx, int ny, int nz, double hx,
                              double hy, double hz, int mode);
void level_set_combine3_cuda_impl(const double *x, const double *y, const double *z,
                                  double *out, double alpha, double beta,
                                  double gamma, std::size_t n);
void grain_count_cuda_impl(const int *ids, int *counts, std::size_t n);
void grain_band_cuda_impl(const int *ids,
                          const field::grain_interface::BandSite *band, int n_band,
                          int nx, int ny, int nz, int *mismatches);
#endif

#if defined(OpenPFC_ENABLE_HIP)
void multiply_complex_real_hip_impl(const std::complex<double> *a, const double *b,
                                    std::complex<double> *out, std::size_t n);
void multiply_complex_real_hip_impl(const std::complex<float> *a, const float *b,
                                    std::complex<float> *out, std::size_t n);
void combine_two_term_hip_impl(const std::complex<double> *x0,
                               const std::complex<double> *x1, const double *w0,
                               const double *w1, std::complex<double> *out,
                               std::size_t n);
void combine_two_term_hip_impl(const std::complex<double> *x0,
                               const std::complex<double> *x1,
                               const std::complex<double> *w0,
                               const std::complex<double> *w1,
                               std::complex<double> *out, std::size_t n);
void combine_two_term_hip_impl(const std::complex<float> *x0,
                               const std::complex<float> *x1,
                               const std::complex<float> *w0,
                               const std::complex<float> *w1,
                               std::complex<float> *out, std::size_t n);
void combine_two_term_hip_impl(const std::complex<float> *x0,
                               const std::complex<float> *x1, const float *w0,
                               const float *w1, std::complex<float> *out,
                               std::size_t n);
void combine_two_term_hip_impl(const double *x0, const double *x1, const double *w0,
                               const double *w1, double *out, std::size_t n);
void combine_two_term_hip_impl(const float *x0, const float *x1, const float *w0,
                               const float *w1, float *out, std::size_t n);
void axpy_fill_hip_impl(double *out, const double *x, double alpha, double beta,
                        std::size_t n);
void axpy_fill_hip_impl(float *out, const float *x, float alpha, float beta,
                        std::size_t n);
void amplitude_pair_etd1_hip_impl(const std::complex<double> *re_hat,
                                  const std::complex<double> *im_hat,
                                  const std::complex<double> *nre_hat,
                                  const std::complex<double> *nim_hat,
                                  const double *exp_plus, const double *phi_plus,
                                  const double *exp_minus, const double *phi_minus,
                                  std::complex<double> *out_re,
                                  std::complex<double> *out_im, std::size_t n);
void level_set_rate_hip_impl(const double *phi, const double *vx, const double *vy,
                             const double *vz, const double *vn, const double *phi0,
                             double *rate, int nx, int ny, int nz, double hx,
                             double hy, double hz, int mode);
void level_set_combine3_hip_impl(const double *x, const double *y, const double *z,
                                 double *out, double alpha, double beta,
                                 double gamma, std::size_t n);
void grain_count_hip_impl(const int *ids, int *counts, std::size_t n);
void grain_band_hip_impl(const int *ids,
                         const field::grain_interface::BandSite *band, int n_band,
                         int nx, int ny, int nz, int *mismatches);
#endif

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
