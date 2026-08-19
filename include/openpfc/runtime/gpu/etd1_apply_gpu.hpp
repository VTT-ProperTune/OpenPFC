// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file etd1_apply_gpu.hpp
 * @brief Device-resident ETD1 combine: `out = exp_Ldt * u + phi1_L * N`
 *
 * @details
 * Pointers must be device memory. Real coefficients (`exp_Ldt`, `phi1_L`)
 * scale complex or real fields. Host combine is
 * `pfc::integrator::apply_etd1_update`.
 *
 * @see kernel/integrator/etd1_apply.hpp
 * @see runtime/gpu/elementwise_ops_gpu.hpp
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <complex>
#include <cstddef>

#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>

namespace pfc::integrator {

#if defined(OpenPFC_ENABLE_CUDA)
inline void apply_etd1_update_cuda(const std::complex<double> *u,
                                   const std::complex<double> *n_of_u,
                                   const double *exp_Ldt, const double *phi1_L,
                                   std::complex<double> *candidate,
                                   std::size_t n) {
  pfc::combine_two_term_cuda_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}

inline void apply_etd1_update_cuda(const std::complex<float> *u,
                                   const std::complex<float> *n_of_u,
                                   const float *exp_Ldt, const float *phi1_L,
                                   std::complex<float> *candidate,
                                   std::size_t n) {
  pfc::combine_two_term_cuda_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}

inline void apply_etd1_update_cuda(const double *u, const double *n_of_u,
                                   const double *exp_Ldt, const double *phi1_L,
                                   double *candidate, std::size_t n) {
  pfc::combine_two_term_cuda_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}

inline void apply_etd1_update_cuda(const float *u, const float *n_of_u,
                                   const float *exp_Ldt, const float *phi1_L,
                                   float *candidate, std::size_t n) {
  pfc::combine_two_term_cuda_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
inline void apply_etd1_update_hip(const std::complex<double> *u,
                                  const std::complex<double> *n_of_u,
                                  const double *exp_Ldt, const double *phi1_L,
                                  std::complex<double> *candidate,
                                  std::size_t n) {
  pfc::combine_two_term_hip_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}

inline void apply_etd1_update_hip(const std::complex<float> *u,
                                  const std::complex<float> *n_of_u,
                                  const float *exp_Ldt, const float *phi1_L,
                                  std::complex<float> *candidate,
                                  std::size_t n) {
  pfc::combine_two_term_hip_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}

inline void apply_etd1_update_hip(const double *u, const double *n_of_u,
                                  const double *exp_Ldt, const double *phi1_L,
                                  double *candidate, std::size_t n) {
  pfc::combine_two_term_hip_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}

inline void apply_etd1_update_hip(const float *u, const float *n_of_u,
                                  const float *exp_Ldt, const float *phi1_L,
                                  float *candidate, std::size_t n) {
  pfc::combine_two_term_hip_impl(u, n_of_u, exp_Ldt, phi1_L, candidate, n);
}
#endif

} // namespace pfc::integrator

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
