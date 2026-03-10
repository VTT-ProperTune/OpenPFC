// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_ops.hpp
 * @brief Tungsten-specific element-wise operations
 *
 * @details
 * This file provides backend-agnostic element-wise operations specific to the
 * Tungsten Phase Field Crystal model. These operations are used in the Tungsten
 * model's time-stepping algorithm.
 *
 * Operations include:
 * - Complex field multiplication (for mean-field filtering)
 * - Nonlinear term computation (Tungsten-specific: p3*u² + p4*u³ + q3*v² + q4*v³)
 * - Stabilization operations (Tungsten-specific)
 * - Time integration operators (Tungsten-specific exponential integration)
 *
 * @note These operations are Tungsten-specific and should not be used by other
 * models. For generic operations, see openpfc/utils/ or create model-specific
 * operations.
 */

#pragma once

#include <complex>
#include <openpfc/core/backend_tags.hpp>
#include <openpfc/core/databuffer.hpp>

namespace tungsten {
namespace ops {

/**
 * @brief Element-wise multiplication: out = a * b (complex * real)
 *
 * Computes the element-wise product of a complex field and a real field.
 * Used for applying mean-field filters in Fourier space.
 *
 * @tparam BackendTag Backend tag (CpuTag, CudaTag, etc.)
 * @tparam RealType Real number type (float or double)
 * @param a Complex field (input)
 * @param b Real field (input)
 * @param out Complex field (output, must be pre-allocated)
 *
 * @pre a.size() == b.size() == out.size()
 */
template <typename BackendTag, typename RealType = double>
void multiply_complex_real(
    const pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &a,
    const pfc::core::DataBuffer<BackendTag, RealType> &b,
    pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &out);

/**
 * @brief Compute nonlinear term: out = p3*u² + p4*u³ + q3*v² + q4*v³
 *
 * Computes the Tungsten-specific nonlinear term in real space.
 *
 * @tparam BackendTag Backend tag (CpuTag, CudaTag, etc.)
 * @tparam RealType Real number type (float or double)
 * @param u First field (input)
 * @param v Second field (input)
 * @param p3 Coefficient for u²
 * @param p4 Coefficient for u³
 * @param q3 Coefficient for v²
 * @param q4 Coefficient for v³
 * @param out Output field (must be pre-allocated)
 *
 * @pre u.size() == v.size() == out.size()
 */
template <typename BackendTag, typename RealType = double>
void compute_nonlinear(const pfc::core::DataBuffer<BackendTag, RealType> &u,
                       const pfc::core::DataBuffer<BackendTag, RealType> &v,
                       RealType p3, RealType p4, RealType q3, RealType q4,
                       pfc::core::DataBuffer<BackendTag, RealType> &out);

/**
 * @brief Apply stabilization: out = in - stabP * field
 *
 * Applies a Tungsten-specific stabilization term to a field.
 *
 * @tparam BackendTag Backend tag (CpuTag, CudaTag, etc.)
 * @tparam RealType Real number type (float or double)
 * @param in Input field
 * @param field Field to subtract
 * @param stabP Stabilization parameter
 * @param out Output field (must be pre-allocated)
 *
 * @pre in.size() == field.size() == out.size()
 */
template <typename BackendTag, typename RealType = double>
void apply_stabilization(const pfc::core::DataBuffer<BackendTag, RealType> &in,
                         const pfc::core::DataBuffer<BackendTag, RealType> &field,
                         RealType stabP,
                         pfc::core::DataBuffer<BackendTag, RealType> &out);

/**
 * @brief Apply exponential time integration: out = opL * psi_F + opN * psiN_F
 *
 * Applies the Tungsten-specific exponential time integration operator in Fourier
 * space.
 *
 * @tparam BackendTag Backend tag (CpuTag, CudaTag, etc.)
 * @tparam RealType Real number type (float or double)
 * @param psi_F Current state in Fourier space
 * @param psiN_F Nonlinear term in Fourier space
 * @param opL Linear operator (precomputed)
 * @param opN Nonlinear operator (precomputed)
 * @param out Output field (must be pre-allocated)
 *
 * @pre psi_F.size() == psiN_F.size() == opL.size() == opN.size() == out.size()
 */
template <typename BackendTag, typename RealType = double>
void apply_time_integration(
    const pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &psi_F,
    const pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &psiN_F,
    const pfc::core::DataBuffer<BackendTag, RealType> &opL,
    const pfc::core::DataBuffer<BackendTag, RealType> &opN,
    pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &out);

// Implementation details using helper structs for template specialization
namespace detail {

template <typename BackendTag, typename RealType = double> struct TungstenOps;

// CPU specialization for double precision
template <> struct TungstenOps<pfc::backend::CpuTag, double> {
  static void multiply_complex_real_impl(
      const pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<double>> &a,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, double> &b,
      pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<double>> &out) {
    const size_t N = a.size();
    if (b.size() != N || out.size() != N) {
      throw std::runtime_error("Size mismatch in multiply_complex_real");
    }
    for (size_t i = 0; i < N; ++i) {
      out[i] = a[i] * b[i];
    }
  }

  static void compute_nonlinear_impl(
      const pfc::core::DataBuffer<pfc::backend::CpuTag, double> &u,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, double> &v, double p3,
      double p4, double q3, double q4,
      pfc::core::DataBuffer<pfc::backend::CpuTag, double> &out) {
    const size_t N = u.size();
    if (v.size() != N || out.size() != N) {
      throw std::runtime_error("Size mismatch in compute_nonlinear");
    }
    for (size_t i = 0; i < N; ++i) {
      double u_val = u[i];
      double v_val = v[i];
      double u2 = u_val * u_val;
      double u3 = u2 * u_val;
      double v2 = v_val * v_val;
      double v3 = v2 * v_val;
      out[i] = p3 * u2 + p4 * u3 + q3 * v2 + q4 * v3;
    }
  }

  static void apply_stabilization_impl(
      const pfc::core::DataBuffer<pfc::backend::CpuTag, double> &in,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, double> &field, double stabP,
      pfc::core::DataBuffer<pfc::backend::CpuTag, double> &out) {
    const size_t N = in.size();
    if (field.size() != N || out.size() != N) {
      throw std::runtime_error("Size mismatch in apply_stabilization");
    }
    for (size_t i = 0; i < N; ++i) {
      out[i] = in[i] - stabP * field[i];
    }
  }

  static void apply_time_integration_impl(
      const pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<double>> &psi_F,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<double>>
          &psiN_F,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, double> &opL,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, double> &opN,
      pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<double>> &out) {
    const size_t N = psi_F.size();
    if (psiN_F.size() != N || opL.size() != N || opN.size() != N ||
        out.size() != N) {
      throw std::runtime_error("Size mismatch in apply_time_integration");
    }
    for (size_t i = 0; i < N; ++i) {
      out[i] = opL[i] * psi_F[i] + opN[i] * psiN_F[i];
    }
  }
};

// CPU specialization for float precision
template <> struct TungstenOps<pfc::backend::CpuTag, float> {
  static void multiply_complex_real_impl(
      const pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<float>> &a,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, float> &b,
      pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<float>> &out) {
    const size_t N = a.size();
    if (b.size() != N || out.size() != N) {
      throw std::runtime_error("Size mismatch in multiply_complex_real");
    }
    for (size_t i = 0; i < N; ++i) {
      out[i] = a[i] * b[i];
    }
  }

  static void
  compute_nonlinear_impl(const pfc::core::DataBuffer<pfc::backend::CpuTag, float> &u,
                         const pfc::core::DataBuffer<pfc::backend::CpuTag, float> &v,
                         float p3, float p4, float q3, float q4,
                         pfc::core::DataBuffer<pfc::backend::CpuTag, float> &out) {
    const size_t N = u.size();
    if (v.size() != N || out.size() != N) {
      throw std::runtime_error("Size mismatch in compute_nonlinear");
    }
    for (size_t i = 0; i < N; ++i) {
      float u_val = u[i];
      float v_val = v[i];
      float u2 = u_val * u_val;
      float u3 = u2 * u_val;
      float v2 = v_val * v_val;
      float v3 = v2 * v_val;
      out[i] = p3 * u2 + p4 * u3 + q3 * v2 + q4 * v3;
    }
  }

  static void apply_stabilization_impl(
      const pfc::core::DataBuffer<pfc::backend::CpuTag, float> &in,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, float> &field, float stabP,
      pfc::core::DataBuffer<pfc::backend::CpuTag, float> &out) {
    const size_t N = in.size();
    if (field.size() != N || out.size() != N) {
      throw std::runtime_error("Size mismatch in apply_stabilization");
    }
    for (size_t i = 0; i < N; ++i) {
      out[i] = in[i] - stabP * field[i];
    }
  }

  static void apply_time_integration_impl(
      const pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<float>> &psi_F,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<float>> &psiN_F,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, float> &opL,
      const pfc::core::DataBuffer<pfc::backend::CpuTag, float> &opN,
      pfc::core::DataBuffer<pfc::backend::CpuTag, std::complex<float>> &out) {
    const size_t N = psi_F.size();
    if (psiN_F.size() != N || opL.size() != N || opN.size() != N ||
        out.size() != N) {
      throw std::runtime_error("Size mismatch in apply_time_integration");
    }
    for (size_t i = 0; i < N; ++i) {
      out[i] = opL[i] * psi_F[i] + opN[i] * psiN_F[i];
    }
  }
};

// CUDA specialization (only when CUDA is enabled)
// Forward declarations - implementations in tungsten_ops_kernels.cu
#if defined(OpenPFC_ENABLE_CUDA)
template <> struct TungstenOps<pfc::backend::CudaTag, double> {
  static void multiply_complex_real_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &a,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &b,
      pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &out);

  static void compute_nonlinear_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &u,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &v, double p3,
      double p4, double q3, double q4,
      pfc::core::DataBuffer<pfc::backend::CudaTag, double> &out);

  static void apply_stabilization_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &in,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &field,
      double stabP, pfc::core::DataBuffer<pfc::backend::CudaTag, double> &out);

  static void apply_time_integration_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>>
          &psi_F,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>>
          &psiN_F,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &opL,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &opN,
      pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &out);
};

template <> struct TungstenOps<pfc::backend::CudaTag, float> {
  static void multiply_complex_real_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &a,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &b,
      pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &out);

  static void compute_nonlinear_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &u,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &v, float p3,
      float p4, float q3, float q4,
      pfc::core::DataBuffer<pfc::backend::CudaTag, float> &out);

  static void apply_stabilization_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &in,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &field, float stabP,
      pfc::core::DataBuffer<pfc::backend::CudaTag, float> &out);

  static void apply_time_integration_impl(
      const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &psi_F,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>>
          &psiN_F,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &opL,
      const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &opN,
      pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &out);
};
#endif // OpenPFC_ENABLE_CUDA

} // namespace detail

// Template function implementations that dispatch to helper structs
template <typename BackendTag, typename RealType>
void multiply_complex_real(
    const pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &a,
    const pfc::core::DataBuffer<BackendTag, RealType> &b,
    pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &out) {
  detail::TungstenOps<BackendTag, RealType>::multiply_complex_real_impl(a, b, out);
}

template <typename BackendTag, typename RealType>
void compute_nonlinear(const pfc::core::DataBuffer<BackendTag, RealType> &u,
                       const pfc::core::DataBuffer<BackendTag, RealType> &v,
                       RealType p3, RealType p4, RealType q3, RealType q4,
                       pfc::core::DataBuffer<BackendTag, RealType> &out) {
  detail::TungstenOps<BackendTag, RealType>::compute_nonlinear_impl(u, v, p3, p4, q3,
                                                                    q4, out);
}

template <typename BackendTag, typename RealType>
void apply_stabilization(const pfc::core::DataBuffer<BackendTag, RealType> &in,
                         const pfc::core::DataBuffer<BackendTag, RealType> &field,
                         RealType stabP,
                         pfc::core::DataBuffer<BackendTag, RealType> &out) {
  detail::TungstenOps<BackendTag, RealType>::apply_stabilization_impl(in, field,
                                                                      stabP, out);
}

template <typename BackendTag, typename RealType>
void apply_time_integration(
    const pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &psi_F,
    const pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &psiN_F,
    const pfc::core::DataBuffer<BackendTag, RealType> &opL,
    const pfc::core::DataBuffer<BackendTag, RealType> &opN,
    pfc::core::DataBuffer<BackendTag, std::complex<RealType>> &out) {
  detail::TungstenOps<BackendTag, RealType>::apply_time_integration_impl(
      psi_F, psiN_F, opL, opN, out);
}

} // namespace ops
} // namespace tungsten
