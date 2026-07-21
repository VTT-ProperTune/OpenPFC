// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_etd_workspace.hpp
 * @brief Method-owned Tungsten ETD coefficient workspace over SpectralExpCoefficientCache
 *
 * @details
 * Owns diagonal @c L = k_laplacian * opCk samples, a
 * @ref pfc::integrator::SpectralExpCoefficientCache, and derived apply weights
 * @c n_weight[i] = k_laplacian[i] * phi1_L[i] (historically @c opN).
 *
 * Transient / recomputable — **not** registered as Model checkpoint fields.
 * Rebuild via @ref ensure when operator / dt / config identity changes.
 *
 * TODO(remove-tungsten-etd-workspace): replace with Etd1Stepper after #169
 * (wire driver-owned time advance through App/Simulator; drop this adapter).
 */

#pragma once

#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/runtime/hip/backend_tags_hip.hpp>
#include <openpfc/runtime/hip/databuffer_hip.hpp>
#endif

namespace tungsten {
namespace etd {

/**
 * @brief Method-owned ETD weights for Tungsten spectral advance.
 *
 * @tparam RealType Storage type for optional device weight buffers (float/double).
 *         Host cache always uses @c double via SpectralExpCoefficientCache.
 *
 * TODO(remove-tungsten-etd-workspace): replace with Etd1Stepper after #169
 */
template <typename RealType = double> class TungstenEtdWorkspace {
public:
  /**
   * @brief Reserve host scratch for @p n_modes modes.
   *
   * Does not register buffers as Model fields. Device buffers are allocated
   * separately via @ref allocate_cuda / @ref allocate_hip.
   */
  void reserve(std::size_t n_modes) {
    m_L.resize(n_modes);
    m_k_laplacian.resize(n_modes);
    m_n_weight.resize(n_modes);
#if defined(OpenPFC_ENABLE_CUDA)
    m_cuda_ready = false;
#endif
#if defined(OpenPFC_ENABLE_HIP)
    m_hip_ready = false;
#endif
  }

#if defined(OpenPFC_ENABLE_CUDA)
  /** Allocate CUDA device weight buffers (call from TungstenCUDA::allocate). */
  void allocate_cuda(std::size_t n_modes) {
    m_cuda_exp_Ldt =
        pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(n_modes);
    m_cuda_n_weight =
        pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(n_modes);
    m_cuda_ready = false;
  }
#endif

#if defined(OpenPFC_ENABLE_HIP)
  /** Allocate HIP device weight buffers (call from TungstenHIP::allocate). */
  void allocate_hip(std::size_t n_modes) {
    m_hip_exp_Ldt =
        pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(n_modes);
    m_hip_n_weight =
        pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(n_modes);
    m_hip_ready = false;
  }
#endif

  /**
   * @brief Build @c L = k_lap*opCk, ensure cache coeffs, derive @c n_weight.
   *
   * @param k_laplacian Per-mode Laplacian symbols.
   * @param opCk Per-mode physics linear symbols from @c physics_for_mode.
   * @param dt Timestep.
   * @param op_id Operator identity (bump when physics params change).
   * @param config_id Configuration identity (scheme / mapping flags).
   */
  void ensure(std::span<const double> k_laplacian, std::span<const double> opCk,
              double dt, pfc::integrator::SpectralExpOperatorId op_id,
              pfc::integrator::SpectralExpConfigId config_id) {
    if (k_laplacian.size() != opCk.size()) {
      throw std::invalid_argument(
          "TungstenEtdWorkspace::ensure: k_laplacian and opCk size mismatch");
    }
    const std::size_t n = k_laplacian.size();
    if (m_L.size() != n) {
      reserve(n);
    }

    for (std::size_t i = 0; i < n; ++i) {
      m_k_laplacian[i] = k_laplacian[i];
      m_L[i] = k_laplacian[i] * opCk[i];
    }

    m_cache.ensure(m_L, dt, op_id,
                   pfc::integrator::SpectralExpDtId::from_bits(dt), config_id);

    const auto phi1 = m_cache.phi1_L();
    for (std::size_t i = 0; i < n; ++i) {
      m_n_weight[i] = m_k_laplacian[i] * phi1[i];
    }

#if defined(OpenPFC_ENABLE_CUDA)
    m_cuda_ready = false;
#endif
#if defined(OpenPFC_ENABLE_HIP)
    m_hip_ready = false;
#endif
  }

  /**
   * @brief CPU ETD combine: @c psi_F = exp_Ldt*psi_F + n_weight*psiN_F.
   *
   * In-place on @p psi_F.
   */
  void apply_etd(std::span<std::complex<double>> psi_F,
                 std::span<const std::complex<double>> psiN_F) const {
    const auto exp_w = exp_Ldt();
    if (psi_F.size() != psiN_F.size() || psi_F.size() != exp_w.size() ||
        psi_F.size() != m_n_weight.size()) {
      throw std::invalid_argument(
          "TungstenEtdWorkspace::apply_etd: span size mismatch");
    }
    for (std::size_t i = 0; i < psi_F.size(); ++i) {
      psi_F[i] = exp_w[i] * psi_F[i] + m_n_weight[i] * psiN_F[i];
    }
  }

  [[nodiscard]] std::span<const double> exp_Ldt() const noexcept {
    return m_cache.exp_Ldt();
  }

  [[nodiscard]] std::span<const double> n_weight() const noexcept {
    return m_n_weight;
  }

  [[nodiscard]] bool cache_valid() const noexcept { return m_cache.valid(); }

  /// Host coefficient scratch bytes (transient; not checkpointed).
  /// Counts L, k_laplacian, n_weight, and the two cache vectors (exp/phi1).
  [[nodiscard]] std::size_t host_scratch_bytes() const noexcept {
    const std::size_t n = m_L.size();
    return 5 * n * sizeof(double);
  }

#if defined(OpenPFC_ENABLE_CUDA)
  /** Upload host weights to CUDA device buffers (call after @ref ensure). */
  void upload_cuda() {
    const auto exp_w = exp_Ldt();
    if (m_cuda_exp_Ldt.size() != exp_w.size()) {
      m_cuda_exp_Ldt =
          pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(exp_w.size());
      m_cuda_n_weight =
          pfc::core::DataBuffer<pfc::backend::CudaTag, RealType>(exp_w.size());
    }
    std::vector<RealType> exp_h(exp_w.size());
    std::vector<RealType> n_h(m_n_weight.size());
    for (std::size_t i = 0; i < exp_w.size(); ++i) {
      exp_h[i] = static_cast<RealType>(exp_w[i]);
      n_h[i] = static_cast<RealType>(m_n_weight[i]);
    }
    m_cuda_exp_Ldt.copy_from_host(exp_h);
    m_cuda_n_weight.copy_from_host(n_h);
    m_cuda_ready = true;
  }

  [[nodiscard]] pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> &
  cuda_exp_Ldt() {
    if (!m_cuda_ready) {
      throw std::runtime_error(
          "TungstenEtdWorkspace: call upload_cuda() after ensure()");
    }
    return m_cuda_exp_Ldt;
  }

  [[nodiscard]] pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> &
  cuda_n_weight() {
    if (!m_cuda_ready) {
      throw std::runtime_error(
          "TungstenEtdWorkspace: call upload_cuda() after ensure()");
    }
    return m_cuda_n_weight;
  }

  [[nodiscard]] std::size_t cuda_scratch_bytes() const noexcept {
    return (m_cuda_exp_Ldt.size() + m_cuda_n_weight.size()) * sizeof(RealType);
  }
#endif

#if defined(OpenPFC_ENABLE_HIP)
  /** Upload host weights to HIP device buffers (call after @ref ensure). */
  void upload_hip() {
    const auto exp_w = exp_Ldt();
    if (m_hip_exp_Ldt.size() != exp_w.size()) {
      m_hip_exp_Ldt =
          pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(exp_w.size());
      m_hip_n_weight =
          pfc::core::DataBuffer<pfc::backend::HipTag, RealType>(exp_w.size());
    }
    std::vector<RealType> exp_h(exp_w.size());
    std::vector<RealType> n_h(m_n_weight.size());
    for (std::size_t i = 0; i < exp_w.size(); ++i) {
      exp_h[i] = static_cast<RealType>(exp_w[i]);
      n_h[i] = static_cast<RealType>(m_n_weight[i]);
    }
    m_hip_exp_Ldt.copy_from_host(exp_h);
    m_hip_n_weight.copy_from_host(n_h);
    m_hip_ready = true;
  }

  [[nodiscard]] pfc::core::DataBuffer<pfc::backend::HipTag, RealType> &
  hip_exp_Ldt() {
    if (!m_hip_ready) {
      throw std::runtime_error(
          "TungstenEtdWorkspace: call upload_hip() after ensure()");
    }
    return m_hip_exp_Ldt;
  }

  [[nodiscard]] pfc::core::DataBuffer<pfc::backend::HipTag, RealType> &
  hip_n_weight() {
    if (!m_hip_ready) {
      throw std::runtime_error(
          "TungstenEtdWorkspace: call upload_hip() after ensure()");
    }
    return m_hip_n_weight;
  }

  [[nodiscard]] std::size_t hip_scratch_bytes() const noexcept {
    return (m_hip_exp_Ldt.size() + m_hip_n_weight.size()) * sizeof(RealType);
  }
#endif

private:
  pfc::integrator::SpectralExpCoefficientCache m_cache;
  std::vector<double> m_L;
  std::vector<double> m_k_laplacian;
  std::vector<double> m_n_weight;

#if defined(OpenPFC_ENABLE_CUDA)
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> m_cuda_exp_Ldt;
  pfc::core::DataBuffer<pfc::backend::CudaTag, RealType> m_cuda_n_weight;
  bool m_cuda_ready{false};
#endif
#if defined(OpenPFC_ENABLE_HIP)
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> m_hip_exp_Ldt;
  pfc::core::DataBuffer<pfc::backend::HipTag, RealType> m_hip_n_weight;
  bool m_hip_ready{false};
#endif
};

/// Tungsten ETD mapping config token (n_weight = k_laplacian * phi1_L).
inline constexpr pfc::integrator::SpectralExpConfigId
    k_tungsten_etd_config_id{.value = 1};

} // namespace etd
} // namespace tungsten
