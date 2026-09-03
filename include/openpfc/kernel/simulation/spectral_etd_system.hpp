// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_etd_system.hpp
 * @brief The framework-owned pseudo-spectral ETD1 driver on `SimulationState`.
 *
 * @details
 * One class for every spectral-ETD physics and every memory space:
 *
 *   `SpectralETDSystem<Physics, MemorySpace = HostSpace>`
 *
 * The physics supplies k-space symbols and a device-capable pointwise
 * nonlinearity (see `physics_concepts.hpp`). The driver owns the work fields
 * on the caller's `SimulationState`, prepares the diagonal ETD coefficients
 * once with `for_each_kpoint` + `SpectralExpCoefficientCache`, and advances
 * with an attempt / commit protocol:
 *
 *   attempt(t):  psi_hat = F psi
 *                [psi_mf = F⁻¹(χ psi_hat)]        if Physics::filter_mf
 *                [p_star = F⁻¹(P psi_hat)]        if Physics::correlation_kernel
 *                N = pointwise(psi, psi_mf, p_star, x, t)   (host or device)
 *                N_hat = F N                     [× 2/3-rule mask]
 *                candidate = exp(L dt) psi_hat + M φ₁(L dt) N_hat
 *   commit():    psi_hat ← candidate;  psi = F⁻¹ psi_hat
 *
 * with \f$M(k)\f$ = `Physics::nonlinear_symbol` (default 1). Rejecting an
 * attempt is free: nothing but the candidate scratch has changed. `set_dt`
 * re-prepares the coefficients, so an adaptive controller can drive this
 * system through `attempt` / `commit` / `reject`.
 *
 * All backend-specific work goes through `SpectralETDOps<MemorySpace>`
 * (host: `spectral_etd_ops.hpp`; CUDA/HIP:
 * `runtime/gpu/spectral_etd_ops_gpu.hpp`). Device instantiations additionally
 * need the physics' pointwise functor compiled into one CUDA/HIP translation
 * unit — see `OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE` in
 * `runtime/gpu/spectral_pointwise_gpu.hpp`.
 *
 * Optional observable: when the pointwise functor has
 * `free_energy_density(cell)`, the driver fills the `fe_density` field each
 * attempt and reduces it (`last_free_energy_sum`, `last_free_energy`).
 */

#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/observable_reduce.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace pfc::sim {

/**
 * @brief Field names and switches for @ref SpectralETDSystem.
 *
 * Only the fields the physics needs are allocated (`psi_mf*` for
 * `filter_mf`, `P_*` for `correlation_kernel`, `fe_density` for
 * `free_energy_density`).
 */
struct SpectralETDOptions {
  std::string psi_name{"psi"};
  std::string psi_hat_name{"psi_hat"};
  std::string n_name{"N"};
  std::string n_hat_name{"N_hat"};
  std::string psi_mf_name{"psi_mf"};
  std::string psi_mf_hat_name{"psi_mf_hat"};
  std::string p_star_name{"P_star_psi"};
  std::string p_hat_name{"P_hat"};
  std::string fe_name{"fe_density"};
  bool dealias{false};              ///< Orszag 2/3-rule mask on \f$\hat N\f$
  MPI_Comm comm{MPI_COMM_WORLD};    ///< reduction communicator for observables
};

/**
 * @brief Outcome of one `SpectralETDSystem::attempt`.
 *
 * The candidate lives in system-owned scratch until `commit()` or the next
 * `attempt()`. On success `t1 == t0 + dt`.
 */
struct SpectralStepAttempt {
  double t0{};
  double dt{};
  double t1{};
  bool success{false};
};

/**
 * @brief Pseudo-spectral ETD1 system for one real field and its Fourier hat.
 *
 * @tparam Physics     Models @ref SpectralETDPhysics.
 * @tparam MemorySpace `HostSpace`, `CUDASpace`, or `HIPSpace`.
 */
template <class Physics, class MemorySpace = pfc::HostSpace>
  requires SpectralETDPhysics<Physics>
class SpectralETDSystem {
public:
  using Ops = SpectralETDOps<MemorySpace>;
  using Complex = std::complex<double>;
  using FFT = typename Ops::FFT;
  using RealField = pfc::data::Field<double, MemorySpace>;
  using ComplexField = pfc::data::Field<Complex, MemorySpace>;
  using Pointwise = spectral_pointwise_t<Physics>;
  using memory_space = MemorySpace;

  static constexpr bool has_mean_field = HasMeanFieldFilter<Physics>;
  static constexpr bool has_correlation = HasCorrelationKernel<Physics>;
  static constexpr bool has_nonlinear_symbol = HasNonlinearSymbol<Physics>;
  static constexpr bool has_free_energy = HasFreeEnergyDensity<Pointwise>;

  SpectralETDSystem(Physics physics, FFT &fft, SimulationState &state, double dt,
                    SpectralETDOptions opt = {})
      : m_physics(std::move(physics)), m_fft(fft), m_state(state), m_dt(dt),
        m_opt(std::move(opt)), m_candidate(Ops::make_complex(fft.size_outbox())),
        m_n_masked(Ops::make_complex(m_opt.dealias ? fft.size_outbox() : 0)) {
    if (m_dt <= 0.0) {
      throw std::invalid_argument("SpectralETDSystem: dt must be > 0");
    }
    if (!m_state.has_field(m_opt.psi_name)) {
      throw std::invalid_argument("SpectralETDSystem: primary field '" +
                                  m_opt.psi_name +
                                  "' is missing; call physics.declare_fields first");
    }
    auto &psi = this->psi();
    if (psi.size() != m_fft.size_inbox()) {
      throw std::invalid_argument(
          "SpectralETDSystem: psi.size() != FFT inbox size");
    }
    if (psi.storage_halo() != 0) {
      throw std::invalid_argument(
          "SpectralETDSystem: psi must be an unpadded (halo 0) field");
    }
    allocate_work_fields(psi.domain());
    m_geometry = geometry_of(psi);
    prepare_operators();
  }

  SpectralETDSystem(const SpectralETDSystem &) = delete;
  SpectralETDSystem &operator=(const SpectralETDSystem &) = delete;

  /**
   * @brief Build the k-space diagonals for the current `dt` and upload them.
   *
   * Called by the constructor and by `set_dt`. Cheap to call again after a
   * parameter change on `physics()`.
   */
  void prepare_operators() {
    const auto outbox = m_fft.get_outbox_bounds();
    const auto &dom = psi().domain();
    const std::size_t n = m_fft.size_outbox();

    m_L.assign(n, 0.0);
    std::vector<double> nonlinear_symbol(n, 1.0);
    std::vector<double> filter(has_mean_field ? n : 0, 0.0);
    std::vector<double> kernel(has_correlation ? n : 0, 0.0);
    fft::kspace::for_each_kpoint(
        outbox, dom,
        [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
          const double k_lap = fft::kspace::k_laplacian_value(kx, ky, kz);
          m_L[idx] = m_physics.linear_symbol(k_lap);
          if constexpr (has_nonlinear_symbol) {
            nonlinear_symbol[idx] = m_physics.nonlinear_symbol(k_lap);
          }
          if constexpr (has_mean_field) {
            filter[idx] = m_physics.filter_mf(k_lap);
          }
          if constexpr (has_correlation) {
            kernel[idx] = m_physics.correlation_kernel(k_lap);
          }
        });

    m_cache.ensure(std::span<const double>(m_L), m_dt,
                   integrator::SpectralExpOperatorId{.value = 1},
                   integrator::SpectralExpDtId::from_bits(m_dt),
                   integrator::SpectralExpConfigId{.value = 1});
    const auto phi1 = m_cache.phi1_L();
    m_n_weight.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      m_n_weight[i] = nonlinear_symbol[i] * phi1[i];
    }
    m_filter_host = std::move(filter);
    m_kernel_host = std::move(kernel);

    m_exp_dev = Ops::make_real(n);
    m_n_weight_dev = Ops::make_real(n);
    Ops::upload(m_exp_dev, m_cache.exp_Ldt());
    Ops::upload(m_n_weight_dev, std::span<const double>(m_n_weight));
    if constexpr (has_mean_field) {
      m_filter_dev = Ops::make_real(n);
      Ops::upload(m_filter_dev, std::span<const double>(m_filter_host));
    }
    if constexpr (has_correlation) {
      m_kernel_dev = Ops::make_real(n);
      Ops::upload(m_kernel_dev, std::span<const double>(m_kernel_host));
    }
    if (m_opt.dealias) {
      m_mask_host.assign(n, 0.0);
      fft::kspace::fill_two_thirds_mask(outbox, pfc::domain::get_size(dom),
                                        pfc::domain::get_spacing(dom),
                                        m_mask_host.data(), m_mask_host.size());
      m_mask_dev = Ops::make_real(n);
      Ops::upload(m_mask_dev, std::span<const double>(m_mask_host));
    }
    m_pointwise = m_physics.pointwise();
  }

  /// Change the step size and rebuild the ETD coefficients.
  void set_dt(double dt) {
    if (dt <= 0.0) {
      throw std::invalid_argument("SpectralETDSystem::set_dt: dt must be > 0");
    }
    if (dt != m_dt) {
      m_dt = dt;
      prepare_operators();
    }
  }

  /**
   * @brief Form the ETD1 candidate for `t → t + dt` without touching `psi`.
   */
  SpectralStepAttempt attempt(double t) {
    auto &psi = this->psi();
    auto &psi_hat = complex_field(m_opt.psi_hat_name);
    auto &n_real = real_field(m_opt.n_name);
    auto &n_hat = complex_field(m_opt.n_hat_name);

    Ops::forward(m_fft, psi, psi_hat);

    RealField *psi_mf = nullptr;
    RealField *p_star = nullptr;
    RealField *fe = nullptr;
    if constexpr (has_mean_field) {
      psi_mf = &real_field(m_opt.psi_mf_name);
      auto &psi_mf_hat = complex_field(m_opt.psi_mf_hat_name);
      Ops::multiply(psi_hat, m_filter_dev, psi_mf_hat);
      Ops::backward(m_fft, psi_mf_hat, *psi_mf);
    }
    if constexpr (has_correlation) {
      p_star = &real_field(m_opt.p_star_name);
      auto &p_hat = complex_field(m_opt.p_hat_name);
      Ops::multiply(psi_hat, m_kernel_dev, p_hat);
      Ops::backward(m_fft, p_hat, *p_star);
    }
    if constexpr (has_free_energy) {
      fe = &real_field(m_opt.fe_name);
    }

    Ops::pointwise(m_geometry, t, psi, psi_mf, p_star, n_real, fe, m_pointwise);
    Ops::forward(m_fft, n_real, n_hat);

    if (m_opt.dealias) {
      Ops::multiply(n_hat, m_mask_dev, m_n_masked);
      Ops::combine(psi_hat, m_n_masked, m_exp_dev, m_n_weight_dev, m_candidate);
    } else {
      Ops::combine(psi_hat, n_hat, m_exp_dev, m_n_weight_dev, m_candidate);
    }

    if constexpr (has_free_energy) {
      reduce_free_energy(*fe);
    }
    return SpectralStepAttempt{.t0 = t, .dt = m_dt, .t1 = t + m_dt, .success = true};
  }

  /// Accept the last candidate: `psi_hat ← candidate`, `psi = F⁻¹ psi_hat`.
  void commit() {
    auto &psi_hat = complex_field(m_opt.psi_hat_name);
    Ops::swap(psi_hat, m_candidate);
    Ops::backward(m_fft, psi_hat, psi());
  }

  /// Discard the last candidate (no state was modified).
  void reject() noexcept {}

  /**
   * @brief One ETD1 step (`attempt` + `commit`). Returns `t + dt`.
   */
  double step(double t) {
    const auto a = attempt(t);
    if (!a.success) {
      throw std::runtime_error("SpectralETDSystem: ETD1 attempt failed");
    }
    commit();
    return a.t1;
  }

  // ---- queries ----------------------------------------------------------
  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] const Physics &physics() const noexcept { return m_physics; }
  [[nodiscard]] Physics &physics() noexcept { return m_physics; }
  [[nodiscard]] const Pointwise &pointwise() const noexcept { return m_pointwise; }
  [[nodiscard]] const SpectralETDOptions &options() const noexcept { return m_opt; }
  /// \f$L(k)\f$ on this rank's outbox.
  [[nodiscard]] const std::vector<double> &linear_symbol() const noexcept {
    return m_L;
  }
  /// \f$M(k)\,\varphi_1(L\,dt)\f$ on this rank's outbox.
  [[nodiscard]] const std::vector<double> &nonlinear_weight() const noexcept {
    return m_n_weight;
  }
  /// \f$\chi(k)\f$ (empty unless `Physics::filter_mf`).
  [[nodiscard]] const std::vector<double> &filter_mf() const noexcept {
    return m_filter_host;
  }
  /// \f$P(k)\f$ (empty unless `Physics::correlation_kernel`).
  [[nodiscard]] const std::vector<double> &correlation_kernel() const noexcept {
    return m_kernel_host;
  }
  /// Rank-local sum of the free-energy density after the last attempt.
  [[nodiscard]] double last_free_energy_sum() const noexcept { return m_fe_sum; }
  /// Communicator-wide integral of the free-energy density.
  [[nodiscard]] double last_free_energy() const noexcept { return m_fe_integral; }

  [[nodiscard]] RealField &psi() { return real_field(m_opt.psi_name); }
  [[nodiscard]] const RealField &psi() const {
    return m_state.template get_field<double, MemorySpace>(m_opt.psi_name);
  }

private:
  RealField &real_field(const std::string &name) {
    return m_state.template get_field<double, MemorySpace>(name);
  }
  ComplexField &complex_field(const std::string &name) {
    return m_state.template get_field<Complex, MemorySpace>(name);
  }

  static PointwiseGeometry geometry_of(const RealField &f) {
    const auto &o = f.origin();
    const auto &s = f.spacing();
    const auto &box = f.box();
    return PointwiseGeometry{.nx = box.size[0],
                             .ny = box.size[1],
                             .nz = box.size[2],
                             .low_x = box.low[0],
                             .low_y = box.low[1],
                             .low_z = box.low[2],
                             .origin_x = o[0],
                             .origin_y = o[1],
                             .origin_z = o[2],
                             .dx = s[0],
                             .dy = s[1],
                             .dz = s[2]};
  }

  void allocate_work_fields(const Domain &domain) {
    const auto inbox = m_fft.get_inbox_bounds();
    const auto outbox = m_fft.get_outbox_bounds();
    add_if_missing<double>(m_opt.n_name, domain, inbox);
    add_if_missing<Complex>(m_opt.psi_hat_name, domain, outbox);
    add_if_missing<Complex>(m_opt.n_hat_name, domain, outbox);
    if constexpr (has_mean_field) {
      add_if_missing<double>(m_opt.psi_mf_name, domain, inbox);
      add_if_missing<Complex>(m_opt.psi_mf_hat_name, domain, outbox);
    }
    if constexpr (has_correlation) {
      add_if_missing<double>(m_opt.p_star_name, domain, inbox);
      add_if_missing<Complex>(m_opt.p_hat_name, domain, outbox);
    }
    if constexpr (has_free_energy) {
      add_if_missing<double>(m_opt.fe_name, domain, inbox);
    }
  }

  template <class T>
  void add_if_missing(const std::string &name, const Domain &domain,
                      const Box3i &box) {
    if (m_state.has_field(name)) {
      return;
    }
    add_declared_field<T, MemorySpace>(m_state, name, domain, box, 0);
  }

  void reduce_free_energy(RealField &fe) {
    m_fe_sum = sum_owned(fe);
    int mpi_ready = 0;
    int mpi_done = 0;
    MPI_Initialized(&mpi_ready);
    MPI_Finalized(&mpi_done);
    if (mpi_ready != 0 && mpi_done == 0 && m_opt.comm != MPI_COMM_NULL) {
      m_fe_integral = integrate_owned(fe, m_opt.comm);
    } else {
      m_fe_integral = m_fe_sum * cell_volume(fe.domain());
    }
  }

  Physics m_physics;
  FFT &m_fft;
  SimulationState &m_state;
  double m_dt{};
  SpectralETDOptions m_opt{};
  PointwiseGeometry m_geometry{};
  Pointwise m_pointwise{};

  std::vector<double> m_L;
  std::vector<double> m_n_weight;
  std::vector<double> m_filter_host;
  std::vector<double> m_kernel_host;
  std::vector<double> m_mask_host;
  integrator::SpectralExpCoefficientCache<> m_cache{};

  typename Ops::real_coeffs m_exp_dev{};
  typename Ops::real_coeffs m_n_weight_dev{};
  typename Ops::real_coeffs m_filter_dev{};
  typename Ops::real_coeffs m_kernel_dev{};
  typename Ops::real_coeffs m_mask_dev{};
  typename Ops::complex_scratch m_candidate;
  typename Ops::complex_scratch m_n_masked;

  double m_fe_sum{0.0};
  double m_fe_integral{0.0};
};

} // namespace pfc::sim
