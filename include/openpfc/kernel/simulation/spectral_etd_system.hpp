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

#include <algorithm>
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
  bool dealias{false};           ///< Orszag 2/3-rule mask on \f$\hat N\f$
  MPI_Comm comm{MPI_COMM_WORLD}; ///< reduction communicator for observables
  /// Linear symbols or the pointwise term change with `t`. `step_n` refuses
  /// this: a captured graph would replay the coefficients of the first step.
  bool time_dependent_coefficients{false};
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

/// True when `T::time_dependent_coefficients` is present and true.
template <class T> constexpr bool declares_time_dependent() {
  if constexpr (requires { T::time_dependent_coefficients; }) {
    return static_cast<bool>(T::time_dependent_coefficients);
  } else {
    return false;
  }
}

/**
 * @brief Pseudo-spectral ETD1 system for one real field and its Fourier hat.
 *
 * @tparam Physics     Models `SpectralETDPhysics`.
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
  static constexpr bool has_complex_linear = HasComplexLinearSymbol<Physics>;
  static constexpr bool has_complex_nonlinear = HasComplexNonlinearSymbol<Physics>;
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
      throw std::invalid_argument("SpectralETDSystem: psi.size() != FFT inbox size");
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
  ~SpectralETDSystem() { release_graph(); }

  /**
   * @brief Build the k-space diagonals for the current `dt` and upload them.
   *
   * Called by the constructor and by `set_dt`. Cheap to call again after a
   * parameter change on `physics()`.
   */
  void prepare_operators() {
    release_graph();
    const auto outbox = m_fft.get_outbox_bounds();
    const auto &dom = psi().domain();
    const std::size_t n = m_fft.size_outbox();

    std::vector<double> filter(has_mean_field ? n : 0, 0.0);
    std::vector<double> kernel(has_correlation ? n : 0, 0.0);

    if constexpr (has_complex_linear) {
      m_L_c.assign(n, Complex{0.0, 0.0});
      std::vector<Complex> nonlinear_symbol(n, Complex{1.0, 0.0});
      fft::kspace::for_each_kpoint(
          outbox, dom,
          [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
            m_L_c[idx] = m_physics.linear_symbol(kx, ky, kz);
            if constexpr (has_complex_nonlinear) {
              nonlinear_symbol[idx] = m_physics.nonlinear_symbol(kx, ky, kz);
            } else if constexpr (has_nonlinear_symbol) {
              const double k_lap = fft::kspace::k_laplacian_value(kx, ky, kz);
              nonlinear_symbol[idx] = m_physics.nonlinear_symbol(k_lap);
            }
            if constexpr (has_mean_field) {
              const double k_lap = fft::kspace::k_laplacian_value(kx, ky, kz);
              filter[idx] = m_physics.filter_mf(k_lap);
            }
            if constexpr (has_correlation) {
              const double k_lap = fft::kspace::k_laplacian_value(kx, ky, kz);
              kernel[idx] = m_physics.correlation_kernel(k_lap);
            }
          });
      m_exp_c.resize(n);
      m_n_weight_c.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        const auto c = integrator::spectral_exp_coeffs(m_L_c[i], m_dt);
        m_exp_c[i] = c.exp_Ldt;
        m_n_weight_c[i] = nonlinear_symbol[i] * c.phi1_L;
      }
      m_exp_c_dev = Ops::make_complex(n);
      m_n_weight_c_dev = Ops::make_complex(n);
      Ops::upload(m_exp_c_dev, std::span<const Complex>(m_exp_c));
      Ops::upload(m_n_weight_c_dev, std::span<const Complex>(m_n_weight_c));
    } else {
      m_L.assign(n, 0.0);
      std::vector<double> nonlinear_symbol(n, 1.0);
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
      m_exp_dev = Ops::make_real(n);
      m_n_weight_dev = Ops::make_real(n);
      Ops::upload(m_exp_dev, m_cache.exp_Ldt());
      Ops::upload(m_n_weight_dev, std::span<const double>(m_n_weight));
    }

    m_filter_host = std::move(filter);
    m_kernel_host = std::move(kernel);
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

    note_launch();
    Ops::forward(m_fft, psi, psi_hat);
    ++m_forward_count;

    RealField *psi_mf = nullptr;
    RealField *p_star = nullptr;
    RealField *fe = nullptr;
    if constexpr (has_mean_field) {
      psi_mf = &real_field(m_opt.psi_mf_name);
      auto &psi_mf_hat = complex_field(m_opt.psi_mf_hat_name);
      etd_multiply(psi_hat, m_filter_dev, psi_mf_hat);
      note_launch();
      Ops::backward(m_fft, psi_mf_hat, *psi_mf);
      ++m_backward_count;
    }
    if constexpr (has_correlation) {
      p_star = &real_field(m_opt.p_star_name);
      auto &p_hat = complex_field(m_opt.p_hat_name);
      etd_multiply(psi_hat, m_kernel_dev, p_hat);
      note_launch();
      Ops::backward(m_fft, p_hat, *p_star);
      ++m_backward_count;
    }
    if constexpr (has_free_energy) {
      fe = &real_field(m_opt.fe_name);
    }

    etd_pointwise(m_geometry, t, psi, psi_mf, p_star, n_real, fe, m_pointwise);
    ++m_nonlinear_count;
    note_launch();
    Ops::forward(m_fft, n_real, n_hat);
    ++m_forward_count;

    if (m_opt.dealias) {
      etd_multiply(n_hat, m_mask_dev, m_n_masked);
      if constexpr (has_complex_linear) {
        etd_combine(psi_hat, m_n_masked, m_exp_c_dev, m_n_weight_c_dev, m_candidate);
      } else {
        etd_combine(psi_hat, m_n_masked, m_exp_dev, m_n_weight_dev, m_candidate);
      }
    } else if constexpr (has_complex_linear) {
      etd_combine(psi_hat, n_hat, m_exp_c_dev, m_n_weight_c_dev, m_candidate);
    } else {
      etd_combine(psi_hat, n_hat, m_exp_dev, m_n_weight_dev, m_candidate);
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
    note_launch();
    Ops::backward(m_fft, psi_hat, psi());
    ++m_backward_count;
  }

  /// Discard the last candidate (no state was modified).
  void reject() noexcept {}

  /**
   * @brief One ETD1 step (`attempt` + `commit`). Returns `t + dt`.
   */
  double step(double t) {
    if (!m_capturing) {
      release_graph();
    }
    const auto a = attempt(t);
    if (!a.success) {
      throw std::runtime_error("SpectralETDSystem: ETD1 attempt failed");
    }
    commit();
    return a.t1;
  }

  /**
   * @brief Bind the stream used by the elementwise kernels.
   *
   * The device FFT must be constructed on this same stream. `nullptr` is the
   * legacy stream, and it is the default. Graph replay requires a non-legacy
   * stream because capture cannot record the legacy stream.
   */
  template <class Stream> void set_stream(Stream stream) {
    if (m_capturing) {
      throw std::logic_error("SpectralETDSystem::set_stream during graph capture");
    }
    release_graph();
    if constexpr (std::is_null_pointer_v<Stream>) {
      m_compute_stream = nullptr;
    } else {
      m_compute_stream = reinterpret_cast<void *>(stream);
    }
  }

  /**
   * @brief Advance `n` steps, capturing the first and replaying the rest.
   *
   * The first call records one step and replays that record `n` times. Capture
   * does not execute the launches. Time-dependent coefficients and a host
   * free-energy reduction are refused: both make the recorded launches a
   * different step from the next one. Returns the time after `n` steps.
   */
  double step_n(double t, int n) {
    if (n < 1) {
      throw std::invalid_argument("SpectralETDSystem::step_n: n must be >= 1");
    }
    if (m_opt.time_dependent_coefficients || declares_time_dependent<Physics>() ||
        declares_time_dependent<Pointwise>()) {
      throw std::invalid_argument(
          "SpectralETDSystem::step_n refuses time-dependent coefficients");
    }
    if constexpr (has_free_energy) {
      throw std::invalid_argument(
          "SpectralETDSystem::step_n refuses a host free-energy reduction");
    }
    if constexpr (!is_device_spectral_space<MemorySpace>::value) {
      throw std::logic_error(
          "SpectralETDSystem::step_n requires a device memory space");
    } else {
      return step_n_device(t, n);
    }
  }

  /// Elementwise and transform launches recorded while the graph was captured.
  [[nodiscard]] int captured_op_launches() const noexcept {
    return m_captured_launches;
  }
  /// Nodes in the captured graph, including the device FFT.
  [[nodiscard]] std::size_t graph_node_count() const noexcept {
    return m_graph_nodes;
  }
  /// Kernel nodes in the captured graph.
  [[nodiscard]] std::size_t graph_kernel_nodes() const noexcept {
    return m_graph_kernel_nodes;
  }

  // ---- queries ----------------------------------------------------------
  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] const Physics &physics() const noexcept { return m_physics; }
  [[nodiscard]] Physics &physics() noexcept { return m_physics; }
  [[nodiscard]] const Pointwise &pointwise() const noexcept { return m_pointwise; }
  [[nodiscard]] const SpectralETDOptions &options() const noexcept { return m_opt; }
  /// \f$L(k)\f$ on this rank's outbox (empty when the physics uses complex L).
  [[nodiscard]] const std::vector<double> &linear_symbol() const noexcept {
    return m_L;
  }
  /// Complex \f$L(\mathbf{k})\f$ (empty when the physics uses real L).
  [[nodiscard]] const std::vector<Complex> &linear_symbol_complex() const noexcept {
    return m_L_c;
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

  /// Forward transforms since construction, including those inside `attempt`.
  [[nodiscard]] int forward_count() const noexcept { return m_forward_count; }
  /// Backward transforms, including a candidate realization.
  [[nodiscard]] int backward_count() const noexcept { return m_backward_count; }
  /// Pointwise nonlinear evaluations since construction.
  [[nodiscard]] int nonlinear_count() const noexcept { return m_nonlinear_count; }

  [[nodiscard]] RealField &psi() { return real_field(m_opt.psi_name); }
  [[nodiscard]] const RealField &psi() const {
    return m_state.template get_field<double, MemorySpace>(m_opt.psi_name);
  }
  [[nodiscard]] ComplexField &psi_hat() { return complex_field(m_opt.psi_hat_name); }
  [[nodiscard]] const ComplexField &psi_hat() const {
    return m_state.template get_field<Complex, MemorySpace>(m_opt.psi_hat_name);
  }

  /**
   * @brief Copy the uncommitted Fourier candidate.
   *
   * Host reference path. The copy is independent of a later `set_dt` or
   * `attempt`, so a full-step candidate cannot be overwritten by half-step
   * coefficients.
   */
  void copy_candidate(std::vector<Complex> &dst) const {
    if constexpr (!std::is_same_v<MemorySpace, pfc::HostSpace>) {
      throw std::logic_error("SpectralETDSystem::copy_candidate is CPU-only");
    } else {
      dst = m_candidate;
    }
  }

  /**
   * @brief Inverse-transform the uncommitted candidate into a real vector.
   *
   * Does not modify `psi` or `psi_hat`. Uses the same backward transform as
   * `commit`, so the real candidate is in the same normalization as `psi`.
   */
  void realize_candidate(std::vector<double> &dst) {
    if constexpr (!std::is_same_v<MemorySpace, pfc::HostSpace>) {
      throw std::logic_error("SpectralETDSystem::realize_candidate is CPU-only");
    } else {
      dst.resize(psi().size());
      m_fft.backward(m_candidate, dst);
      ++m_backward_count;
    }
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

  void note_launch() noexcept {
    if (m_capturing) {
      ++m_captured_launches;
    }
  }

  template <class... Args> void etd_multiply(Args &&...args) {
    note_launch();
    if constexpr (is_device_spectral_space<MemorySpace>::value) {
      Ops::multiply(std::forward<Args>(args)..., m_compute_stream);
    } else {
      Ops::multiply(std::forward<Args>(args)...);
    }
  }

  template <class... Args> void etd_combine(Args &&...args) {
    note_launch();
    if constexpr (is_device_spectral_space<MemorySpace>::value) {
      Ops::combine(std::forward<Args>(args)..., m_compute_stream);
    } else {
      Ops::combine(std::forward<Args>(args)...);
    }
  }

  template <class... Args> void etd_pointwise(Args &&...args) {
    note_launch();
    if constexpr (is_device_spectral_space<MemorySpace>::value) {
      Ops::pointwise(std::forward<Args>(args)..., m_compute_stream);
    } else {
      Ops::pointwise(std::forward<Args>(args)...);
    }
  }

  void release_graph() noexcept {
    if constexpr (is_device_spectral_space<MemorySpace>::value) {
      if (m_graph != nullptr || m_exec != nullptr) {
        Ops::destroy_graph(m_graph, m_exec);
        m_graph = nullptr;
        m_exec = nullptr;
        m_graph_nodes = 0;
        m_graph_kernel_nodes = 0;
      }
    }
  }

  double step_n_device(double t, int n) {
    if (m_compute_stream == nullptr) {
      throw std::invalid_argument(
          "SpectralETDSystem::step_n requires an explicit non-legacy stream");
    }
    if (m_exec == nullptr) {
      // HeFFTe builds the cuFFT/rocFFT plan on the first execute of each
      // direction, and that allocates. Capture rejects an allocation, so both
      // directions run once here. The backward overwrites `psi`; the saved
      // field is restored before the record. Capture itself does not execute.
      psi().sync_to_device();
      std::vector<double> saved_psi;
      psi().with_host_read([&](double const *data, std::size_t count) {
        saved_psi.assign(data, data + count);
      });
      Ops::forward(m_fft, psi(), psi_hat());
      Ops::backward(m_fft, psi_hat(), psi());
      Ops::stream_synchronize(m_compute_stream);
      Ops::device_synchronize();
      psi().with_host_view([&](double *data, std::size_t count) {
        std::copy(saved_psi.begin(), saved_psi.end(), data);
      });
      psi().sync_to_device();

      m_captured_launches = 0;
      const int forwards = m_forward_count;
      const int backwards = m_backward_count;
      const int nonlinear = m_nonlinear_count;
      bool capture_started = false;
      try {
        // Begin, record and instantiate must share the same cleanup guard.
        Ops::capture_begin(m_compute_stream);
        capture_started = true;
        m_capturing = true;
        step(t);
        const auto captured = Ops::capture_end(m_compute_stream);
        capture_started = false;
        m_capturing = false;
        m_graph = captured.graph;
        m_exec = captured.exec;
        m_graph_nodes = captured.nodes;
        m_graph_kernel_nodes = captured.kernel_nodes;
        m_forward_delta = m_forward_count - forwards;
        m_backward_delta = m_backward_count - backwards;
        m_nonlinear_delta = m_nonlinear_count - nonlinear;
      } catch (...) {
        m_capturing = false;
        if (capture_started) Ops::capture_abort(m_compute_stream);
        m_forward_count = forwards;
        m_backward_count = backwards;
        m_nonlinear_count = nonlinear;
        m_captured_launches = 0;
        release_graph();
        throw;
      }
      m_forward_count = forwards;
      m_backward_count = backwards;
      m_nonlinear_count = nonlinear;
    }
    for (int i = 0; i < n; ++i) {
      Ops::replay(m_exec, m_compute_stream);
      Ops::stream_synchronize(m_compute_stream);
      m_forward_count += m_forward_delta;
      m_backward_count += m_backward_delta;
      m_nonlinear_count += m_nonlinear_delta;
      t += m_dt;
    }
    return t;
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
  std::vector<Complex> m_L_c;
  std::vector<Complex> m_exp_c;
  std::vector<Complex> m_n_weight_c;
  std::vector<double> m_filter_host;
  std::vector<double> m_kernel_host;
  std::vector<double> m_mask_host;
  integrator::SpectralExpCoefficientCache<> m_cache{};

  typename Ops::real_coeffs m_exp_dev{};
  typename Ops::real_coeffs m_n_weight_dev{};
  typename Ops::complex_scratch m_exp_c_dev{};
  typename Ops::complex_scratch m_n_weight_c_dev{};
  typename Ops::real_coeffs m_filter_dev{};
  typename Ops::real_coeffs m_kernel_dev{};
  typename Ops::real_coeffs m_mask_dev{};
  typename Ops::complex_scratch m_candidate;
  typename Ops::complex_scratch m_n_masked;

  double m_fe_sum{0.0};
  double m_fe_integral{0.0};
  int m_forward_count{0};
  int m_backward_count{0};
  int m_nonlinear_count{0};
  void *m_compute_stream{nullptr};
  void *m_graph{nullptr};
  void *m_exec{nullptr};
  std::size_t m_graph_nodes{0};
  std::size_t m_graph_kernel_nodes{0};
  int m_captured_launches{0};
  int m_forward_delta{0};
  int m_backward_delta{0};
  int m_nonlinear_delta{0};
  bool m_capturing{false};
};

} // namespace pfc::sim
