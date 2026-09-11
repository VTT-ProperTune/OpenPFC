// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_stepper_hip.hpp
 * @brief Device-resident owner of the sixteen fields and three halo groups
 *        that one step of equations (1)-(4) needs.
 *
 * @details
 * ## What this class is for
 *
 * `alloy_dendrite::Stepper` is the host stepper; this is its GPU twin's
 * bookkeeping half. It allocates the same sixteen fields, registers the same
 * three halo groups in the same order with the same `exchange_base` tags, and
 * calls the four launchers of `device_step_hip.hpp` in the same sequence.
 * The arithmetic lives in the `.hip` translation unit; nothing here does any.
 *
 * ## Why it does not take an `FDGPUStack`
 *
 * `pfc::sim::stacks::FDGPUStack` builds its decomposition internally with
 * `pfc::decomposition::create(domain, nproc)`, which picks a minimum-surface
 * brick. That is the right choice for a pure FD run and the wrong one when
 * the same fields also have to be handed to a HeFFTe transform, because
 * `SpectralCPUStack` uses `spectral_fft_proc_grid` -- a slab -- and two
 * different decompositions of the same domain means an MPI redistribution on
 * every elastic solve, on top of the host round-trip. Taking the
 * decomposition as a constructor argument lets the coupled driver build both
 * halves on the *FFT's* grid so the round-trip is a straight `memcpy` per
 * rank; the parity driver passes the FD-optimal grid and never notices.
 * See `src/hip/alloy_dendrite_coupled_cost.cpp` for what that costs.
 *
 * ## Residency
 *
 * The fields are `pfc::data::Field<double, pfc::HIPSpace>`, which tracks
 * which side last wrote. Every kernel launch is followed by
 * `note_device_write()` on the fields it wrote, so a later `with_host_view`
 * pulls fresh data and a later `HaloExchange` packs from the device buffer.
 * Forgetting one of those is the classic silent-wrong-answer bug on this
 * path, so they are grouped immediately after the launch they belong to.
 *
 * Elasticity stays on the host. `set_elastic_driving_force` is the same hook
 * as on `Stepper`: a device-resident `dF_el/dphi` field, owned cells only,
 * written by whoever solved equations (5)-(7). The coupled-cost driver fills
 * it by a host round-trip through `elasticity.hpp`; the parity driver leaves
 * it null.
 *
 * @see device_step_hip.hpp -- the launch surface
 * @see step.hpp -- the host twin whose stage order this reproduces
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "device_stepper_hip.hpp requires HIP (-DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>

#include <alloy_dendrite/device_step_hip.hpp>
#include <alloy_dendrite/parameters.hpp>

namespace alloy_dendrite {

/**
 * @brief GPU twin of @ref Stepper.
 *
 * @tparam Dim 2 for an `nz = 1` slab, 3 for a full brick. Same meaning and
 *         same consequences as on the host: it selects the halo direction set
 *         and decides whether the `z` terms are compiled at all.
 */
template <int Dim> class DeviceStepper {
  static_assert(Dim == 2 || Dim == 3, "DeviceStepper: Dim must be 2 or 3");

public:
  using Field = pfc::data::Field<double, pfc::HIPSpace>;
  using Exchange = pfc::comm::HaloExchange<pfc::HIPSpace, double>;

  /// Halo direction set matching @p Dim; mirrors `Stepper<Dim>::directions()`.
  [[nodiscard]] static pfc::halo::HaloDirectionSet directions() {
    if constexpr (Dim == 2) {
      return pfc::halo::presets::Axes2D();
    } else {
      return pfc::halo::presets::Axes3D();
    }
  }

  DeviceStepper(const DeviceStepper &) = delete;
  DeviceStepper &operator=(const DeviceStepper &) = delete;
  DeviceStepper(DeviceStepper &&) = delete;
  DeviceStepper &operator=(DeviceStepper &&) = delete;

  /**
   * @param domain   Global domain; supplies spacing and origin.
   * @param decomp   Decomposition the fields are cut on. Pass the same one
   *                 the host stepper uses for a parity run, or the FFT's for
   *                 a coupled run.
   * @param rank     Caller's rank on @p comm.
   * @param comm     Communicator for the three halo groups.
   * @param params   Physical parameters of (1)-(4).
   * @param fd_order Even central-difference order in `[2, 14]`.
   */
  DeviceStepper(const pfc::Domain &domain,
                const pfc::decomposition::Decomposition &decomp, int rank,
                MPI_Comm comm, const ModelParams &params, int fd_order)
      : m_p(params), m_order(fd_order),
        m_box(pfc::decomposition::local_box(decomp, rank)),
        m_phi(domain, m_box, fd_order / 2), m_U(make_field_(domain, fd_order)),
        m_theta(make_field_(domain, fd_order)), m_psi(make_field_(domain, fd_order)),
        m_dphidt(make_field_(domain, fd_order)), m_tau(make_field_(domain, fd_order)),
        m_lap_theta(make_field_(domain, fd_order)),
        m_gx(make_field_(domain, fd_order)), m_gy(make_field_(domain, fd_order)),
        m_gz(make_field_(domain, fd_order)), m_Fx(make_field_(domain, fd_order)),
        m_Fy(make_field_(domain, fd_order)), m_Fz(make_field_(domain, fd_order)),
        m_Jx(make_field_(domain, fd_order)), m_Jy(make_field_(domain, fd_order)),
        m_Jz(make_field_(domain, fd_order)),
        m_ex_state(state_group_(), decomp, rank, comm, opts_(0)),
        m_ex_flux(vector_group_(m_Fx, m_Fy, m_Fz), decomp, rank, comm, opts_(1000)),
        m_ex_solute(vector_group_(m_Jx, m_Jy, m_Jz), decomp, rank, comm,
                    opts_(2000)) {
    if (fd_order < 2 || fd_order > 14 || (fd_order % 2) != 0) {
      throw std::invalid_argument(
          "alloy_dendrite::DeviceStepper: fd_order must be even and in [2, 14]");
    }
    if (m_p.k <= 0.0 || m_p.k >= 1.0) {
      throw std::invalid_argument("alloy_dendrite::DeviceStepper: k must be in (0,1)");
    }
    // Same floor as the host stepper: a non-positive interface width is not a
    // model. Refuse rather than produce plausible-looking nonsense.
    if (m_p.eps4 < 0.0 || m_p.eps4 >= 1.0 / 3.0) {
      throw std::invalid_argument(
          "alloy_dendrite::DeviceStepper: eps4 must be in [0, 1/3); the "
          "normalised a_s = (1-3 eps4) + 4 eps4 sum n_i^4 is non-positive "
          "beyond that");
    }
    if constexpr (Dim == 3) {
      if (m_phi.local_size()[2] < 2) {
        throw std::invalid_argument(
            "alloy_dendrite::DeviceStepper<3>: nz must be > 1; use Dim = 2");
      }
    }
    build_geometry_();
    build_stencil_(domain);
    build_params_();
    bind_pointers_();
  }

  /// Recompute `psi = P(phi) U` on the device. Call once, after the initial
  /// condition has been pushed; from then on `psi` is the primary variable.
  void seed_conserved_solute() {
    m_phi.sync_to_device();
    m_U.sync_to_device();
    hip::alloy_seed_psi_hip(m_f, m_g, m_p.k, Dim);
    m_psi.note_device_write();
  }

  /// Install `dF_el/dphi` (device-resident, owned cells only), or `nullptr`.
  void set_elastic_driving_force(const Field *dfel) noexcept {
    m_f.dfel = (dfel != nullptr) ? dfel->data() : nullptr;
  }

  /// Advance `phi`, `psi` (hence `U`) and `theta` by @p dt. Stage order and
  /// halo groups are identical to `Stepper<Dim>::step`.
  void step(double dt) {
    m_ex_state.exchange();

    hip::alloy_stage_a_hip(m_f, m_g, m_s, m_dp, Dim);
    m_gx.note_device_write();
    m_gy.note_device_write();
    if constexpr (Dim == 3) m_gz.note_device_write();
    m_tau.note_device_write();
    m_Fx.note_device_write();
    m_Fy.note_device_write();
    if constexpr (Dim == 3) m_Fz.note_device_write();
    m_lap_theta.note_device_write();

    m_ex_flux.exchange();

    hip::alloy_stage_b_hip(m_f, m_g, m_s, m_dp, Dim);
    m_dphidt.note_device_write();

    hip::alloy_stage_c_hip(m_f, m_g, m_s, m_dp, Dim);
    m_Jx.note_device_write();
    m_Jy.note_device_write();
    if constexpr (Dim == 3) m_Jz.note_device_write();

    m_ex_solute.exchange();

    hip::alloy_stage_d_hip(m_f, m_g, m_s, m_dp, dt, Dim);
    m_psi.note_device_write();
    m_phi.note_device_write();
    m_U.note_device_write();
    if (m_p.evolve_theta) m_theta.note_device_write();
  }

  [[nodiscard]] Field &phi() noexcept { return m_phi; }
  [[nodiscard]] Field &solute() noexcept { return m_U; }
  [[nodiscard]] Field &temperature() noexcept { return m_theta; }
  [[nodiscard]] Field &conserved_solute() noexcept { return m_psi; }
  [[nodiscard]] const ModelParams &params() const noexcept { return m_p; }
  [[nodiscard]] int fd_order() const noexcept { return m_order; }
  [[nodiscard]] const pfc::Box3i &box() const noexcept { return m_box; }

private:
  [[nodiscard]] Field make_field_(const pfc::Domain &domain, int fd_order) const {
    return Field(domain, m_box, fd_order / 2);
  }

  [[nodiscard]] static pfc::comm::HaloExchangeOptions opts_(int base) {
    pfc::comm::HaloExchangeOptions o;
    o.directions = directions();
    o.exchange_base = base;
    return o;
  }

  [[nodiscard]] std::vector<Field *> state_group_() {
    return std::vector<Field *>{&m_phi, &m_U, &m_theta};
  }

  [[nodiscard]] static std::vector<Field *> vector_group_(Field &fx, Field &fy,
                                                          Field &fz) {
    std::vector<Field *> v{&fx, &fy};
    if constexpr (Dim == 3) {
      v.push_back(&fz);
    }
    return v;
  }

  void build_geometry_() {
    const auto n = m_phi.local_size();
    m_g.nx = n[0];
    m_g.ny = n[1];
    m_g.nz = n[2];
    m_g.hw = m_phi.storage_halo();
    m_g.sy = m_phi.padded_extent(0);
    m_g.sz = static_cast<long long>(m_phi.padded_extent(0)) * m_phi.padded_extent(1);
  }

  /// Load the *same* integer stencil tables the host evaluator uses, and keep
  /// the `1/(h^n denom)` factor separate so the device sums and scales in the
  /// host's order. See the note in `device_step_hip.hpp` on why the library's
  /// pre-scaled `FDGradientDevice` payload is deliberately not used.
  void build_stencil_(const pfc::Domain &domain) {
    pfc::field::fd::EvenCentralD1View st1{};
    if (!pfc::field::fd::lookup_even_central_d1(m_order, &st1)) {
      throw std::invalid_argument("DeviceStepper: no D1 stencil for order " +
                                  std::to_string(m_order));
    }
    pfc::field::fd::EvenCentralD2View st2{};
    if (!pfc::field::fd::lookup_even_central_d2(m_order, &st2)) {
      throw std::invalid_argument("DeviceStepper: no D2 stencil for order " +
                                  std::to_string(m_order));
    }
    if (st1.half_width > hip::kMaxHalfWidth1 ||
        st2.half_width > hip::kMaxHalfWidth2) {
      throw std::invalid_argument("DeviceStepper: stencil wider than the device "
                                  "payload; raise kMaxHalfWidth* and rebuild");
    }
    m_s.hw1 = st1.half_width;
    m_s.hw2 = st2.half_width;
    for (int k = 1; k <= st1.half_width; ++k) {
      m_s.c1[k] = static_cast<double>(st1.coeffs[k]);
    }
    for (int k = 0; k <= st2.half_width; ++k) {
      m_s.c2[k] = static_cast<double>(st2.coeffs[k]);
    }
    const auto sp = domain.spacing;
    const double inv_d1 = 1.0 / static_cast<double>(st1.denom);
    const double inv_d2 = 1.0 / static_cast<double>(st2.denom);
    m_s.s1x = (1.0 / sp[0]) * inv_d1;
    m_s.s1y = (1.0 / sp[1]) * inv_d1;
    m_s.s1z = (1.0 / sp[2]) * inv_d1;
    m_s.s2x = (1.0 / sp[0]) * (1.0 / sp[0]) * inv_d2;
    m_s.s2y = (1.0 / sp[1]) * (1.0 / sp[1]) * inv_d2;
    m_s.s2z = (1.0 / sp[2]) * (1.0 / sp[2]) * inv_d2;
  }

  void build_params_() {
    m_dp.W0 = m_p.W0;
    m_dp.tau0 = m_p.tau0;
    m_dp.lambda = m_p.lambda;
    m_dp.k = m_p.k;
    m_dp.D_l = m_p.D_l;
    m_dp.D_th = m_p.D_th;
    m_dp.M_c = m_p.M_c;
    m_dp.eps4 = m_p.eps4;
    // Precomputed host-side so the device never re-evaluates 1/(2 sqrt 2).
    m_dp.at = m_p.at_scale * kAntiTrapCoeff * m_p.W0;
    m_dp.lambda_el = m_p.lambda_el;
    m_dp.grad_floor2 = kGradNormFloor2;
    m_dp.spec_source = m_p.spec_source ? 1 : 0;
    m_dp.evolve_theta = m_p.evolve_theta ? 1 : 0;
    m_dp.thermal = (m_p.evolve_theta && m_p.D_th > 0.0) ? 1 : 0;
  }

  void bind_pointers_() {
    m_f.phi = m_phi.data();
    m_f.U = m_U.data();
    m_f.theta = m_theta.data();
    m_f.psi = m_psi.data();
    m_f.dphidt = m_dphidt.data();
    m_f.tau = m_tau.data();
    m_f.lap_theta = m_lap_theta.data();
    m_f.gx = m_gx.data();
    m_f.gy = m_gy.data();
    m_f.gz = m_gz.data();
    m_f.Fx = m_Fx.data();
    m_f.Fy = m_Fy.data();
    m_f.Fz = m_Fz.data();
    m_f.Jx = m_Jx.data();
    m_f.Jy = m_Jy.data();
    m_f.Jz = m_Jz.data();
  }

  ModelParams m_p{};
  int m_order{2};
  pfc::Box3i m_box{};

  Field m_phi;
  Field m_U, m_theta, m_psi, m_dphidt, m_tau, m_lap_theta;
  Field m_gx, m_gy, m_gz;
  Field m_Fx, m_Fy, m_Fz;
  Field m_Jx, m_Jy, m_Jz;

  Exchange m_ex_state;
  Exchange m_ex_flux;
  Exchange m_ex_solute;

  hip::DeviceGeom m_g{};
  hip::DeviceStencil m_s{};
  hip::DeviceParams m_dp{};
  hip::DeviceFields m_f{};
};

} // namespace alloy_dendrite
