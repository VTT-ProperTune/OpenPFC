// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file step.hpp
 * @brief Explicit four-stage step for equations (1)-(4) of `MODEL_SPEC.md`:
 *        quantitative dilute-alloy phase field with anti-trapping current,
 *        coupled to solute and to temperature with latent heat.
 *
 * @details
 * ## What is here and what is deliberately not
 *
 * Equations (1)-(4). Equations (5)-(7) -- the eigenstrain microelasticity
 * and its Fourier Green-operator solve -- live in `elasticity.hpp`, which
 * wraps `openpfc_apps/microelasticity.hpp`; this file knows about them only
 * through one line, marked `ELASTIC HOOK` in @ref Stepper::stage_b_. A
 * caller sets @ref Stepper::set_elastic_driving_force to a field holding
 * `dF_el/dphi` and @ref ModelParams::lambda_el to a nonzero value, and the
 * term `- lambda_el (1-phi^2)^2 dF_el/dphi` joins the phase-field right-hand
 * side with no other change to this file. The quasi-static solve is lagged
 * by `n_el_substep` steps in the driver's loop (spec, "Numerics"), which is
 * why the hook is a *field* rather than a callback: the stepper must be able
 * to reuse a solution computed several steps ago, and a callback would
 * invite someone to re-solve inside stage B where the FFT would be executed
 * once per cell-loop rather than once per step.
 *
 * ## Why four stages and not one
 *
 * Two of the four terms are divergences of quantities that are themselves
 * functions of a gradient:
 *
 *  - `div[W(n)^2 grad phi] + sum_i d_i[|grad phi|^2 W dW/d(d_i phi)]`
 *  - `div[D_l q(phi) grad U + j_at]`
 *
 * Evaluating those with a *collocated* high-order central stencil means the
 * flux has to exist in the halo, which means one MPI exchange per divergence.
 * The anti-trapping current additionally needs `d_t phi`, which is only known
 * after the phase-field right-hand side is complete. The dependency chain
 * `phi -> flux -> d_t phi -> solute flux -> divergence` has no shorter
 * schedule than
 *
 *     A: gradients, anisotropy, phase-field flux F_i,  thermal Laplacian
 *        (exchange F)
 *     B: div F  ->  d_t phi
 *     C: solute flux J_i = D_l q grad U + j_at(d_t phi)
 *        (exchange J)
 *     D: div J  ->  advance psi, phi, theta; recover U
 *
 * i.e. three halo exchanges per step (state, F, J). Kobayashi's two-stage
 * driver is the same idea with one divergence instead of two.
 *
 * ## Why the solute variable is `psi = P(phi) U`, not `U`
 *
 * Equation (3) is written for the conserved density. Advancing `psi` and
 * recovering `U = psi / P(phi_new)` at the end of the step buys exact
 * discrete solute conservation, and "exact" here means to round-off, not to
 * truncation error:
 *
 *  - a central antisymmetric stencil has `sum_m a_m = 0`, so on a periodic
 *    grid `sum_cells div J` telescopes to zero identically -- collocated
 *    central differencing of a flux is conservative;
 *  - `P` is affine in `phi`, so `sum_cells [P(phi_new) - P(phi_old)]/(1-k)`
 *    equals `-(dt/2) sum_cells d_t phi` exactly;
 *  - the source in (3) is `+(1/2) d_t phi` with the *same* `d_t phi` array
 *    that advances `phi`.
 *
 * The three cancel, so `sum_cells [P/(1-k) + psi]`, which is
 * `sum_cells c / (c_l^0 (1-k))`, is invariant. @ref diagnostics.hpp measures
 * exactly that quantity, and a drift above round-off means one of the three
 * bullets has been broken.
 *
 * `P(phi) = ((1+k) - (1-k) phi)/2` is bounded below by `k > 0`, so the
 * recovery `U = psi / P` never divides by zero even at `phi = +1`.
 *
 * ## Where this differs from `MODEL_SPEC.md`
 *
 * Equation (3) of the spec pairs a conservative left-hand side,
 * `d_t[P U]`, with the source `(1/2)(1 + (1-k) U) d_t phi` that belongs to
 * the *non*-conservative form `P d_t U`. Only two pairings are consistent
 * with `c/c_l^0 = P (1 + (1-k) U)` and a flux `-D q c_l^0 (1-k) grad U`:
 *
 *     d_t[P U] = div[...] + (1/2) d_t phi                      (conservative)
 *     P d_t U  = div[...] + (1/2)(1 + (1-k) U) d_t phi         (Echebarria et al.)
 *
 * They are the same equation. The spec's mixture is neither, and it is not a
 * harmless mismatch: it injects a spurious `(1-k) U d_t phi / 2` source that
 * scales with the local supersaturation, i.e. exactly the term that biases
 * the partition coefficient. This file implements the conservative pairing.
 *
 * How much the mismatch costs is not a matter of opinion. Redo the 1-D
 * steady-state integration of @ref kAntiTrapCoeff with the spec's source and
 * the solid-side gradient becomes
 *
 *     U'(phi = +1) = -(V / D_l) (1 - k) U_solid
 *
 * instead of zero: a residual chemical-potential gradient inside a phase with
 * no diffusivity, proportional to the local supersaturation and to `V`. That
 * is exactly a velocity-dependent bias on the partition coefficient, i.e. it
 * defeats the one thing the anti-trapping current exists to fix.
 * @ref ModelParams::spec_source runs the spec's version so the size of the
 * bias can be quoted rather than argued about.
 *
 * The spec's *sign* for `j_at`, on the other hand, is right; see
 * @ref kAntiTrapCoeff for the cancellation that fixes it, and for why the
 * plausible-sounding transport argument for the opposite sign is wrong.
 *
 * @see parameters.hpp for the thin-interface relations these equations obey
 * @see diagnostics.hpp for the conservation and interface measurements
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>

#include <alloy_dendrite/parameters.hpp>

namespace alloy_dendrite {

/// Per-point grads catalogs. Only the members a stage actually needs are
/// declared, so `FDGradient` never touches an axis whose halo is not
/// exchanged -- which is what makes an `nz = 1` slab safe under `Axes2D()`.
struct FirstDerivs2 {
  double x{}, y{};
};
struct FirstDerivs3 {
  double x{}, y{}, z{};
};
struct SecondDerivs2 {
  double xx{}, yy{};
};
struct SecondDerivs3 {
  double xx{}, yy{}, zz{};
};
struct DerivX {
  double x{};
};
struct DerivY {
  double y{};
};
struct DerivZ {
  double z{};
};

template <int Dim>
using FirstDerivs = std::conditional_t<Dim == 3, FirstDerivs3, FirstDerivs2>;
template <int Dim>
using SecondDerivs = std::conditional_t<Dim == 3, SecondDerivs3, SecondDerivs2>;

/**
 * @brief Anisotropy of equation (1) evaluated from a raw gradient.
 *
 * `a_s(n) = (1 - 3 eps4)[1 + (4 eps4/(1 - 3 eps4))(n_x^4+n_y^4+n_z^4)]`,
 * `W = W0 a_s`, `tau = tau0 a_s^2`, plus the flux
 *
 *     A_i = |grad phi|^2 W dW/d(d_i phi)
 *         = 16 eps4 W0^2 a_s [ g_i^3 G^2 - (sum_j g_j^4) g_i ] / G^4
 *
 * with `g_i = d_i phi` and `G^2 = sum_j g_j^2`. The algebra above is worth
 * spelling out because it is where the usual `0/0` disappears: the numerator
 * is `O(g^5)` against a `G^4` denominator, so `A_i` vanishes linearly in the
 * gradient and needs no ad-hoc floor. Only the `G^4` division does, and it
 * is guarded by @ref kGradNormFloor2.
 *
 * @note **Normalised Karma-Rappel form** (`MODEL_SPEC.md` equation (1) as
 *       corrected 2026-09-11):
 *
 *           a_s = (1 - 3 eps4) [ 1 + (4 eps4 / (1 - 3 eps4)) sum n_i^4 ]
 *               = (1 - 3 eps4) + 4 eps4 sum n_i^4
 *
 *       which is the second, cheaper form used below. Why the normalisation
 *       matters rather than being cosmetic: in 2-D
 *       `n_x^4 + n_y^4 = (3 + cos 4 theta)/4`, so
 *
 *           a_s = (1 - 3 eps4) + eps4 (3 + cos 4 theta) = 1 + eps4 cos 4 theta
 *
 *       *exactly*. `eps4` is then the anisotropy strength of the selection
 *       theory, `a_s` averages to 1 over orientation, and `W0` is the
 *       interface width of the `<110>` (soft) direction rather than of no
 *       orientation at all. The earlier un-normalised `a_s = 1 + eps4 sum
 *       n_i^4` gave an effective strength `(eps4/4)/(1 + 0.75 eps4)` --
 *       about a quarter of nominal -- so Karma-Rappel's `eps4 = 0.02` acted
 *       like 0.005 and grew a blob instead of a dendrite. In 3-D the same
 *       expression is the standard `<100>` cubic form with `sum n_i^4` in
 *       `[1/3, 1]`, i.e. `a_s` in `[1 - 5 eps4/3, 1 + eps4]`.
 *
 *       `d a_s / d(sum n_i^4) = 4 eps4`, which is the *only* place the
 *       normalisation enters the flux: `A_i` below carries a factor
 *       `4 eps4` where the un-normalised form carried `eps4`. Getting the
 *       function right and the flux wrong would give a model whose
 *       anisotropy and whose surface-stiffness disagree, which is worse
 *       than either convention used consistently.
 *
 * @note **Crystal frame.** The quartic is not taken on the lab normal.
 *       With `θ_c` the in-plane rotation of the `<100>` axes (about `z`
 *       in 3-D), `n' = R(θ_c)^T n` and `a_s` uses `sum n_i'^4`. The
 *       Cahn–Hoffman flux is a vector, so it is formed in the crystal
 *       frame and rotated back with `R`, not `R^T`:
 *
 *           g' = R^T g,   A = R A'(g')
 *
 *       `θ_c = 0` skips the rotation entirely, so the cubic-axis unit
 *       tests keep their last bits. A second grain (bicrystal) passes a
 *       per-cell `θ_c` rather than a second `a_s`.
 *
 * @note Two ceilings on `eps4` worth knowing. `a_s > 0` needs
 *       `eps4 < 1/3` (3-D) -- below that the interface width of the soft
 *       orientation goes through zero. Long before that, the 2-D
 *       interfacial stiffness `a_s + a_s''` = `1 - 15 eps4 cos 4 theta`
 *       changes sign at `eps4 = 1/15 = 0.0667`: past that the equilibrium
 *       shape has missing orientations (corners/ears) and the
 *       smooth-tip selection theory this application compares against no
 *       longer applies. @ref ModelParams::eps4 is checked against both.
 */
struct AnisotropyPoint {
  double a_s{1.0};
  double W{1.0};
  double tau{1.0};
  double flux[3]{0.0, 0.0, 0.0};
};

template <int Dim>
[[nodiscard]] inline AnisotropyPoint
evaluate_anisotropy(const ModelParams &p, double gx, double gy, double gz,
                    double theta_c) noexcept {
  AnisotropyPoint out;
  out.W = p.W0;
  out.tau = p.tau0;
  if (p.eps4 == 0.0) {
    return out;
  }
  // Crystal-frame rotation about z: n' = R(θ_c)^T n. θ_c = 0 is skipped so
  // the lab-frame cubic-axis checks keep their last bits.
  const bool rotate = (theta_c != 0.0);
  const double cth = rotate ? std::cos(theta_c) : 1.0;
  const double sth = rotate ? std::sin(theta_c) : 0.0;
  const double gx_c = rotate ? (cth * gx + sth * gy) : gx;
  const double gy_c = rotate ? (-sth * gx + cth * gy) : gy;
  const double gz_c = gz;
  const double g2 = gx_c * gx_c + gy_c * gy_c + (Dim == 3 ? gz_c * gz_c : 0.0);
  if (g2 < kGradNormFloor2) {
    return out;
  }
  const double g4sum = gx_c * gx_c * gx_c * gx_c + gy_c * gy_c * gy_c * gy_c +
                       (Dim == 3 ? gz_c * gz_c * gz_c * gz_c : 0.0);
  const double inv_g2 = 1.0 / g2;
  const double inv_g4 = inv_g2 * inv_g2;
  const double s = g4sum * inv_g4; // sum n_i'^4
  // Normalised Karma-Rappel: a_s = (1-3 eps4) + 4 eps4 sum n_i'^4, so that in
  // 2-D a_s = 1 + eps4 cos 4 (theta - theta_c) exactly. See the note above.
  out.a_s = (1.0 - 3.0 * p.eps4) + 4.0 * p.eps4 * s;
  out.W = p.W0 * out.a_s;
  out.tau = p.tau0 * out.a_s * out.a_s;
  // A'_i in the crystal frame, then A = R A' in the lab frame.
  const double pre = 16.0 * p.eps4 * p.W0 * p.W0 * out.a_s * inv_g4;
  const double fx_c = pre * (gx_c * gx_c * gx_c * g2 - g4sum * gx_c);
  const double fy_c = pre * (gy_c * gy_c * gy_c * g2 - g4sum * gy_c);
  out.flux[0] = rotate ? (cth * fx_c - sth * fy_c) : fx_c;
  out.flux[1] = rotate ? (sth * fx_c + cth * fy_c) : fy_c;
  if constexpr (Dim == 3) {
    out.flux[2] = pre * (gz_c * gz_c * gz_c * g2 - g4sum * gz_c);
  }
  return out;
}

template <int Dim>
[[nodiscard]] inline AnisotropyPoint
evaluate_anisotropy(const ModelParams &p, double gx, double gy, double gz) noexcept {
  return evaluate_anisotropy<Dim>(p, gx, gy, gz, p.crystal_angle);
}

/**
 * @brief Owns the working fields and drives one explicit step of (1)-(4).
 *
 * Non-copyable and non-movable: the `FDGradient` evaluators hold raw pointers
 * into the member fields, so moving the stepper would dangle every one of
 * them. Construct in place and take a reference, exactly as the shipped
 * stacks do.
 *
 * @tparam Dim 2 for an `nz = 1` slab, 3 for a full brick. The dimension
 *         decides which grads catalogs are instantiated, which halo
 *         directions are exchanged, and whether the `z` terms exist at all;
 *         it is a template parameter rather than a run-time flag so that a
 *         2-D run cannot accidentally read an unexchanged `z` halo.
 */
template <int Dim> class Stepper {
  static_assert(Dim == 2 || Dim == 3, "Stepper: Dim must be 2 or 3");

public:
  using Field = pfc::data::Field<double, pfc::HostSpace>;
  using Stack = pfc::sim::stacks::FDPaddedCPUStack;
  using Exchange = pfc::comm::HaloExchange<pfc::HostSpace, double>;

  /// Halo direction set matching @p Dim. `Axes2D()` on a slab avoids
  /// exchanging a `z` face that has no owned cells behind it.
  [[nodiscard]] static pfc::halo::HaloDirectionSet directions() {
    if constexpr (Dim == 2) {
      return pfc::halo::presets::Axes2D();
    } else {
      return pfc::halo::presets::Axes3D();
    }
  }

  Stepper(const Stepper &) = delete;
  Stepper &operator=(const Stepper &) = delete;
  Stepper(Stepper &&) = delete;
  Stepper &operator=(Stepper &&) = delete;

  /**
   * @param stack    Padded FD stack; its storage halo must be `fd_order / 2`.
   * @param params   Physical parameters of (1)-(4).
   * @param fd_order Even central-difference order. First derivatives are
   *                 tabulated for 2..14 and second derivatives for 2..20, so
   *                 the usable range here (both are needed) is 2..14.
   */
  Stepper(Stack &stack, const ModelParams &params, int fd_order)
      : m_stack(stack), m_p(params), m_order(fd_order), m_phi(stack.u()),
        m_U(stack.make_field()), m_theta(stack.make_field()),
        m_psi(stack.make_field()), m_dphidt(stack.make_field()),
        m_tau(stack.make_field()), m_lap_theta(stack.make_field()),
        m_gx(stack.make_field()), m_gy(stack.make_field()), m_gz(stack.make_field()),
        m_Fx(stack.make_field()), m_Fy(stack.make_field()), m_Fz(stack.make_field()),
        m_Jx(stack.make_field()), m_Jy(stack.make_field()), m_Jz(stack.make_field()),
        m_ex_state(make_state_exchange_()),
        m_ex_flux(make_group_exchange_(stack, m_Fx, m_Fy, m_Fz, 1000)),
        m_ex_solute(make_group_exchange_(stack, m_Jx, m_Jy, m_Jz, 2000)),
        m_grad_phi(pfc::field::create<FirstDerivs<Dim>>(m_phi, fd_order)),
        m_grad_U(pfc::field::create<FirstDerivs<Dim>>(m_U, fd_order)),
        m_lap_theta_eval(pfc::field::create<SecondDerivs<Dim>>(m_theta, fd_order)),
        m_dFx(pfc::field::create<DerivX>(m_Fx, fd_order)),
        m_dFy(pfc::field::create<DerivY>(m_Fy, fd_order)),
        m_dFz(pfc::field::create<DerivZ>(m_Fz, fd_order)),
        m_dJx(pfc::field::create<DerivX>(m_Jx, fd_order)),
        m_dJy(pfc::field::create<DerivY>(m_Jy, fd_order)),
        m_dJz(pfc::field::create<DerivZ>(m_Jz, fd_order)) {
    if (fd_order < 2 || fd_order > 14 || (fd_order % 2) != 0) {
      throw std::invalid_argument(
          "alloy_dendrite::Stepper: fd_order must be even and in [2, 14] "
          "(first derivatives are only tabulated to order 14)");
    }
    if (stack.halo_width() < fd_order / 2) {
      throw std::invalid_argument(
          "alloy_dendrite::Stepper: stack halo width must be >= fd_order/2");
    }
    if (m_p.k <= 0.0 || m_p.k >= 1.0) {
      throw std::invalid_argument("alloy_dendrite::Stepper: k must be in (0,1)");
    }
    // The normalised anisotropy of equation (1) has a hard floor: a_s reaches
    // 1 - 5 eps4/3 (3-D) or 1 - eps4 (2-D) and a non-positive interface width
    // is not a model, it is a crash waiting for a gradient to point the wrong
    // way. Refuse rather than produce plausible-looking nonsense.
    if (m_p.eps4 < 0.0 || m_p.eps4 >= 1.0 / 3.0) {
      throw std::invalid_argument(
          "alloy_dendrite::Stepper: eps4 must be in [0, 1/3); the normalised "
          "a_s = (1-3 eps4) + 4 eps4 sum n_i^4 is non-positive beyond that");
    }
    if constexpr (Dim == 3) {
      if (m_phi.local_size()[2] < 2) {
        throw std::invalid_argument(
            "alloy_dendrite::Stepper<3>: nz must be > 1; use Stepper<2>");
      }
    }
  }

  /// Recompute the conserved solute density from the current `phi` and `U`.
  /// Call once after setting the initial condition, and never again -- from
  /// then on `psi` is the primary variable and `U` is derived from it.
  void seed_conserved_solute() {
    const double k = m_p.k;
    m_phi.for_each_owned([&](int i, int j, int kk) {
      m_psi(i, j, kk) = solute_prefactor(k, m_phi(i, j, kk)) * m_U(i, j, kk);
    });
  }

  /**
   * @brief Supply `dF_el/dphi` for the elastic feedback term of equation (2).
   *
   * Pass `nullptr` (the default) to leave the term out. The field must have
   * the same owned box as the stepper's fields and must stay alive for as
   * long as it is installed; it is read, never written. Only the owned cells
   * are read, so the caller does not have to exchange its halo.
   */
  void set_elastic_driving_force(const Field *dfel_dphi) noexcept {
    m_dfel_dphi = dfel_dphi;
  }

  /**
   * @brief Per-cell crystal angle, radians, overriding
   *        @ref ModelParams::crystal_angle.
   *
   * Pass `nullptr` (the default) to use the scalar on `ModelParams`. The
   * bicrystal IC writes a nearest-seed Voronoi field and installs it here;
   * a single grain never needs to. Owned cells only, like the elastic hook.
   */
  void set_crystal_angle_field(const Field *angle) noexcept { m_angle = angle; }

  /// Advance `phi`, `psi` (hence `U`) and `theta` by @p dt.
  void step(double dt) {
    m_ex_state.exchange();
    stage_a_();
    m_ex_flux.exchange();
    stage_b_();
    stage_c_();
    m_ex_solute.exchange();
    stage_d_(dt);
  }

  [[nodiscard]] Field &phi() noexcept { return m_phi; }
  [[nodiscard]] const Field &phi() const noexcept { return m_phi; }
  [[nodiscard]] Field &solute() noexcept { return m_U; }
  [[nodiscard]] const Field &solute() const noexcept { return m_U; }
  [[nodiscard]] Field &temperature() noexcept { return m_theta; }
  [[nodiscard]] const Field &temperature() const noexcept { return m_theta; }
  [[nodiscard]] const Field &conserved_solute() const noexcept { return m_psi; }
  [[nodiscard]] const Field &phase_rate() const noexcept { return m_dphidt; }
  [[nodiscard]] const ModelParams &params() const noexcept { return m_p; }
  [[nodiscard]] Stack &stack() noexcept { return m_stack; }
  [[nodiscard]] int fd_order() const noexcept { return m_order; }

private:
  /// Halo group for the prognostic state. Called from the member
  /// initialiser list, which is safe because `m_phi`, `m_U` and `m_theta` are
  /// declared before `m_ex_state` and members are initialised in declaration
  /// order.
  Exchange make_state_exchange_() {
    pfc::comm::HaloExchangeOptions opt;
    opt.directions = directions();
    opt.exchange_base = 0;
    return m_stack.make_exchange({&m_phi, &m_U, &m_theta}, opt);
  }

  static Exchange make_group_exchange_(Stack &stack, Field &fx, Field &fy, Field &fz,
                                       int base) {
    pfc::comm::HaloExchangeOptions opt;
    opt.directions = directions();
    opt.exchange_base = base;
    std::vector<Field *> fields{&fx, &fy};
    if constexpr (Dim == 3) {
      fields.push_back(&fz);
    }
    return stack.make_exchange(std::move(fields), opt);
  }

  /// Stage A: gradients of `phi`, anisotropy, the phase-field flux
  /// `F_i = W^2 d_i phi + A_i`, and the thermal Laplacian.
  ///
  /// The thermal Laplacian is computed *here*, three stages before it is
  /// used, for one reason: stage D writes `theta` in place, and a Laplacian
  /// evaluated there would read neighbours that the same sweep had already
  /// advanced. Precomputing into a scratch field is the cheapest way to keep
  /// the update simultaneous rather than Gauss-Seidel.
  void stage_a_() {
    const auto &p = m_p;
    const bool thermal = p.evolve_theta && p.D_th > 0.0;
    m_phi.for_each_owned([&](int i, int j, int kk) {
      const auto g = m_grad_phi(i, j, kk);
      const double gx = g.x;
      const double gy = g.y;
      const double gz = z_of_(g);
      m_gx(i, j, kk) = gx;
      m_gy(i, j, kk) = gy;
      if constexpr (Dim == 3) {
        m_gz(i, j, kk) = gz;
      }
      const double th =
          (m_angle != nullptr) ? (*m_angle)(i, j, kk) : p.crystal_angle;
      const auto a = evaluate_anisotropy<Dim>(p, gx, gy, gz, th);
      const double w2 = a.W * a.W;
      m_tau(i, j, kk) = a.tau;
      m_Fx(i, j, kk) = w2 * gx + a.flux[0];
      m_Fy(i, j, kk) = w2 * gy + a.flux[1];
      if constexpr (Dim == 3) {
        m_Fz(i, j, kk) = w2 * gz + a.flux[2];
      }
      if (thermal) {
        const auto l = m_lap_theta_eval(i, j, kk);
        double lap = l.xx + l.yy;
        if constexpr (Dim == 3) {
          lap += l.zz;
        }
        m_lap_theta(i, j, kk) = lap;
      } else {
        m_lap_theta(i, j, kk) = 0.0;
      }
    });
  }

  /// Stage B: `div F` and the complete phase-field right-hand side of (2).
  void stage_b_() {
    const auto &p = m_p;
    const double lam = p.lambda;
    const double mc = p.M_c;
    const double lam_el = p.lambda_el;
    const Field *dfel = m_dfel_dphi;
    m_phi.for_each_owned([&](int i, int j, int kk) {
      double div = m_dFx(i, j, kk).x + m_dFy(i, j, kk).y;
      if constexpr (Dim == 3) {
        div += m_dFz(i, j, kk).z;
      }
      const double ph = m_phi(i, j, kk);
      const double well = ph - ph * ph * ph;
      const double gwell = (1.0 - ph * ph) * (1.0 - ph * ph);
      double rhs =
          div + well - lam * gwell * (m_U(i, j, kk) + mc * m_theta(i, j, kk));
      // ---- ELASTIC HOOK -------------------------------------------------
      // Equation (2) of MODEL_SPEC.md ends with
      //     - lambda_el (1-phi^2)^2 dF_el/dphi
      // where dF_el/dphi is equation (7). `elasticity.hpp` solves (5)-(7) and
      // writes the result into a field on the same owned box; the driver
      // installs it with set_elastic_driving_force() and the line below
      // fires. Lagging the solve by n_el_substep steps is legitimate (the
      // mechanics are quasi-static) and needs no change here -- the stepper
      // simply keeps reading whatever the last solve left in the field.
      if (dfel != nullptr && lam_el != 0.0) {
        rhs -= lam_el * gwell * (*dfel)(i, j, kk);
      }
      // -------------------------------------------------------------------
      m_dphidt(i, j, kk) = rhs / m_tau(i, j, kk);
    });
  }

  /// Stage C: the solute flux `J = D_l q(phi) grad U + j_at`.
  void stage_c_() {
    const auto &p = m_p;
    const double k = p.k;
    const double at = p.at_scale * kAntiTrapCoeff * p.W0;
    m_phi.for_each_owned([&](int i, int j, int kk) {
      const auto gu = m_grad_U(i, j, kk);
      const double q = p.D_l * solute_mobility(m_phi(i, j, kk));
      const double gx = m_gx(i, j, kk);
      const double gy = m_gy(i, j, kk);
      const double gz = (Dim == 3) ? m_gz(i, j, kk) : 0.0;
      const double g2 = gx * gx + gy * gy + (Dim == 3 ? gz * gz : 0.0);
      // Unit normal, gated rather than softened: the anti-trapping current
      // carries no spare power of |grad phi|, so a soft denominator would
      // leave an O(1) direction attached to numerical noise in the bulk.
      // Below the floor the factor d_t phi is at round-off anyway.
      const double inv_gn = (g2 > kGradNormFloor2) ? 1.0 / std::sqrt(g2) : 0.0;
      const double amp =
          at * (1.0 + (1.0 - k) * m_U(i, j, kk)) * m_dphidt(i, j, kk) * inv_gn;
      m_Jx(i, j, kk) = q * gu.x + amp * gx;
      m_Jy(i, j, kk) = q * gu.y + amp * gy;
      if constexpr (Dim == 3) {
        m_Jz(i, j, kk) = q * z_of_(gu) + amp * gz;
      }
    });
  }

  /// Stage D: `div J`, then the simultaneous update of `psi`, `phi`, `theta`
  /// and the recovery `U = psi / P(phi_new)`.
  void stage_d_(double dt) {
    const auto &p = m_p;
    const double k = p.k;
    const bool thermal = p.evolve_theta;
    const bool spec_src = p.spec_source;
    const double dth = p.D_th;
    m_phi.for_each_owned([&](int i, int j, int kk) {
      double divJ = m_dJx(i, j, kk).x + m_dJy(i, j, kk).y;
      if constexpr (Dim == 3) {
        divJ += m_dJz(i, j, kk).z;
      }
      const double rate = m_dphidt(i, j, kk);
      // Conservative source of equation (3): (1/2) d_t phi. `spec_source`
      // swaps in the spec's literal (1/2)(1 + (1-k) U) d_t phi, which belongs
      // to the non-conservative form and breaks both the discrete solute
      // conservation and the anti-trapping cancellation. See the file
      // comment; it exists to be measured, not to be used.
      const double src =
          spec_src ? 0.5 * (1.0 + (1.0 - k) * m_U(i, j, kk)) * rate : 0.5 * rate;
      const double psi_new = m_psi(i, j, kk) + dt * (divJ + src);
      const double phi_new = m_phi(i, j, kk) + dt * rate;
      if (thermal) {
        m_theta(i, j, kk) += dt * (dth * m_lap_theta(i, j, kk) + 0.5 * rate);
      }
      m_psi(i, j, kk) = psi_new;
      m_phi(i, j, kk) = phi_new;
      m_U(i, j, kk) = psi_new / solute_prefactor(k, phi_new);
    });
  }

  template <class G> [[nodiscard]] static double z_of_(const G &g) noexcept {
    if constexpr (requires { g.z; }) {
      return g.z;
    } else {
      return 0.0;
    }
  }

  Stack &m_stack;
  ModelParams m_p{};
  int m_order{2};

  Field &m_phi;
  Field m_U;
  Field m_theta;
  Field m_psi;
  Field m_dphidt;
  Field m_tau;
  Field m_lap_theta;
  Field m_gx, m_gy, m_gz;
  Field m_Fx, m_Fy, m_Fz;
  Field m_Jx, m_Jy, m_Jz;

  Exchange m_ex_state;
  Exchange m_ex_flux;
  Exchange m_ex_solute;

  pfc::gradient::FDGradient<FirstDerivs<Dim>> m_grad_phi;
  pfc::gradient::FDGradient<FirstDerivs<Dim>> m_grad_U;
  pfc::gradient::FDGradient<SecondDerivs<Dim>> m_lap_theta_eval;
  pfc::gradient::FDGradient<DerivX> m_dFx;
  pfc::gradient::FDGradient<DerivY> m_dFy;
  pfc::gradient::FDGradient<DerivZ> m_dFz;
  pfc::gradient::FDGradient<DerivX> m_dJx;
  pfc::gradient::FDGradient<DerivY> m_dJy;
  pfc::gradient::FDGradient<DerivZ> m_dJz;

  const Field *m_dfel_dphi{nullptr};
  const Field *m_angle{nullptr};
};

} // namespace alloy_dendrite
