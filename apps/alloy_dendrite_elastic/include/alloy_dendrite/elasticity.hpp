// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file elasticity.hpp
 * @brief Equations (5)-(7) of `MODEL_SPEC.md` wired onto the thermo-solutal
 *        core: eigenstrain microelasticity solved spectrally, its
 *        `d f_el/d phi` handed back to the phase field.
 *
 * @details
 * ## What this file is, and what it is not
 *
 * It is *not* an elastic solver. `openpfc_apps/microelasticity.hpp` is the
 * solver -- Eshelby-validated, with a finite-difference-checked
 * `d f_el/d phi` -- and nothing here re-derives any of it. This file is the
 * adapter that makes the solver usable from inside a finite-difference time
 * loop, and the three jobs it does are the three places where a coupled
 * local-FD / global-FFT application can quietly go wrong:
 *
 *  1. **Two layouts, one grid.** The phase field lives on a *padded* FD field
 *     (`FDPaddedCPUStack`, storage halo `fd_order/2`); the elastic solver
 *     lives on flat HeFFTe inbox fields with no halo. They must describe the
 *     same owned cells or the coupling is silently wrong on every rank but
 *     zero. See @ref ElasticCoupling::require_matching_layout_.
 *  2. **Three fields, not one.** The solver wants `h`, the eigenstrain
 *     amplitude `a`, and *both* their `phi`-derivatives. Passing `nullptr`
 *     for the derivatives is accepted by the solver and yields
 *     `dfel_dphi() == 0` everywhere -- a coupled run that is silently
 *     uncoupled. @ref ElasticCoupling always supplies all four.
 *  3. **Units.** `lambda_el` is not a free knob if the elastic constants are
 *     in GPa and `U` is dimensionless. The conversion is derived below and
 *     implemented in @ref chemical_energy_scale.
 *
 * ## The model, spelled out in this application's variables
 *
 *     eps*_ij  = a(x) delta_ij,    a = h(phi) [ eps_c (U - U_ref)
 *                                             + eps_T (theta - theta_ref) ]
 *     h(phi)   = (1 + phi) / 2
 *     C(phi)   = C_solid h + C_liquid (1 - h)
 *     d a / d phi  = (1/2) [ eps_c (U - U_ref) + eps_T (theta - theta_ref) ]
 *     d h / d phi  = 1/2
 *
 * `U` and `theta` are held fixed while `phi` is differentiated, which is
 * correct: they are independent fields of the system, not functions of
 * `phi`. The partial derivative at frozen strain is also the total
 * variational derivative, because the strain is at mechanical equilibrium
 * (`delta F / delta u = -div sigma = 0`); `test_microelasticity.cpp` checks
 * that against a finite difference of the re-converged energy rather than
 * assuming it.
 *
 * ## Plane strain in 2-D comes out for free, and that is worth knowing
 *
 * A 2-D run has `nz = 1`, so every wave vector has `k_z = 0`. The Green
 * operator `Gamma_ijkl = sym(k_j G_ik k_l)` then has `Gamma_zzkl = 0`
 * identically, and with `eps_hat(0)` fixed by the macroscopic strain the
 * solve returns `eps_zz == 0` everywhere. That is **plane strain**, not
 * plane stress: the dilatational eigenstrain still has a `zz` component, so
 * `sigma_zz = -C_1122 ... a` is nonzero and does work through equation (7).
 * It is the right 2-D reduction for a dendrite in a thick sample, and it is
 * a property of the discretisation rather than something this file imposes
 * -- which is why it is written down here instead of being discovered later.
 *
 * ## The macroscopic strain, and where the spec is ambiguous
 *
 * `MODEL_SPEC.md` says `eps_hat(0) = applied macroscopic strain (zero for a
 * free body)`. Those two clauses are not the same condition. A periodic cell
 * with `eps_hat(0) = 0` is *clamped* at its mean strain: a body that
 * uniformly transforms cannot expand, so it develops a uniform stress
 * `-C : <eps*>` whose magnitude grows with the solid fraction, i.e. with time
 * and with how small the box is. A *free* body has zero mean **stress**, and
 * for a homogeneous modulus that is exactly `eps_hat(0) = <eps*>`.
 *
 * @ref MacroStrainMode therefore offers both, and the default is
 * @ref MacroStrainMode::ZeroMeanStress, because a dendrite growing into a
 * large melt is not clamped and because the clamped choice makes the elastic
 * driving force depend on the box size -- which would contaminate exactly
 * the domain-size study Stage 2 needs. The choice is not asserted to be
 * harmless: @ref ElasticReport::mean_stress_trace reports the residual mean
 * stress every solve (it is exactly zero only for a homogeneous modulus),
 * and the drivers can run both.
 *
 * ## Units: why `lambda_el = lambda` is the calibrated value
 *
 * Equation (2) is dimensionless. Its chemical term is
 * `- lambda (1-phi^2)^2 U`. In the underlying variational form the
 * dimensionless free energy is the physical one divided by the double-well
 * barrier density `H`, so a physical energy density `f` contributes
 * `(1/H) df/dphi`. Matching the chemical term: a unit of `U` corresponds to
 * a physical driving-force density
 *
 *     f_ref = L * dT_0 / T_M,     dT_0 = |m| c_l^0 (1 - k)
 *
 * (`L` latent heat per unit volume, `dT_0` the freezing range of the alloy,
 * `T_M` the melting point of the pure solvent), and therefore
 * `lambda = f_ref / H`. The elastic term is `- lambda_el (1-phi^2)^2
 * df_el/dphi`, so
 *
 *     lambda_el = 1 / H = lambda / f_ref .
 *
 * Equivalently -- and this is how the code does it -- **express the
 * stiffnesses in units of `f_ref`, and then `lambda_el = lambda`**. One
 * number, one meaning, and a `lambda_el` that is not equal to `lambda` is
 * then an explicit statement that the coupling is being scaled away from its
 * calibrated value, which is a legitimate thing to do in a sensitivity scan
 * and an illegitimate thing to do silently.
 *
 * The `(1-phi^2)^2` weight is the spec's, and it is a modelling choice
 * rather than a consequence: `df_el/dphi` already carries all of the `phi`
 * dependence, so the extra weight localises the elastic feedback to the
 * interface and suppresses it in both bulk phases. At `phi = 0` the weight
 * is 1, so the calibration above is the interface value.
 *
 * ## Lagging the solve (`n_el_substep`)
 *
 * Mechanical equilibrium is elliptic: there is no time derivative in
 * equation (5), so the displacement is slaved to the instantaneous `phi`,
 * `U` and `theta`. Physically the slaving is exact on the acoustic time
 * scale `L_box / c_sound`, which for any metal is ten or more orders of
 * magnitude below `tau0`. So the error of re-solving only every `N`-th step
 * is *not* a physical relaxation error at all -- it is purely the difference
 * between `df_el/dphi[phi(t)]` and `df_el/dphi[phi(t - n dt)]`, i.e. a
 * first-order-in-`N dt` staleness of a field that changes at the rate the
 * interface moves. It is bounded by `N dt |V| |grad(df_el/dphi)|`, and the
 * right way to size `N` is to measure the resulting change in a dendrite
 * observable rather than to argue about it. The drivers do exactly that.
 *
 * @see openpfc_apps/microelasticity.hpp for the solver and its verification
 * @see MODEL_SPEC.md equations (5)-(7)
 * @see Khachaturyan, *Theory of Structural Transformations in Solids* (1983)
 * @see Hu & Chen, *Acta Mater.* **49**, 1879 (2001)
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>

#include <openpfc_apps/microelasticity.hpp>

#include <alloy_dendrite/parameters.hpp>

namespace alloy_dendrite {

using pfc::apps::Stiffness;
using pfc::apps::Sym3;

/// Which condition fixes the `k = 0` Fourier mode of the strain.
enum class MacroStrainMode : int {
  /**
   * @brief `eps_hat(0) = 0`: the periodic cell is clamped at its mean strain.
   *
   * The literal reading of `MODEL_SPEC.md`. A uniformly transforming body
   * then carries a uniform stress proportional to the mean eigenstrain, so
   * the elastic driving force acquires a term that grows with the solid
   * fraction and shrinks with the box volume. Kept because it is the spec's
   * text and because the difference between the two modes is a measurement
   * worth having.
   */
  Clamped = 0,
  /**
   * @brief `eps_hat(0) = <a> P`: zero mean stress, i.e. a genuinely free body.
   *
   * Exact for a homogeneous modulus (`<sigma> = C(eps_bar - <eps*>)`), and
   * accurate to the modulus contrast otherwise. The default, because "free
   * body" is what the spec's parenthesis asks for and because it is the only
   * one of the two whose answer does not depend on the size of the box.
   */
  ZeroMeanStress = 1
};

/**
 * @brief Chemical free-energy density scale `f_ref = L dT_0 / T_M`.
 *
 * The physical energy density that corresponds to one unit of the
 * dimensionless supersaturation `U`. Dividing every stiffness by it is what
 * makes `lambda_el = lambda` the calibrated coupling; see the file comment.
 *
 * @param latent_heat_per_volume `L` in J/m^3
 * @param freezing_range         `dT_0 = |m| c_l^0 (1-k)` in K
 * @param melting_point          `T_M` in K
 */
[[nodiscard]] inline constexpr double
chemical_energy_scale(double latent_heat_per_volume, double freezing_range,
                      double melting_point) noexcept {
  return latent_heat_per_volume * freezing_range / melting_point;
}

/**
 * @brief Physical inputs of equations (5)-(7), already non-dimensionalised.
 *
 * Stiffnesses are in units of @ref chemical_energy_scale; strains are
 * dimensionless; `eps_c` and `eps_T` are derivatives of the eigenstrain with
 * respect to `U` and `theta`, which are themselves dimensionless. The
 * drivers carry the provenance of the numbers and the conversion from SI;
 * this struct only carries the numbers the equations use.
 */
struct ElasticParams {
  /// Solid stiffness, cubic, `<100>` along the grid, in units of `f_ref`.
  Stiffness c_solid{};
  /**
   * @brief `mu_l / mu_s` for @ref pfc::apps::soft_liquid.
   *
   * A liquid has no shear modulus, and a phase-field elastic solve cannot
   * use zero: the local stiffness becomes singular in two channels and every
   * fixed point of this family has contraction factor 1. It is a
   * regularisation parameter. The default here is the solver's own
   * (`kDefaultLiquidShearFraction = 0.05`, contrast 20), whose measured cost
   * is documented there; the drivers scan it and report the iteration count,
   * because "we picked 0.05" is not a justification and "0.01 costs twice
   * the iterations and moves the tip velocity by X" is.
   */
  double mu_liquid_fraction{pfc::apps::kDefaultLiquidShearFraction};
  /// `K_l / K_s`. 1 by default: liquids are nearly as stiff in bulk as solids.
  double bulk_liquid_fraction{1.0};

  /// `d eps* / d U` inside the solid -- the solutal (Vegard) misfit.
  double eps_c{0.0};
  /// `d eps* / d theta` inside the solid -- the thermal misfit.
  double eps_T{0.0};
  /// Reference supersaturation at which the solid is unstrained.
  double U_ref{0.0};
  /// Reference undercooling at which the solid is unstrained.
  double theta_ref{0.0};

  /// Which condition fixes `eps_hat(0)`; see @ref MacroStrainMode.
  MacroStrainMode macro_strain{MacroStrainMode::ZeroMeanStress};
  /// `EyreMilton` unless a comparison against the reference scheme is wanted.
  pfc::apps::MicroelasticityScheme scheme{
      pfc::apps::MicroelasticityScheme::EyreMilton};
  /// Relative polarisation change at which the fixed point stops.
  double tol_el{1.0e-6};
  /// Hard cap on Green-operator applications per solve.
  int n_el_iter{50};
  /// Re-solve every `n_el_substep` phase-field steps. 1 = every step.
  int n_el_substep{1};
  /// Start each solve from the previous one. See @ref ElasticCoupling.
  bool warm_start{true};
};

/// What one elastic solve cost and what it produced.
struct ElasticReport {
  int iterations{0};
  double residual{0.0};
  bool converged{true};
  /// `int f_el dV` over the whole domain.
  double total_energy{0.0};
  /// Global max `|d f_el / d phi|`.
  double max_dfel_dphi{0.0};
  /// `<sigma_xx + sigma_yy + sigma_zz> / 3`, the residual mean pressure.
  /// Exactly zero only for a homogeneous modulus under
  /// @ref MacroStrainMode::ZeroMeanStress; reported so the approximation is
  /// visible rather than assumed.
  double mean_stress_trace{0.0};
};

/**
 * @brief Couples @ref pfc::apps::EigenstrainMicroelasticity to a
 *        `Stepper<Dim>` running on an `FDPaddedCPUStack`.
 *
 * Owns the FFT, the solver, the four scalar inputs the solver needs, and the
 * padded field that the stepper reads through
 * `Stepper::set_elastic_driving_force`. One instance per run; construct
 * after the stack and before the time loop.
 *
 * Non-copyable and non-movable, like everything else that holds an FFT plan
 * and fields that point at each other.
 */
class ElasticCoupling {
public:
  using PaddedField = pfc::data::Field<double, pfc::HostSpace>;
  using FlatField = pfc::data::Field<double>;
  using Stack = pfc::sim::stacks::FDPaddedCPUStack;

  ElasticCoupling(const ElasticCoupling &) = delete;
  ElasticCoupling &operator=(const ElasticCoupling &) = delete;
  ElasticCoupling(ElasticCoupling &&) = delete;
  ElasticCoupling &operator=(ElasticCoupling &&) = delete;

  /**
   * @param stack   The same stack the `Stepper` runs on. Its decomposition is
   *                reused verbatim for the FFT, which is the only way to be
   *                sure the two layouts agree on more than one rank.
   * @param params  Equations (5)-(7).
   * @param rank    Caller rank on @p comm.
   * @param comm    Communicator; must be the stack's.
   */
  ElasticCoupling(Stack &stack, const ElasticParams &params, int rank,
                  MPI_Comm comm)
      : m_params(params), m_comm(comm),
        // The FFT is built from the *stack's own* decomposition rather than
        // from `nproc`. `SpectralCPUStack` would build its own via
        // `spectral_fft_proc_grid`, which agrees with the FD stack's
        // `min_surface_proc_grid` only below nine ranks; reusing the object
        // removes the coincidence from the contract. The layout check in the
        // body then verifies what is left (HeFFTe is free to return an inbox
        // that is not the decomposition box).
        m_fft(pfc::fft::create(stack.decomposition(), rank, comm, 0)),
        m_h(pfc::data::field_from_inbox<double>(stack.domain(),
                                                m_fft.get_inbox_bounds())),
        m_amp(pfc::data::field_from_inbox<double>(stack.domain(),
                                                  m_fft.get_inbox_bounds())),
        m_dh(pfc::data::field_from_inbox<double>(stack.domain(),
                                                 m_fft.get_inbox_bounds())),
        m_damp(pfc::data::field_from_inbox<double>(stack.domain(),
                                                   m_fft.get_inbox_bounds())),
        m_dfel(stack.make_field()), m_solver(stack.domain(), m_fft,
                                             make_solver_params_(params, comm)) {
    if (params.n_el_substep < 1) {
      throw std::invalid_argument(
          "ElasticCoupling: n_el_substep must be >= 1 (1 = solve every step)");
    }
    require_matching_layout_(stack);
    // dh/dphi is the constant 1/2 for h = (1+phi)/2. Filled once: writing it
    // every step would be a per-cell store of a compile-time constant.
    std::fill(m_dh.vec().begin(), m_dh.vec().end(), 0.5);
    m_dh.note_host_write();
    std::fill(m_dfel.vec().begin(), m_dfel.vec().end(), 0.0);
    m_dfel.note_host_write();
  }

  /// The `d f_el / d phi` field, in the stepper's padded layout. Hand this to
  /// `Stepper::set_elastic_driving_force`. Zero until the first @ref solve.
  [[nodiscard]] const PaddedField &driving_force() const noexcept {
    return m_dfel;
  }

  [[nodiscard]] const pfc::apps::EigenstrainMicroelasticity &
  solver() const noexcept {
    return m_solver;
  }
  [[nodiscard]] const ElasticParams &params() const noexcept { return m_params; }
  /// Liquid stiffness actually in use, derived from the solid by `soft_liquid`.
  [[nodiscard]] Stiffness c_liquid() const noexcept {
    return pfc::apps::soft_liquid(m_params.c_solid, m_params.mu_liquid_fraction,
                                  m_params.bulk_liquid_fraction);
  }

  /// True when step @p step (1-based) is one the solve runs on.
  [[nodiscard]] bool due(int step) const noexcept {
    return (step % m_params.n_el_substep) == 0;
  }

  /**
   * @brief Rebuild the eigenstrain, solve equilibrium, refresh the driving
   *        force.
   *
   * @param phi,U,theta  Current state, in the stepper's padded layout.
   *
   * Every call re-solves from scratch in the sense that the eigenstrain is
   * rebuilt from the current fields; `warm_start` only chooses the *initial
   * iterate*, which is the previous converged strain. That is legitimate for
   * a fixed point (it changes how many iterations are needed, not what they
   * converge to) and it is where most of the cost goes away once the
   * interface is moving less than a cell per step.
   */
  ElasticReport solve(const PaddedField &phi, const PaddedField &U,
                      const PaddedField &theta) {
    const double ec = m_params.eps_c;
    const double et = m_params.eps_T;
    const double ur = m_params.U_ref;
    const double tr = m_params.theta_ref;
    double amp_local = 0.0;
    m_h.for_each_owned([&](int i, int j, int k) {
      const double ph = phi(i, j, k);
      const double hv = 0.5 * (1.0 + ph);
      // s is the bracket of equation (5); a = h s and da/dphi = s/2, both
      // at frozen U and theta.
      const double s = ec * (U(i, j, k) - ur) + et * (theta(i, j, k) - tr);
      m_h(i, j, k) = hv;
      m_amp(i, j, k) = hv * s;
      m_damp(i, j, k) = 0.5 * s;
      amp_local += hv * s;
    });
    m_h.note_host_write();
    m_amp.note_host_write();
    m_damp.note_host_write();

    if (m_params.macro_strain == MacroStrainMode::ZeroMeanStress) {
      double amp_sum = 0.0;
      MPI_Allreduce(&amp_local, &amp_sum, 1, MPI_DOUBLE, MPI_SUM, m_comm);
      const auto g = m_h.global_size();
      const double ncells = static_cast<double>(g[0]) *
                            static_cast<double>(g[1]) * static_cast<double>(g[2]);
      const double mean_amp = amp_sum / ncells;
      Sym3 bar;
      for (int c = 0; c < pfc::apps::kSymComponents; ++c) {
        bar[c] = mean_amp * m_solver.params().eigenstrain_pattern[c];
      }
      m_solver.params().applied_strain = bar;
    }

    const auto rep = m_solver.solve(m_h, m_amp, &m_dh, &m_damp);

    // Back to the padded layout. The owned boxes are identical (checked at
    // construction), so this is an index-for-index copy; the halo of m_dfel
    // is never read by the stepper, which only touches owned cells.
    const auto &src = m_solver.dfel_dphi();
    double local_max = 0.0;
    m_dfel.for_each_owned([&](int i, int j, int k) {
      const double v = src(i, j, k);
      m_dfel(i, j, k) = v;
      local_max = std::fmax(local_max, std::fabs(v));
    });
    m_dfel.note_host_write();

    double p_local = 0.0;
    const auto &sig = m_solver.stress();
    for (std::size_t q = 0; q < sig[0].size(); ++q) {
      p_local += (sig[0].data()[q] + sig[1].data()[q] + sig[2].data()[q]) / 3.0;
    }

    ElasticReport out;
    out.iterations = rep.iterations;
    out.residual = rep.residual;
    out.converged = rep.converged;
    out.total_energy = m_solver.total_elastic_energy();
    MPI_Allreduce(&local_max, &out.max_dfel_dphi, 1, MPI_DOUBLE, MPI_MAX, m_comm);
    double p_sum = 0.0;
    MPI_Allreduce(&p_local, &p_sum, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    const auto g = m_h.global_size();
    out.mean_stress_trace =
        p_sum / (static_cast<double>(g[0]) * static_cast<double>(g[1]) *
                 static_cast<double>(g[2]));
    return out;
  }

private:
  static pfc::apps::MicroelasticityParams
  make_solver_params_(const ElasticParams &p, MPI_Comm comm) {
    pfc::apps::MicroelasticityParams q;
    q.c_solid = p.c_solid;
    q.c_liquid =
        pfc::apps::soft_liquid(p.c_solid, p.mu_liquid_fraction, p.bulk_liquid_fraction);
    q.eigenstrain_pattern = Sym3::identity(); // dilatational, equation (5)
    q.applied_strain = Sym3{};
    q.scheme = p.scheme;
    q.tol_el = p.tol_el;
    q.n_el_iter = p.n_el_iter;
    q.warm_start = p.warm_start;
    q.comm = comm;
    return q;
  }

  /**
   * @brief Refuse to run unless the FD and FFT layouts describe the same cells.
   *
   * The two stacks index their fields the same way, `(i, j, k)` local with
   * `(0,0,0)` the first owned cell, so the copy in @ref solve is correct
   * *provided* the owned boxes coincide. They do when the FFT is built from
   * the stack's decomposition and HeFFTe returns that decomposition's box as
   * its real-space inbox, which is the normal case. If some future backend
   * or plan option reshapes the inbox, the copy would silently mix up cells
   * on every rank but one -- a bug that a single-rank test cannot see and
   * that a multi-rank run would report as "the elastic field looks odd".
   * Cheap to check once, so check it.
   */
  void require_matching_layout_(const Stack &stack) const {
    const auto &fd = stack.u().box();
    const auto el = m_fft.get_inbox_bounds();
    for (int d = 0; d < 3; ++d) {
      if (fd.low[d] != el.low[d] || fd.high[d] != el.high[d]) {
        throw std::runtime_error(
            "ElasticCoupling: the FD owned box and the FFT real-space inbox "
            "differ on axis " +
            std::to_string(d) + " (FD [" + std::to_string(fd.low[d]) + ", " +
            std::to_string(fd.high[d]) + "], FFT [" + std::to_string(el.low[d]) +
            ", " + std::to_string(el.high[d]) +
            "]). The elastic coupling copies index for index between the two "
            "and would be wrong. Reduce the rank count or force a matching "
            "process grid with OPENPFC_FFT_PROC_GRID.");
      }
    }
  }

  ElasticParams m_params;
  MPI_Comm m_comm;
  pfc::fft::CPUFFT m_fft;
  FlatField m_h;
  FlatField m_amp;
  FlatField m_dh;
  FlatField m_damp;
  PaddedField m_dfel;
  pfc::apps::EigenstrainMicroelasticity m_solver;
};

} // namespace alloy_dendrite
