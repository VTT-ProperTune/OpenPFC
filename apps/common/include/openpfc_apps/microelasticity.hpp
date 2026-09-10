// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file microelasticity.hpp
 * @brief Quasi-static 3-D eigenstrain microelasticity on the existing FFT stack.
 *
 * @details
 * A phase field that changes the local lattice parameter loads the solid it
 * grows into. The load is *not* a time-integrated field: mechanical
 * equilibrium relaxes on the acoustic time scale, which is many orders of
 * magnitude faster than diffusion, so the displacement is slaved to the
 * instantaneous phase field and must be re-solved from scratch every time
 * step. Adding a displacement to the ETD state vector would be wrong physics
 * *and* an intolerable stiffness; what is needed instead is an elliptic solve
 * that runs inside one time step and hands back a body force on the phase
 * field. That is what this header is.
 *
 * \f[
 *   \nabla\cdot\boldsymbol\sigma = 0,\qquad
 *   \boldsymbol\sigma = \mathbf{C}(\phi):\bigl(\boldsymbol\varepsilon(\mathbf u)
 *                                              - \boldsymbol\varepsilon^{*}\bigr),
 *   \qquad
 *   \varepsilon_{ij}(\mathbf u) = \tfrac12\,(\partial_i u_j + \partial_j u_i).
 * \f]
 *
 * ## Why Fourier, and why an iteration on top of it
 *
 * For a *homogeneous* stiffness the equilibrium equation is diagonal in
 * Fourier space: the Green operator of the reference medium inverts it in one
 * pass. Writing \f$\mathbf C = \mathbf C_0 + \Delta\mathbf C\f$ and collecting
 * everything that is not \f$\mathbf C_0:\boldsymbol\varepsilon\f$ into a
 * polarisation
 *
 * \f[
 *   \boldsymbol\tau \;=\; \boldsymbol\sigma - \mathbf C_0:\boldsymbol\varepsilon
 *     \;=\; \underbrace{(\mathbf C-\mathbf C_0):
 *            (\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*})}_{\text{modulus
 * contrast}}
 *        \;-\;\underbrace{\mathbf C_0:\boldsymbol\varepsilon^{*}}_{\text{eigenstress
 * drive}},
 * \f]
 *
 * equilibrium becomes \f$\nabla\cdot(\mathbf C_0:\boldsymbol\varepsilon
 * + \boldsymbol\tau) = 0\f$, which in Fourier space is
 *
 * \f[
 *   \hat\varepsilon_{ij}(\mathbf k) =
 *     -\,\hat\Gamma_{ijkl}(\mathbf k)\,\hat\tau_{kl}(\mathbf k),
 *   \qquad \mathbf k \neq \mathbf 0,
 *   \qquad
 *   \hat{\boldsymbol\varepsilon}(\mathbf 0) = \bar{\boldsymbol\varepsilon}
 *   \ \text{(applied macroscopic strain, zero by default)} ,
 * \f]
 *
 * with \f$\hat\Gamma\f$ built from the acoustic (Christoffel) tensor of
 * \f$\mathbf C_0\f$,
 * \f$\;(G^{-1})_{ik} = C^0_{ijkl}k_jk_l,\;
 * \Gamma_{ijkl} = \mathrm{sym}_{ij}\mathrm{sym}_{kl}\,[\,k_j G_{ik} k_l\,]\f$
 * (Khachaturyan 1983).
 *
 * \f$\boldsymbol\tau\f$ depends on \f$\boldsymbol\varepsilon\f$ whenever the
 * modulus is inhomogeneous, and here it always is: the liquid is soft and the
 * solid is stiff, that contrast is the whole reason the elastic field is
 * interesting, and it is exactly what makes this *not* a one-shot solve. The
 * fix is the Hu & Chen (2001) polarisation fixed point — evaluate
 * \f$\boldsymbol\tau\f$ in real space where the modulus lives, apply
 * \f$\Gamma\f$ in Fourier space where the Green operator lives, repeat. The
 * scheme is a Neumann series whose contraction factor is
 * \f$\lVert(\mathbf C-\mathbf C_0)\mathbf C_0^{-1}\rVert\f$, which is why
 * \f$\mathbf C_0\f$ defaults to the Voigt (arithmetic) average of the two
 * phases: for a two-phase medium that is the choice that minimises the factor,
 * giving \f$(r-1)/(r+1)\f$ for a stiffness ratio \f$r\f$. Measured on a
 * \f$32^3\f$ grid with a tanh solid sphere (R = 7, w = 1.5), isotropic
 * \f$\nu = 0.3\f$, cold start, `tol_el = 1e-6`:
 *
 * | \f$C_{\text{solid}}/C_{\text{liquid}}\f$ | 1 | 2 | 4 | 10 |
 * |---|---|---|---|---|
 * | iterations | 1 | 11 | 22 | 53 |
 * | observed contraction | — | 0.265 | 0.524 | 0.767 |
 * | \f$(r-1)/(r+1)\f$ | — | 0.333 | 0.600 | 0.818 |
 *
 * The bound is tight enough to be predictive, and the consequence is blunt:
 * the default cap `n_el_iter = 20` covers a stiffness ratio of about 3 and no
 * more. That is a property of the plain fixed point, not of this
 * implementation. A caller running at ratio 10 must raise the cap (and pay
 * ~53 × 12 transforms per solve), lag the solve over several phase-field
 * steps (`n_el_substep` in the spec — it is quasi-static, so this is
 * legitimate), or warm-start from the previous step, which is the default and
 * cuts the count sharply once the interface is only moving a cell per step.
 * `MicroelasticityReport::converged` is returned rather than thrown, so the driver
 * decides.
 *
 * ## What the iteration count means here
 *
 * The spec's stopping test is the relative change of the strain between
 * successive passes. This implementation tests the relative change of the
 * *polarisation* instead, measured in real space **before** the transforms.
 * The two are the same fixed point one step apart —
 * \f$\varepsilon_{n+1}-\varepsilon_n = -\Gamma:(\tau_n - \tau_{n-1})\f$ with
 * \f$\Gamma\f$ linear and bounded — but testing on \f$\tau\f$ costs no FFTs,
 * so the pass that merely *confirms* convergence is free. That is what makes
 * the homogeneous case cost exactly one \f$\Gamma\f$ application and report
 * `iterations == 1`: with \f$\mathbf C \equiv \mathbf C_0\f$ the polarisation
 * collapses to \f$-\mathbf C_0:\boldsymbol\varepsilon^{*}\f$, independent of
 * \f$\boldsymbol\varepsilon\f$, so the second pass reproduces it exactly and
 * exits before touching the FFT. `MicroelasticityReport::residual_history` records
 * every measured residual so a caller (or a test) can check monotonicity.
 *
 * ## Eigenstrain model
 *
 * \f$\boldsymbol\varepsilon^{*}(\mathbf x) = a(\mathbf x)\,\mathbf P\f$: one
 * scalar amplitude field times one constant symmetric pattern tensor
 * (`MicroelasticityParams::eigenstrain_pattern`, identity by default, i.e.
 * dilatational).
 * That covers the capstone's
 * \f$\varepsilon^{*}_{ij} = h(\phi)\,[\varepsilon_c(U-U_{\text{ref}}) +
 * \varepsilon_T(\theta-\theta_{\text{ref}})]\,\delta_{ij}\f$ and any
 * single-variant transformation strain, and it keeps the per-call state at two
 * scalar fields instead of twelve. A multi-variant eigenstrain
 * \f$\sum_v a_v \mathbf P_v\f$ is *not* supported; the spec lists a
 * multi-variant orientation field as a non-goal.
 *
 * ## Energy and the phase-field feedback
 *
 * \f[
 *   f_{\mathrm{el}} = \tfrac12(\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}):
 *                      \mathbf
 * C:(\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}),
 *   \qquad
 *   \frac{\partial f_{\mathrm{el}}}{\partial \phi} =
 *     -\,\boldsymbol\sigma:\frac{\partial\boldsymbol\varepsilon^{*}}{\partial\phi}
 *     + \tfrac12(\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}):
 *        \frac{\partial\mathbf C}{\partial\phi}:
 *        (\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}).
 * \f]
 *
 * Both terms are returned. The first is the transformation-work term and
 * dominates; the second is the modulus-contrast term, it is small, it is the
 * one an implementation is tempted to drop, and dropping it makes the elastic
 * driving force wrong wherever the stiffness varies — which is precisely the
 * interface, i.e. the only place the phase field cares. Note that this
 * *partial* derivative at frozen \f$\boldsymbol\varepsilon\f$ is also the
 * *total* variational derivative of the elastic energy functional, because the
 * strain is at equilibrium and \f$\delta F/\delta\mathbf u = -\nabla\cdot
 * \boldsymbol\sigma = 0\f$. `test_microelasticity.cpp` checks that against a
 * finite difference of the fully re-converged energy rather than assuming it.
 *
 * ## Scope — what this deliberately does not do
 *
 * - **Host only.** Every loop here is a plain host loop over
 *   `pfc::data::Field<double>` and the transforms go through
 *   `pfc::fft::IHostFFT`. `spectral_flux.hpp` is templated on `MemorySpace`
 *   because its elementwise work already had CUDA/HIP kernels behind
 *   `SpectralETDOps`; the pointwise work here is a six-component symmetric
 *   tensor contraction with a spatially varying stiffness, which has no such
 *   kernel yet. A device path is a separate piece of work, not a template
 *   parameter away.
 * - **Periodic only.** The Green operator *is* the periodic boundary
 *   condition. A free surface or a clamped face needs a different solver.
 * - **Small strain, no plasticity, no finite rotation.** Linear kinematics
 *   throughout.
 * - **No displacement output.** Only the strain, stress, energy density and
 *   \f$\partial f_{\mathrm{el}}/\partial\phi\f$ are stored; \f$\mathbf u\f$ is
 *   an intermediate and nothing downstream in the capstone reads it.
 * - **Cubic or isotropic stiffness, crystal axes aligned with the grid.** A
 *   rotated or lower-symmetry \f$\mathbf C\f$ would need the general 21-constant
 *   contraction; `Stiffness` is deliberately three numbers.
 *
 * ## MPI
 *
 * Rank-agnostic. The transforms are HeFFTe's, the pointwise work is local, and
 * the only global coupling is the convergence reduction (`MPI_Allreduce` on the
 * max norms) and the \f$\mathbf k=\mathbf 0\f$ mode, which is written by
 * whichever rank owns outbox index (0,0,0). The answer is bit-comparable
 * across decompositions up to FFT round-off.
 *
 * @see Khachaturyan, *Theory of Structural Transformations in Solids* (1983)
 * @see Hu & Chen, *Acta Mater.* **49**, 1879 (2001)
 * @see Eshelby, *Proc. R. Soc. A* **241**, 376 (1957) — the test oracle
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

namespace pfc::apps {

/**
 * @brief Component order of every symmetric second-rank tensor in this header.
 *
 * These are **tensor** components, not Voigt engineering components: the shear
 * entries are \f$\varepsilon_{yz}\f$, not \f$\gamma_{yz}=2\varepsilon_{yz}\f$.
 * Mixing the two conventions is the classic factor-of-two bug in elasticity
 * code, so the engineering form never appears anywhere below.
 */
enum SymIndex : int {
  SYM_XX = 0,
  SYM_YY = 1,
  SYM_ZZ = 2,
  SYM_YZ = 3,
  SYM_XZ = 4,
  SYM_XY = 5
};

/// Number of independent components of a symmetric second-rank tensor.
inline constexpr int kSymComponents = 6;

/// A symmetric second-rank tensor, tensor (not engineering) shear components.
struct Sym3 {
  std::array<double, kSymComponents> c{};

  [[nodiscard]] double &operator[](int i) noexcept {
    return c[static_cast<std::size_t>(i)];
  }
  [[nodiscard]] double operator[](int i) const noexcept {
    return c[static_cast<std::size_t>(i)];
  }

  /// The identity \f$\delta_{ij}\f$ — the dilatational eigenstrain pattern.
  [[nodiscard]] static Sym3 identity() noexcept {
    return Sym3{{1.0, 1.0, 1.0, 0.0, 0.0, 0.0}};
  }
  [[nodiscard]] double trace() const noexcept { return c[0] + c[1] + c[2]; }
};

/// Double contraction \f$A_{ij}B_{ij}\f$ (the off-diagonals count twice).
[[nodiscard]] inline double ddot(const Sym3 &a, const Sym3 &b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] +
         2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5]);
}

/**
 * @brief Cubic stiffness in the crystal frame, grid axes = <100>.
 *
 * Three constants in the tensor convention \f$c_{11}=C_{1111}\f$,
 * \f$c_{12}=C_{1122}\f$, \f$c_{44}=C_{1212}\f$, so that
 * \f$\sigma_{xy} = 2c_{44}\varepsilon_{xy}\f$. Isotropy is the special case
 * \f$c_{44} = (c_{11}-c_{12})/2\f$ and is reachable through `isotropic()`;
 * `zener()` reports how far from it a given set of constants is.
 */
struct Stiffness {
  double c11{1.0};
  double c12{0.0};
  double c44{0.5};

  /// Lamé form: \f$c_{11}=\lambda+2\mu,\ c_{12}=\lambda,\ c_{44}=\mu\f$.
  [[nodiscard]] static Stiffness from_lame(double lambda, double mu) noexcept {
    return Stiffness{lambda + 2.0 * mu, lambda, mu};
  }

  /// Isotropic from Young's modulus and Poisson ratio.
  [[nodiscard]] static Stiffness isotropic(double youngs, double poisson) {
    if (poisson <= -1.0 || poisson >= 0.5) {
      throw std::invalid_argument(
          "Stiffness::isotropic: Poisson ratio must lie in (-1, 1/2)");
    }
    const double mu = youngs / (2.0 * (1.0 + poisson));
    const double lambda =
        youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    return from_lame(lambda, mu);
  }

  [[nodiscard]] static Stiffness cubic(double c11_, double c12_,
                                       double c44_) noexcept {
    return Stiffness{c11_, c12_, c44_};
  }

  /// Zener anisotropy ratio \f$2c_{44}/(c_{11}-c_{12})\f$; 1 means isotropic.
  [[nodiscard]] double zener() const noexcept { return 2.0 * c44 / (c11 - c12); }

  /// Bulk modulus \f$(c_{11}+2c_{12})/3\f$ (exact for cubic: dilatation is
  /// isotropic).
  [[nodiscard]] double bulk_modulus() const noexcept {
    return (c11 + 2.0 * c12) / 3.0;
  }

  /// \f$\sigma_{ij} = C_{ijkl}\varepsilon_{kl}\f$.
  [[nodiscard]] Sym3 contract(const Sym3 &e) const noexcept {
    Sym3 s;
    s[SYM_XX] = c11 * e[SYM_XX] + c12 * (e[SYM_YY] + e[SYM_ZZ]);
    s[SYM_YY] = c11 * e[SYM_YY] + c12 * (e[SYM_XX] + e[SYM_ZZ]);
    s[SYM_ZZ] = c11 * e[SYM_ZZ] + c12 * (e[SYM_XX] + e[SYM_YY]);
    s[SYM_YZ] = 2.0 * c44 * e[SYM_YZ];
    s[SYM_XZ] = 2.0 * c44 * e[SYM_XZ];
    s[SYM_XY] = 2.0 * c44 * e[SYM_XY];
    return s;
  }

  /// Componentwise \f$\alpha A + \beta B\f$; the Voigt average is `blend(a, .5, b,
  /// .5)`.
  [[nodiscard]] static Stiffness blend(const Stiffness &a, double wa,
                                       const Stiffness &b, double wb) noexcept {
    return Stiffness{wa * a.c11 + wb * b.c11, wa * a.c12 + wb * b.c12,
                     wa * a.c44 + wb * b.c44};
  }
};

/// Configuration of the fixed point and the reference medium.
struct MicroelasticityParams {
  Stiffness c_solid{};
  Stiffness c_liquid{};
  /// Constant pattern \f$\mathbf P\f$ in \f$\varepsilon^{*} = a(\mathbf x)\mathbf
  /// P\f$.
  Sym3 eigenstrain_pattern{Sym3::identity()};
  /// \f$\hat\varepsilon(\mathbf 0)\f$ — zero is a free (unloaded) periodic body.
  Sym3 applied_strain{};
  /// Relative polarisation change at which the fixed point is declared converged.
  double tol_el{1.0e-6};
  /// Hard cap on \f$\Gamma\f$ applications.
  int n_el_iter{20};
  /**
   * @brief Reference medium. Left at its default (all zeros) the Voigt
   *        average \f$(\mathbf C_s+\mathbf C_l)/2\f$ is used, which is the
   *        contraction-optimal choice for a two-phase medium.
   */
  Stiffness reference{0.0, 0.0, 0.0};
  /// Reuse the previous solution as the initial iterate (big win in a time loop).
  bool warm_start{true};
  MPI_Comm comm{MPI_COMM_WORLD};
};

/// Outcome of one `solve()`.
struct MicroelasticityReport {
  /// Number of \f$\Gamma\f$ applications actually performed.
  int iterations{0};
  /// Last measured relative polarisation change.
  double residual{0.0};
  bool converged{false};
  /// One entry per measured pass, oldest first; monotone for a contracting map.
  std::vector<double> residual_history{};
};

/**
 * @brief Quasi-static eigenstrain microelasticity solver, host / periodic.
 *
 * Construct once per `Domain`+FFT (it precomputes the Green operator over the
 * local outbox) and call `solve()` every time the phase field moves.
 */
class EigenstrainMicroelasticity {
public:
  using RealField = pfc::data::Field<double>;
  using Complex = std::complex<double>;
  using ComplexField = pfc::data::Field<Complex>;
  using SymRealFields = std::array<RealField, kSymComponents>;

  EigenstrainMicroelasticity(const pfc::Domain &domain, pfc::fft::IHostFFT &fft,
                             MicroelasticityParams params)
      : m_fft(fft), m_params(params),
        m_c0(is_zero(params.reference)
                 ? Stiffness::blend(params.c_solid, 0.5, params.c_liquid, 0.5)
                 : params.reference),
        m_strain(make_sym(domain, fft)), m_stress(make_sym(domain, fft)),
        m_tau(make_sym(domain, fft)), m_tau_prev(make_sym(domain, fft)),
        m_f_el(pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_dfel_dphi(
            pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())) {
    if (m_params.tol_el < 0.0) {
      throw std::invalid_argument("EigenstrainMicroelasticity: tol_el must be >= 0");
    }
    if (m_params.n_el_iter < 1) {
      throw std::invalid_argument(
          "EigenstrainMicroelasticity: n_el_iter must be >= 1");
    }
    for (int c = 0; c < kSymComponents; ++c) {
      m_hat[static_cast<std::size_t>(c)] =
          ComplexField(domain, fft.get_outbox_bounds(), 0);
    }
    m_n_local = m_strain[0].size();
    m_n_outbox = fft.size_outbox();
    if (m_strain[0].size() != fft.size_inbox()) {
      throw std::invalid_argument(
          "EigenstrainMicroelasticity: field inbox size does not match the FFT");
    }
    build_green_operator(domain, fft);
  }

  /// Reference medium actually in use (Voigt average unless overridden).
  [[nodiscard]] const Stiffness &reference() const noexcept { return m_c0; }
  [[nodiscard]] const MicroelasticityParams &params() const noexcept {
    return m_params;
  }

  /// Mutable parameters — a driver may retune `tol_el`/`n_el_iter` between steps.
  [[nodiscard]] MicroelasticityParams &params() noexcept { return m_params; }

  /// Discard the warm start (next `solve()` begins from the applied strain).
  void reset() noexcept {
    for (auto &f : m_strain) {
      std::fill(f.vec().begin(), f.vec().end(), 0.0);
    }
    m_has_solution = false;
  }

  [[nodiscard]] const SymRealFields &strain() const noexcept { return m_strain; }
  [[nodiscard]] const SymRealFields &stress() const noexcept { return m_stress; }
  [[nodiscard]] const RealField &elastic_energy_density() const noexcept {
    return m_f_el;
  }
  /// \f$\partial f_{\mathrm{el}}/\partial\phi\f$; only filled when `solve()` got
  /// the two derivative fields.
  [[nodiscard]] const RealField &dfel_dphi() const noexcept { return m_dfel_dphi; }

  /**
   * @brief Solve equilibrium for the given modulus interpolation and eigenstrain.
   *
   * @param h        \f$h(\phi)\in[0,1]\f$; \f$\mathbf C = h\mathbf C_s +
   * (1-h)\mathbf C_l\f$
   * @param amp      \f$a(\mathbf x)\f$ in \f$\varepsilon^{*} = a\,\mathbf P\f$
   * @param dh_dphi  \f$\partial h/\partial\phi\f$, or `nullptr` to skip eq. (7)
   * @param damp_dphi \f$\partial a/\partial\phi\f$, or `nullptr` to skip eq. (7)
   *
   * On return `strain()`, `stress()` and `elastic_energy_density()` are filled;
   * `dfel_dphi()` too when both derivative fields were supplied.
   */
  MicroelasticityReport solve(const RealField &h, const RealField &amp,
                              const RealField *dh_dphi = nullptr,
                              const RealField *damp_dphi = nullptr) {
    check_same_size(h, "h");
    check_same_size(amp, "amp");
    if (dh_dphi != nullptr) check_same_size(*dh_dphi, "dh_dphi");
    if (damp_dphi != nullptr) check_same_size(*damp_dphi, "damp_dphi");

    if (!m_params.warm_start || !m_has_solution) {
      for (int c = 0; c < kSymComponents; ++c) {
        std::fill(m_strain[static_cast<std::size_t>(c)].vec().begin(),
                  m_strain[static_cast<std::size_t>(c)].vec().end(),
                  m_params.applied_strain[c]);
      }
    }

    MicroelasticityReport report;
    for (int it = 1; it <= m_params.n_el_iter; ++it) {
      build_polarisation(h, amp);
      if (it > 1) {
        const double res = relative_change();
        report.residual = res;
        report.residual_history.push_back(res);
        if (res < m_params.tol_el) {
          report.converged = true;
          break;
        }
      }
      swap_tau();
      apply_green_operator();
      report.iterations = it;
    }
    if (!report.converged) {
      // The cap was hit before the confirming pass. Measure once more so the
      // caller sees an honest residual rather than the stale previous one; it
      // is a real-space pass, no transforms.
      build_polarisation(h, amp);
      const double res = relative_change();
      report.residual = res;
      report.residual_history.push_back(res);
      report.converged = res < m_params.tol_el;
    }

    m_has_solution = true;
    finalise(h, amp, dh_dphi, damp_dphi);
    return report;
  }

  /**
   * @brief \f$\int f_{\mathrm{el}}\,\mathrm dV\f$ over the whole domain.
   *
   * Reduced across `params().comm`, so every rank gets the same number.
   */
  [[nodiscard]] double total_elastic_energy() const {
    const auto &s = m_f_el.spacing();
    const double cell = s[0] * s[1] * s[2];
    double local = 0.0;
    for (std::size_t i = 0; i < m_n_local; ++i) local += m_f_el.data()[i];
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, m_params.comm);
    return global * cell;
  }

  /// Local stiffness at cell @p i, i.e. \f$h\mathbf C_s + (1-h)\mathbf C_l\f$.
  [[nodiscard]] Stiffness stiffness_at(double h_value) const noexcept {
    return Stiffness::blend(m_params.c_solid, h_value, m_params.c_liquid,
                            1.0 - h_value);
  }

private:
  static bool is_zero(const Stiffness &s) noexcept {
    return s.c11 == 0.0 && s.c12 == 0.0 && s.c44 == 0.0;
  }

  static SymRealFields make_sym(const pfc::Domain &domain, pfc::fft::IHostFFT &fft) {
    const auto box = fft.get_inbox_bounds();
    return SymRealFields{pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box)};
  }

  void check_same_size(const RealField &f, const char *what) const {
    if (f.size() != m_n_local) {
      throw std::invalid_argument(
          std::string("EigenstrainMicroelasticity::solve: '") + what +
          "' has the wrong local size");
    }
  }

  /**
   * @brief Precompute \f$\mathbf G(\mathbf k)=\mathbf A^{-1}\f$ and \f$\mathbf k\f$.
   *
   * The acoustic tensor of a cubic \f$\mathbf C_0\f$ is
   * \f$A_{ik} = (c_{12}+c_{44})k_ik_k + c_{44}k^2\delta_{ik} + H k_i^2\delta_{ik}\f$
   * with \f$H = c_{11}-c_{12}-2c_{44}\f$ (zero for isotropy, which recovers the
   * textbook \f$(\lambda+\mu)k_ik_k+\mu k^2\delta_{ik}\f$). \f$\Gamma\f$ is
   * homogeneous of degree zero in \f$\mathbf k\f$, so its *magnitude*
   * convention is irrelevant — but its *direction* is not, and that is where
   * the Nyquist mode bites. `k_component` maps index \f$N/2\f$ to
   * \f$+k_{\text{Nyq}}\f$ on every axis, so two modes that are conjugate
   * partners (say \f$(0,N/2,+q)\f$ and \f$(0,N/2,-q)\f$) are handed
   * *different* \f$\Gamma\f$ directions, because \f$\Gamma\f$ is even
   * under \f$\mathbf k\to-\mathbf k\f$ but not under flipping one
   * component. The result is a \f$\hat\varepsilon\f$ that is not Hermitian,
   * the inverse transform silently projects the anti-Hermitian part away, and
   * the strain that comes back no longer satisfies equilibrium — measurably:
   * \f$|\nabla\cdot\sigma|\f$ sat at \f$3\times10^{-4}\f$ of its
   * cancelling terms until this was fixed, and the Hellmann-Feynman identity
   * behind eq. (7) degraded with it.
   *
   * The fix is `k_component_odd`'s rule — zero the Nyquist component — applied
   * per axis, which restores evenness under the partner map. Modes whose every
   * index is 0 or Nyquist would then have \f$\mathbf k=\mathbf 0\f$ and a
   * singular acoustic tensor; those modes are their own conjugate partner, so
   * their coefficient is real and the raw direction is Hermitian-safe. They
   * keep it.
   */
  void build_green_operator(const pfc::Domain &domain, pfc::fft::IHostFFT &fft) {
    const std::size_t n = m_n_outbox;
    const auto gsz = pfc::domain::get_size(domain);
    m_kx.assign(n, 0.0);
    m_ky.assign(n, 0.0);
    m_kz.assign(n, 0.0);
    for (auto &g : m_g) g.assign(n, 0.0);
    m_zero_mode = static_cast<std::size_t>(-1);

    const double c12 = m_c0.c12;
    const double c44 = m_c0.c44;
    const double aniso = m_c0.c11 - m_c0.c12 - 2.0 * m_c0.c44;

    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t idx, double kx_raw, double ky_raw, double kz_raw, int i,
            int j, int k) {
          if (i == 0 && j == 0 && k == 0) {
            m_zero_mode = idx;
            return;
          }
          // Nyquist handling (see the note above this function).
          double kx = pfc::fft::kspace::is_nyquist_index(i, gsz[0]) ? 0.0 : kx_raw;
          double ky = pfc::fft::kspace::is_nyquist_index(j, gsz[1]) ? 0.0 : ky_raw;
          double kz = pfc::fft::kspace::is_nyquist_index(k, gsz[2]) ? 0.0 : kz_raw;
          if (kx == 0.0 && ky == 0.0 && kz == 0.0) {
            // Every index is 0 or Nyquist: the mode is its own conjugate
            // partner, its coefficient is real, and any real multiplier keeps
            // it real. The raw direction is therefore Hermitian-safe here and
            // is the only one that is not degenerate.
            kx = kx_raw;
            ky = ky_raw;
            kz = kz_raw;
          }
          m_kx[idx] = kx;
          m_ky[idx] = ky;
          m_kz[idx] = kz;
          const double k2 = kx * kx + ky * ky + kz * kz;
          const double axx = (c12 + c44) * kx * kx + c44 * k2 + aniso * kx * kx;
          const double ayy = (c12 + c44) * ky * ky + c44 * k2 + aniso * ky * ky;
          const double azz = (c12 + c44) * kz * kz + c44 * k2 + aniso * kz * kz;
          const double ayz = (c12 + c44) * ky * kz;
          const double axz = (c12 + c44) * kx * kz;
          const double axy = (c12 + c44) * kx * ky;

          // Adjugate of the symmetric 3x3 acoustic tensor.
          const double cof_xx = ayy * azz - ayz * ayz;
          const double cof_xy = ayz * axz - axy * azz;
          const double cof_xz = axy * ayz - ayy * axz;
          const double cof_yy = axx * azz - axz * axz;
          const double cof_yz = axy * axz - axx * ayz;
          const double cof_zz = axx * ayy - axy * axy;
          const double det = axx * cof_xx + axy * cof_xy + axz * cof_xz;
          if (!(det > 0.0)) {
            throw std::runtime_error(
                "EigenstrainMicroelasticity: non-positive acoustic determinant -- "
                "the reference stiffness is not positive definite");
          }
          const double inv = 1.0 / det;
          m_g[SYM_XX][idx] = cof_xx * inv;
          m_g[SYM_YY][idx] = cof_yy * inv;
          m_g[SYM_ZZ][idx] = cof_zz * inv;
          m_g[SYM_YZ][idx] = cof_yz * inv;
          m_g[SYM_XZ][idx] = cof_xz * inv;
          m_g[SYM_XY][idx] = cof_xy * inv;
        });
  }

  /// \f$\tau = \mathbf C:(\varepsilon-\varepsilon^{*}) - \mathbf C_0:\varepsilon\f$.
  void build_polarisation(const RealField &h, const RealField &amp) {
    const double *hp = h.data();
    const double *ap = amp.data();
    const Sym3 &pattern = m_params.eigenstrain_pattern;
    std::array<double *, kSymComponents> tau{};
    std::array<const double *, kSymComponents> eps{};
    for (int c = 0; c < kSymComponents; ++c) {
      tau[static_cast<std::size_t>(c)] = m_tau[static_cast<std::size_t>(c)].data();
      eps[static_cast<std::size_t>(c)] =
          m_strain[static_cast<std::size_t>(c)].data();
    }
    for (std::size_t i = 0; i < m_n_local; ++i) {
      Sym3 e;
      Sym3 ediff;
      for (int c = 0; c < kSymComponents; ++c) {
        e[c] = eps[static_cast<std::size_t>(c)][i];
        ediff[c] = e[c] - ap[i] * pattern[c];
      }
      const Sym3 s = stiffness_at(hp[i]).contract(ediff);
      const Sym3 s0 = m_c0.contract(e);
      for (int c = 0; c < kSymComponents; ++c) {
        tau[static_cast<std::size_t>(c)][i] = s[c] - s0[c];
      }
    }
    for (auto &f : m_tau) f.note_host_write();
  }

  void swap_tau() noexcept {
    for (int c = 0; c < kSymComponents; ++c) {
      m_tau_prev[static_cast<std::size_t>(c)].vec().swap(
          m_tau[static_cast<std::size_t>(c)].vec());
    }
  }

  /// \f$\max|\tau-\tau_{\text{prev}}| / \max(|\tau|,\text{floor})\f$, global max.
  [[nodiscard]] double relative_change() const {
    double diff = 0.0;
    double scale = 0.0;
    for (int c = 0; c < kSymComponents; ++c) {
      const double *a = m_tau[static_cast<std::size_t>(c)].data();
      const double *b = m_tau_prev[static_cast<std::size_t>(c)].data();
      for (std::size_t i = 0; i < m_n_local; ++i) {
        diff = std::max(diff, std::abs(a[i] - b[i]));
        scale = std::max(scale, std::abs(a[i]));
      }
    }
    double local[2] = {diff, scale};
    double global[2] = {0.0, 0.0};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, m_params.comm);
    if (global[1] <= 0.0) return 0.0;
    return global[0] / global[1];
  }

  /// One \f$\Gamma\f$ application: 6 forward transforms, the Green multiply, 6 back.
  void apply_green_operator() {
    for (int c = 0; c < kSymComponents; ++c) {
      m_fft.forward(m_tau_prev[static_cast<std::size_t>(c)].vec(),
                    m_hat[static_cast<std::size_t>(c)].vec());
    }

    std::array<Complex *, kSymComponents> t{};
    for (int c = 0; c < kSymComponents; ++c) {
      t[static_cast<std::size_t>(c)] = m_hat[static_cast<std::size_t>(c)].data();
    }

    for (std::size_t i = 0; i < m_n_outbox; ++i) {
      if (i == m_zero_mode) continue;
      const double kx = m_kx[i];
      const double ky = m_ky[i];
      const double kz = m_kz[i];
      const Complex txx = t[SYM_XX][i];
      const Complex tyy = t[SYM_YY][i];
      const Complex tzz = t[SYM_ZZ][i];
      const Complex tyz = t[SYM_YZ][i];
      const Complex txz = t[SYM_XZ][i];
      const Complex txy = t[SYM_XY][i];

      // b_i = k_j tau_ij
      const Complex bx = kx * txx + ky * txy + kz * txz;
      const Complex by = kx * txy + ky * tyy + kz * tyz;
      const Complex bz = kx * txz + ky * tyz + kz * tzz;

      // v = G b  (G symmetric)
      const Complex vx =
          m_g[SYM_XX][i] * bx + m_g[SYM_XY][i] * by + m_g[SYM_XZ][i] * bz;
      const Complex vy =
          m_g[SYM_XY][i] * bx + m_g[SYM_YY][i] * by + m_g[SYM_YZ][i] * bz;
      const Complex vz =
          m_g[SYM_XZ][i] * bx + m_g[SYM_YZ][i] * by + m_g[SYM_ZZ][i] * bz;

      // eps_hat = -sym(k (x) v)   ==   -Gamma : tau_hat
      t[SYM_XX][i] = -(kx * vx);
      t[SYM_YY][i] = -(ky * vy);
      t[SYM_ZZ][i] = -(kz * vz);
      t[SYM_YZ][i] = -0.5 * (ky * vz + kz * vy);
      t[SYM_XZ][i] = -0.5 * (kx * vz + kz * vx);
      t[SYM_XY][i] = -0.5 * (kx * vy + ky * vx);
    }

    if (m_zero_mode != static_cast<std::size_t>(-1)) {
      // HeFFTe scales on the backward transform only, so the k = 0 coefficient
      // that produces a mean of E_applied is N_global * E_applied.
      const auto gs = m_strain[0].global_size();
      const double n_global = static_cast<double>(gs[0]) *
                              static_cast<double>(gs[1]) *
                              static_cast<double>(gs[2]);
      for (int c = 0; c < kSymComponents; ++c) {
        t[static_cast<std::size_t>(c)][m_zero_mode] =
            Complex{n_global * m_params.applied_strain[c], 0.0};
      }
    }

    for (int c = 0; c < kSymComponents; ++c) {
      m_hat[static_cast<std::size_t>(c)].note_host_write();
      m_fft.backward(m_hat[static_cast<std::size_t>(c)].vec(),
                     m_strain[static_cast<std::size_t>(c)].vec());
      m_strain[static_cast<std::size_t>(c)].note_host_write();
    }
  }

  /// Stress, \f$f_{\mathrm{el}}\f$ and (optionally) eq. (7).
  void finalise(const RealField &h, const RealField &amp, const RealField *dh_dphi,
                const RealField *damp_dphi) {
    const double *hp = h.data();
    const double *ap = amp.data();
    const double *dhp = (dh_dphi != nullptr) ? dh_dphi->data() : nullptr;
    const double *dap = (damp_dphi != nullptr) ? damp_dphi->data() : nullptr;
    const bool want_dfel = (dhp != nullptr) && (dap != nullptr);
    const Sym3 &pattern = m_params.eigenstrain_pattern;
    const Stiffness dc =
        Stiffness::blend(m_params.c_solid, 1.0, m_params.c_liquid, -1.0);

    std::array<const double *, kSymComponents> eps{};
    std::array<double *, kSymComponents> sig{};
    for (int c = 0; c < kSymComponents; ++c) {
      eps[static_cast<std::size_t>(c)] =
          m_strain[static_cast<std::size_t>(c)].data();
      sig[static_cast<std::size_t>(c)] =
          m_stress[static_cast<std::size_t>(c)].data();
    }
    double *fel = m_f_el.data();
    double *dfel = m_dfel_dphi.data();

    for (std::size_t i = 0; i < m_n_local; ++i) {
      Sym3 ediff;
      for (int c = 0; c < kSymComponents; ++c) {
        ediff[c] = eps[static_cast<std::size_t>(c)][i] - ap[i] * pattern[c];
      }
      const Sym3 s = stiffness_at(hp[i]).contract(ediff);
      for (int c = 0; c < kSymComponents; ++c)
        sig[static_cast<std::size_t>(c)][i] = s[c];
      fel[i] = 0.5 * ddot(ediff, s);
      if (want_dfel) {
        // Term 1: transformation work, -sigma : d eps*/dphi.
        double work = 0.0;
        for (int c = 0; c < kSymComponents; ++c) {
          const double w = (c < 3) ? 1.0 : 2.0;
          work += w * s[c] * dap[i] * pattern[c];
        }
        // Term 2: modulus contrast, (1/2)(eps-eps*) : dC/dphi : (eps-eps*).
        const Sym3 sc = dc.contract(ediff);
        dfel[i] = -work + 0.5 * dhp[i] * ddot(ediff, sc);
      } else {
        dfel[i] = 0.0;
      }
    }
    for (auto &f : m_stress) f.note_host_write();
    m_f_el.note_host_write();
    m_dfel_dphi.note_host_write();
  }

  pfc::fft::IHostFFT &m_fft;
  MicroelasticityParams m_params;
  Stiffness m_c0;

  SymRealFields m_strain;
  SymRealFields m_stress;
  SymRealFields m_tau;
  SymRealFields m_tau_prev;
  RealField m_f_el;
  RealField m_dfel_dphi;
  std::array<ComplexField, kSymComponents> m_hat{};

  std::vector<double> m_kx, m_ky, m_kz;
  std::array<std::vector<double>, kSymComponents> m_g{};
  std::size_t m_zero_mode{static_cast<std::size_t>(-1)};
  std::size_t m_n_local{0};
  std::size_t m_n_outbox{0};
  bool m_has_solution{false};
};

} // namespace pfc::apps
