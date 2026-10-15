// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file microelasticity.hpp
 * @brief Quasi-static 3-D eigenstrain microelasticity on the existing FFT stack.
 *
 * Periodic representative-volume FFT microelasticity. Not a finite-body,
 * traction, or finite-element solver. The two stiffness fields and the
 * eigenstrain are inputs; mapping them from phase, composition, or
 * temperature belongs to the application.
 *
 * @details
 * The unknown is the periodic displacement that balances a stiffness field
 * and an eigenstrain the caller supplies. Equilibrium is elliptic and is
 * re-solved whenever those inputs change; the displacement is not part of
 * the caller's time integrator.
 *
 * \f[
 *   \nabla\cdot\boldsymbol\sigma = 0,\qquad
 *   \boldsymbol\sigma = \mathbf{C}(w):\bigl(\boldsymbol\varepsilon(\mathbf u)
 *                                              - \boldsymbol\varepsilon^{*}\bigr),
 *   \qquad
 *   \varepsilon_{ij}(\mathbf u) = \tfrac12\,(\partial_i u_j + \partial_j u_i),
 * \f]
 *
 * with \f$\mathbf C(w) = w\,\mathbf C(1) + (1-w)\,\mathbf C(0)\f$.
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
 * modulus is inhomogeneous. The two endpoints differ, and that contrast is
 * exactly what makes this *not* a one-shot solve. The
 * fix is a fixed point — evaluate \f$\boldsymbol\tau\f$ in real space where
 * the modulus lives, apply \f$\Gamma\f$ in Fourier space where the Green
 * operator lives, repeat. Two of them are implemented
 * (`MicroelasticityScheme`), they cost exactly the same per iteration, and
 * they converge to the same answer; the difference is how fast.
 *
 * ## How fast, and why the plain scheme is not enough
 *
 * The Hu & Chen (2001) scheme is a Neumann series, contracting at
 * \f$\lVert(\mathbf C-\mathbf C_0)\mathbf C_0^{-1}\rVert\f$, minimised by
 * taking \f$\mathbf C_0\f$ to be the Voigt (arithmetic) mean and equal to
 * \f$(r-1)/(r+1)\f$ for a stiffness ratio \f$r\f$. That tends to 1 as
 * \f$r\to\infty\f$ *like* \f$1 - 2/r\f$. A soft endpoint that carries little
 * shear sits at \f$r = 10\ldots100\f$. At one solve per time step, several
 * hundred Green applications per step is not a cost anyone can pay.
 *
 * The Eyre–Milton scheme (Eyre & Milton, *Eur. Phys. J. AP* **6**, 41 (1999))
 * fixes this. It rewrites *both* the constitutive law and the
 * equilibrium/compatibility conditions as reflections and alternates them
 * (see `run_eyre_milton` for the derivation); the denominator of the local
 * gain becomes \f$\lambda+\lambda_0\f$ instead of \f$\lambda_0\f$, the
 * optimal reference becomes the *geometric* mean, and the contraction becomes
 * \f$(\sqrt r-1)/(\sqrt r+1)\f$. The square root is the whole point. It is
 * the default here.
 *
 * Measured on a \f$32^3\f$ grid with a tanh sphere (R = 7, w = 1.5),
 * isotropic \f$\nu = 0.3\f$, uniform stiffness scaling, cold start,
 * `relative_tolerance = 1e-6`:
 *
 * | \f$r = C(1)/C(0)\f$ | 1 | 2 | 4 | 10 | 100 |
 * |---|---|---|---|---|---|
 * | `Basic` iterations | 1 | 11 | 22 | 53 | 466 |
 * | `Basic` observed contraction | — | 0.265 | 0.524 | 0.767 | 0.970 |
 * | \f$(r-1)/(r+1)\f$ | 0 | 0.333 | 0.600 | 0.818 | 0.980 |
 * | **`EyreMilton` iterations** | **1** | **7** | **12** | **20** | **66** |
 * | `EyreMilton` observed contraction | — | 0.118 | 0.262 | 0.453 | 0.793 |
 * | \f$(\sqrt r-1)/(\sqrt r+1)\f$ | 0 | 0.172 | 0.333 | 0.519 | 0.818 |
 *
 * Both predictions bound the measurement from above and track it closely
 * (the observed rate is a little better because the worst channel does not
 * dominate every mode), which is what makes the square root a fact about this
 * code and not a citation. `predicted_contraction()` returns the analytic
 * number for the configured scheme; `test_microelasticity.cpp` regenerates
 * every entry of this table and asserts against it.
 *
 * A shear-only contrast of 20 (bulk modulus unchanged) needs **16**
 * accelerated iterations against the basic scheme's **44**. `Basic` remains
 * selectable: it is the reference the accelerated scheme is validated
 * against, and the two agree to 7e-13 in the strain.
 *
 * Other ways to buy back cost, all still available: lag the solve over
 * several caller steps (the problem is quasi-static, so this is legitimate),
 * or warm-start from the previous step, which is the
 * default and cuts the count sharply once the interface is only moving a cell
 * per step. `MicroelasticityReport::converged` is returned rather than
 * thrown, so the driver decides.
 *
 * ## What the iteration count means here
 *
 * `iterations` counts \f$\Gamma\f$ applications, so the number is directly
 * comparable between the two schemes and is what the table above reports.
 *
 * The spec's stopping test is the relative change of the strain between
 * successive passes. Both schemes here test the relative change of the
 * *polarisation* \f$\boldsymbol\tau = \boldsymbol\sigma -
 * \mathbf C_0:\boldsymbol\varepsilon\f$ instead — the same quantity in both,
 * which is why their iteration counts mean the same thing. The two norms are
 * the same fixed point one step apart
 * (\f$\varepsilon_{n+1}-\varepsilon_n = -\Gamma:(\tau_n - \tau_{n-1})\f$ with
 * \f$\Gamma\f$ linear and bounded), and the suite checks directly that the
 * strain change after the reported count is below `relative_tolerance`. For `Basic`
 * the test is applied in real space *before* the transforms, so the pass that merely
 * confirms convergence is free.
 *
 * A homogeneous modulus costs exactly one \f$\Gamma\f$ application in both
 * schemes, for different reasons: in `Basic` the polarisation collapses to
 * \f$-\mathbf C_0:\boldsymbol\varepsilon^{*}\f$, independent of
 * \f$\boldsymbol\varepsilon\f$, so the second pass reproduces it and exits
 * before touching the FFT; in `EyreMilton` the local reflection
 * \f$\mathbf I - 2\mathbf C_0(\mathbf C+\mathbf C_0)^{-1}\f$ is identically
 * zero, so the state is at its fixed point as soon as it is first built.
 * `MicroelasticityReport::residual_history` records every measured residual
 * so a caller (or a test) can check the decay. It is strictly monotone up to
 * \f$r=10\f$; at \f$r=100\f$ the accelerated scheme shows a few
 * single-iteration excursions (the quantity that contracts every step is the
 * error in the \f$\mathbf C_0\f$ energy norm, not the max-norm of the
 * polarisation increment), while still falling over every five-iteration
 * window.
 *
 * ## Eigenstrain model
 *
 * \f$\boldsymbol\varepsilon^{*}(\mathbf x) = a(\mathbf x)\,\mathbf P\f$: one
 * scalar amplitude field times one constant symmetric pattern tensor
 * (`MicroelasticityParams::eigenstrain_pattern`, identity by default, i.e.
 * dilatational). One amplitude field covers a dilatational or
 * single-variant transformation strain and keeps the per-call state at two
 * scalar fields instead of twelve. A multi-variant eigenstrain
 * \f$\sum_v a_v \mathbf P_v\f$ is *not* supported.
 *
 * ## Energy and its derivative
 *
 * \f[
 *   f_{\mathrm{el}} = \tfrac12(\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}):
 *                      \mathbf
 * C:(\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}),
 *   \qquad
 *   \frac{\partial f_{\mathrm{el}}}{\partial s} =
 *     -\,\boldsymbol\sigma:\frac{\partial\boldsymbol\varepsilon^{*}}{\partial s}
 *     + \tfrac12(\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}):
 *        \frac{\partial\mathbf C}{\partial s}:
 *        (\boldsymbol\varepsilon-\boldsymbol\varepsilon^{*}).
 * \f]
 *
 * \f$s\f$ is whatever scalar the caller differentiated. The solver does not
 * name it: `solve()` takes \f$\partial w/\partial s\f$ and
 * \f$\partial a/\partial s\f$ and returns both terms. The first is the
 * transformation-work term and dominates; the second is the modulus-contrast
 * term. Dropping it makes the derivative wrong wherever the stiffness varies.
 * This partial derivative at frozen \f$\boldsymbol\varepsilon\f$ is also the
 * total variational derivative of the elastic energy, because the strain is
 * at equilibrium and \f$\delta F/\delta\mathbf u = -\nabla\cdot
 * \boldsymbol\sigma = 0\f$. `test_microelasticity.cpp` checks that against a
 * finite difference of the fully re-converged energy rather than assuming it.
 *
 * Because the displacement is periodic and \f$\langle\boldsymbol\varepsilon
 * \rangle\f$ is the applied strain (zero-mean stress sets it so that
 * \f$\langle\boldsymbol\sigma\rangle\approx 0\f$), the virtual work identity
 * \f$\int\boldsymbol\sigma:\boldsymbol\varepsilon\,\mathrm{d}V=0\f$ reduces
 * the stored energy to the transformation work already in eq. (6):
 *
 * \f[
 *   F=\int f_{\mathrm{el}}\,\mathrm{d}V
 *     =-\tfrac12\int\boldsymbol\sigma:\boldsymbol\varepsilon^{*}\,\mathrm{d}V
 *     \equiv W_{*}.
 * \f]
 *
 * That is the energy balance a caller can report. The
 * residual is \f$|F-W_{*}|/\max(|F|,|W_{*}|,\varepsilon)\f$ with floor
 * \f$\varepsilon=10^{-30}\f$ so a zero-energy run (no eigenstrain) is
 * identically balanced rather than NaN. It is not an ad-hoc score: it is
 * the same integral oracle `test_microelasticity.cpp` already uses.
 *
 * ## Scope — what this deliberately does not do
 *
 * - **Host reference, optional device twin.** Every loop here is a plain
 *   host loop over `pfc::data::Field<double>` and the transforms go through
 *   `pfc::fft::IHostFFT`. The device path is `microelasticity_device.hpp`
 *   (issues #157, #552): same Eyre–Milton scheme on `IDeviceFFT` plus one
 *   single-source kernel file compiled for CUDA or HIP, not a
 *   `SpectralETDOps` template parameter.
 * - **Periodic only.** The Green operator *is* the periodic boundary
 *   condition. A free surface or a clamped face needs a different solver.
 * - **Small strain, no plasticity, no finite rotation.** Linear kinematics
 *   throughout.
 * - **No displacement output.** Only the strain, stress, energy density and
 *   \f$\partial f_{\mathrm{el}}/\partial s\f$ are stored; \f$\mathbf u\f$ is
 *   an intermediate.
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

namespace pfc::solvers {

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

/// Floor in the relative energy-balance residual so \f$F=W_{*}=0\f$ is 0,
/// not NaN.
inline constexpr double kElasticEnergyBalanceFloor = 1.0e-30;

/**
 * @brief Snapshot energy balance \f$F=W_{*}\f$ from eq. (6) and periodicity.
 *
 * \f$F=\int f_{\mathrm{el}}\,\mathrm{d}V\f$,
 * \f$W_{*}=-\tfrac12\int\boldsymbol\sigma:\boldsymbol\varepsilon^{*}\,\mathrm{d}V\f$,
 * \f$r=|F-W_{*}|/\max(|F|,|W_{*}|,\varepsilon)\f$.
 */
struct ElasticEnergyBalance {
  double F{0.0};
  double W_star{0.0};
  double residual{0.0};
  double residual_rel{0.0};
};

[[nodiscard]] inline ElasticEnergyBalance
elastic_energy_balance_from_integrals(double F, double W_star) noexcept {
  ElasticEnergyBalance b;
  b.F = F;
  b.W_star = W_star;
  b.residual = std::abs(F - W_star);
  const double scale = std::max(std::abs(F), std::abs(W_star));
  b.residual_rel = b.residual / std::max(scale, kElasticEnergyBalanceFloor);
  return b;
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

  /**
   * @brief Cubic stiffness from its three eigenvalue channels.
   *
   * @details
   * A cubic \f$\mathbf C\f$ with axes along the grid is diagonal in three
   * mutually orthogonal subspaces of symmetric tensors, and this is the form
   * in which every question about *contrast* has a clean answer:
   *
   * | channel | subspace | eigenvalue |
   * |---|---|---|
   * | hydrostatic | \f$\delta_{ij}\f$ (dim 1) | \f$3K = c_{11}+2c_{12}\f$ |
   * | tetragonal shear | traceless diagonal (dim 2) | \f$2\mu' = c_{11}-c_{12}\f$ |
   * | trigonal shear | off-diagonal (dim 3) | \f$2\mu'' = 2c_{44}\f$ |
   *
   * Two cubic tensors sharing axes therefore commute, so their geometric mean
   * (which the Eyre–Milton reference needs) is just the channelwise geometric
   * mean, and the contraction factor of either fixed point is the worst
   * channel. `bulk_modulus()`, `shear_tetragonal()` and `shear_trigonal()`
   * read the channels back out.
   */
  [[nodiscard]] static Stiffness from_channels(double bulk, double mu_tetragonal,
                                               double mu_trigonal) noexcept {
    return Stiffness{bulk + 4.0 * mu_tetragonal / 3.0,
                     bulk - 2.0 * mu_tetragonal / 3.0, mu_trigonal};
  }

  /// Zener anisotropy ratio \f$2c_{44}/(c_{11}-c_{12})\f$; 1 means isotropic.
  [[nodiscard]] double zener() const noexcept { return 2.0 * c44 / (c11 - c12); }

  /// Bulk modulus \f$(c_{11}+2c_{12})/3\f$ (exact for cubic: dilatation is
  /// isotropic).
  [[nodiscard]] double bulk_modulus() const noexcept {
    return (c11 + 2.0 * c12) / 3.0;
  }

  /// \f$\mu' = (c_{11}-c_{12})/2\f$, the tetragonal-shear channel.
  [[nodiscard]] double shear_tetragonal() const noexcept {
    return 0.5 * (c11 - c12);
  }

  /// \f$\mu'' = c_{44}\f$, the trigonal-shear channel.
  [[nodiscard]] double shear_trigonal() const noexcept { return c44; }

  /// The three eigenvalues, in the order of `from_channels`'s table.
  [[nodiscard]] std::array<double, 3> eigenvalues() const noexcept {
    return {c11 + 2.0 * c12, c11 - c12, 2.0 * c44};
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

  /**
   * @brief \f$\boldsymbol\varepsilon\f$ from \f$\boldsymbol\sigma\f$, i.e. the
   *        compliance \f$\mathbf C^{-1}:\boldsymbol\sigma\f$.
   *
   * Closed form, from the channel decomposition above: the normal block is
   * \f$(c_{11}-c_{12})\mathbf I + c_{12}\mathbf J\f$ whose inverse is
   * \f$[\mathbf I - c_{12}\mathbf J/(c_{11}+2c_{12})]/(c_{11}-c_{12})\f$, and
   * the shears invert one at a time. Needed by the Eyre–Milton local step,
   * which solves \f$(\mathbf C+\mathbf C_0):\boldsymbol\varepsilon = \ldots\f$
   * once per cell.
   *
   * @throws std::invalid_argument if any channel eigenvalue vanishes.
   */
  [[nodiscard]] Sym3 solve(const Sym3 &s) const {
    const double d = c11 - c12;
    const double t = c11 + 2.0 * c12;
    if (d == 0.0 || t == 0.0 || c44 == 0.0) {
      throw std::invalid_argument(
          "Stiffness::solve: singular stiffness (a channel eigenvalue is zero)");
    }
    const double tr = s[SYM_XX] + s[SYM_YY] + s[SYM_ZZ];
    const double shift = c12 * tr / t;
    Sym3 e;
    e[SYM_XX] = (s[SYM_XX] - shift) / d;
    e[SYM_YY] = (s[SYM_YY] - shift) / d;
    e[SYM_ZZ] = (s[SYM_ZZ] - shift) / d;
    e[SYM_YZ] = s[SYM_YZ] / (2.0 * c44);
    e[SYM_XZ] = s[SYM_XZ] / (2.0 * c44);
    e[SYM_XY] = s[SYM_XY] / (2.0 * c44);
    return e;
  }

  /// Componentwise \f$\alpha A + \beta B\f$; the Voigt average is `blend(a, .5, b,
  /// .5)`.
  [[nodiscard]] static Stiffness blend(const Stiffness &a, double wa,
                                       const Stiffness &b, double wb) noexcept {
    return Stiffness{wa * a.c11 + wb * b.c11, wa * a.c12 + wb * b.c12,
                     wa * a.c44 + wb * b.c44};
  }

  /**
   * @brief The tensor geometric mean \f$(\mathbf A\mathbf B)^{1/2}\f$.
   *
   * Exact, not a heuristic: aligned cubic tensors commute, so the geometric
   * mean is the channelwise geometric mean of the eigenvalues. This is the
   * reference medium the Eyre–Milton scheme wants, and it is what turns the
   * contraction factor from \f$(r-1)/(r+1)\f$ into
   * \f$(\sqrt r-1)/(\sqrt r+1)\f$. A non-positive channel (which a physical
   * stiffness cannot have) falls back to the arithmetic mean for that channel.
   */
  [[nodiscard]] static Stiffness geometric_mean(const Stiffness &a,
                                                const Stiffness &b) noexcept {
    const auto ea = a.eigenvalues();
    const auto eb = b.eigenvalues();
    std::array<double, 3> g{};
    for (int i = 0; i < 3; ++i) {
      g[static_cast<std::size_t>(i)] =
          (ea[static_cast<std::size_t>(i)] > 0.0 &&
           eb[static_cast<std::size_t>(i)] > 0.0)
              ? std::sqrt(ea[static_cast<std::size_t>(i)] *
                          eb[static_cast<std::size_t>(i)])
              : 0.5 * (ea[static_cast<std::size_t>(i)] +
                       eb[static_cast<std::size_t>(i)]);
    }
    return from_channels(g[0] / 3.0, 0.5 * g[1], 0.5 * g[2]);
  }
};

/**
 * @brief Which fixed point to run.
 *
 * @details
 * Both converge to the *same* solution — that is asserted in
 * `test_microelasticity.cpp`, not assumed — and both cost one Green-operator
 * application (12 transforms) plus one local pass per iteration. They differ
 * only in how fast the error decays with the stiffness contrast \f$r\f$:
 *
 * | | reference \f$\mathbf C_0\f$ | contraction | its. at \f$r=100\f$ |
 * |---|---|---|---|
 * | `Basic` | arithmetic (Voigt) mean | \f$(r-1)/(r+1)\f$ | 466 |
 * | `EyreMilton` | geometric mean | \f$(\sqrt r-1)/(\sqrt r+1)\f$ | 66 |
 *
 * The square root is the whole point. A soft endpoint with little shear
 * sits at a contrast of 10–100, and the basic scheme is simply not
 * affordable there at one elastic solve
 * per solve. `EyreMilton` was measured faster at every contrast
 * tried, including \f$r=2\f$, so `Basic` is kept for one reason only: it is
 * the reference the accelerated scheme is validated against, and the suite
 * asserts the two agree to 7e-13 in the strain rather than assuming it.
 *
 * (Moulinec & Suquet's augmented-Lagrangian scheme reaches the same
 * \f$\sqrt r\f$ rate and would have been an acceptable alternative.
 * Eyre–Milton was chosen because its global half is literally the Green
 * application this header already had — \f$\mathbf y = \mathbf z +
 * 2\mathbf C_0:W(\mathbf z)\f$ — so it adds one local 6×6 solve and no new
 * FFT machinery, no extra field of state, and no penalty parameter to tune.)
 */
enum class MicroelasticityScheme : int {
  /// Neumann series on \f$\boldsymbol\tau\f$ (Hu & Chen 2001; Moulinec &
  /// Suquet 1994).
  Basic = 0,
  /// Alternating reflections on \f$\boldsymbol\sigma\pm\mathbf C_0
  /// :\boldsymbol\varepsilon\f$ (Eyre & Milton, *Eur. Phys. J. AP* **6**, 41
  /// (1999)).
  EyreMilton = 1
};

/// Configuration of the fixed point and the reference medium.
struct MicroelasticityParams {
  Stiffness stiffness_at_one{};
  Stiffness stiffness_at_zero{};
  /// Constant pattern \f$\mathbf P\f$ in \f$\varepsilon^{*} = a(\mathbf x)\mathbf
  /// P\f$.
  Sym3 eigenstrain_pattern{Sym3::identity()};
  /// \f$\hat\varepsilon(\mathbf 0)\f$ — zero is a free (unloaded) periodic body.
  Sym3 applied_strain{};
  /// Which fixed point to run; see `MicroelasticityScheme`.
  MicroelasticityScheme scheme{MicroelasticityScheme::EyreMilton};
  /// Relative polarisation change at which the fixed point is declared converged.
  double relative_tolerance{1.0e-6};
  /**
   * @brief Hard cap on \f$\Gamma\f$ applications.
   *
   * 50, not 20. At a shear contrast of 20 the accelerated scheme needs 16
   * iterations (44 for `Basic`); a shear fraction of 0.01 needs 32. 50
   * covers that range. The number is the one the tests pin, not a quiet
   * retune.
   */
  int max_iterations{50};
  /**
   * @brief Reference medium. Left at its default (all zeros) the
   *        contraction-optimal choice for the selected scheme is used: the
   *        arithmetic (Voigt) mean of the two endpoints for `Basic`, the
   *        geometric mean for `EyreMilton`.
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
 * local outbox) and call `solve()` whenever the inputs change.
 */
class EigenstrainMicroelasticity {
public:
  using FFT = pfc::fft::IHostFFT;
  using RealField = pfc::data::Field<double>;
  using Complex = std::complex<double>;
  using ComplexField = pfc::data::Field<Complex>;
  using SymRealFields = std::array<RealField, kSymComponents>;

  EigenstrainMicroelasticity(const pfc::Domain &domain, pfc::fft::IHostFFT &fft,
                             MicroelasticityParams params)
      : m_fft(fft), m_params(params),
        m_reference_stiffness(is_zero(params.reference)
                                  ? optimal_reference(params.scheme,
                                                      params.stiffness_at_one,
                                                      params.stiffness_at_zero)
                                  : params.reference),
        m_strain(make_sym(domain, fft)), m_stress(make_sym(domain, fft)),
        m_tau(make_sym(domain, fft)), m_tau_prev(make_sym(domain, fft)),
        m_elastic_energy_density(
            pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_elastic_energy_derivative(
            pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())) {
    if (m_params.relative_tolerance < 0.0) {
      throw std::invalid_argument(
          "EigenstrainMicroelasticity: relative_tolerance must be >= 0");
    }
    if (m_params.max_iterations < 1) {
      throw std::invalid_argument(
          "EigenstrainMicroelasticity: max_iterations must be >= 1");
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

  /// Reference medium actually in use (scheme-optimal unless overridden).
  [[nodiscard]] const Stiffness &reference() const noexcept {
    return m_reference_stiffness;
  }

  /**
   * @brief The contraction-optimal reference for a scheme and a phase pair.
   *
   * Arithmetic mean for `Basic` (minimises
   * \f$\max\lVert(\mathbf C-\mathbf C_0)\mathbf C_0^{-1}\rVert\f$), geometric
   * mean for `EyreMilton` (minimises
   * \f$\max\lVert(\mathbf C-\mathbf C_0)(\mathbf C+\mathbf C_0)^{-1}\rVert\f$).
   */
  [[nodiscard]] static Stiffness
  optimal_reference(MicroelasticityScheme scheme, const Stiffness &stiffness_at_one,
                    const Stiffness &stiffness_at_zero) noexcept {
    return (scheme == MicroelasticityScheme::EyreMilton)
               ? Stiffness::geometric_mean(stiffness_at_one, stiffness_at_zero)
               : Stiffness::blend(stiffness_at_one, 0.5, stiffness_at_zero, 0.5);
  }

  /**
   * @brief Asymptotic contraction factor predicted for the configured scheme.
   *
   * @details
   * The local stiffness sweeps \f$w\mathbf C(1) + (1-w)\mathbf C(0)\f$ as
   * \f$w\f$ runs over \f$[0,1]\f$, and both schemes' error operators are
   * monotone in each channel eigenvalue, so the worst case sits at one of the
   * two endpoints. Per channel eigenvalue \f$\lambda\f$ against the
   * reference's \f$\lambda_0\f$ the factor is
   * \f$|\lambda-\lambda_0|/\lambda_0\f$ for `Basic` and
   * \f$|\lambda-\lambda_0|/(\lambda+\lambda_0)\f$ for `EyreMilton`; the answer
   * is the maximum over the three channels and the two endpoints. With the
   * optimal reference these reduce to \f$(r-1)/(r+1)\f$ and
   * \f$(\sqrt r-1)/(\sqrt r+1)\f$ for a uniform channel ratio \f$r\f$.
   *
   * A prediction, not a measurement: the observed rate is a little better
   * because the worst channel does not dominate every mode. The test suite
   * checks the measured rate against this number.
   */
  [[nodiscard]] double predicted_contraction() const noexcept {
    const auto e0 = m_reference_stiffness.eigenvalues();
    const auto es = m_params.stiffness_at_one.eigenvalues();
    const auto el = m_params.stiffness_at_zero.eigenvalues();
    double worst = 0.0;
    for (int i = 0; i < 3; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      for (const double lam : {es[idx], el[idx]}) {
        const double num = std::abs(lam - e0[idx]);
        const double den = (m_params.scheme == MicroelasticityScheme::EyreMilton)
                               ? (lam + e0[idx])
                               : e0[idx];
        if (den > 0.0) worst = std::max(worst, num / den);
      }
    }
    return worst;
  }
  [[nodiscard]] const MicroelasticityParams &params() const noexcept {
    return m_params;
  }

  /// Mutable parameters. A driver may retune `relative_tolerance` and
  /// `max_iterations` between steps.
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
    return m_elastic_energy_density;
  }
  /// \f$\partial f_{\mathrm{el}}/\partial s\f$; only filled when `solve()` got
  /// the two derivative fields.
  [[nodiscard]] const RealField &elastic_energy_derivative() const noexcept {
    return m_elastic_energy_derivative;
  }

  /**
   * @brief Solve equilibrium for the given modulus interpolation and eigenstrain.
   *
   * @param stiffness_weight  \f$w\in[0,1]\f$;
   *        \f$\mathbf C = w\mathbf C(1) + (1-w)\mathbf C(0)\f$
   * @param eigenstrain_amplitude \f$a(\mathbf x)\f$ in
   *        \f$\varepsilon^{*} = a\,\mathbf P\f$
   * @param stiffness_weight_derivative \f$\partial w\f$ with respect to the
   *        caller's order parameter, or `nullptr` to skip that term
   * @param eigenstrain_amplitude_derivative \f$\partial a\f$ with respect to
   *        the same parameter, or `nullptr` to skip that term
   *
   * On return `strain()`, `stress()` and `elastic_energy_density()` are filled;
   * `elastic_energy_derivative()` too when both derivative fields were supplied.
   */
  MicroelasticityReport
  solve(const RealField &stiffness_weight, const RealField &eigenstrain_amplitude,
        const RealField *stiffness_weight_derivative = nullptr,
        const RealField *eigenstrain_amplitude_derivative = nullptr) {
    check_same_size(stiffness_weight, "stiffness_weight");
    check_same_size(eigenstrain_amplitude, "eigenstrain_amplitude");
    if (stiffness_weight_derivative != nullptr) {
      check_same_size(*stiffness_weight_derivative, "stiffness_weight_derivative");
    }
    if (eigenstrain_amplitude_derivative != nullptr) {
      check_same_size(*eigenstrain_amplitude_derivative,
                      "eigenstrain_amplitude_derivative");
    }

    if (!m_params.warm_start || !m_has_solution) {
      for (int c = 0; c < kSymComponents; ++c) {
        std::fill(m_strain[static_cast<std::size_t>(c)].vec().begin(),
                  m_strain[static_cast<std::size_t>(c)].vec().end(),
                  m_params.applied_strain[c]);
      }
    }

    const MicroelasticityReport report =
        (m_params.scheme == MicroelasticityScheme::EyreMilton)
            ? run_eyre_milton(stiffness_weight, eigenstrain_amplitude)
            : run_basic(stiffness_weight, eigenstrain_amplitude);

    m_has_solution = true;
    finalise(stiffness_weight, eigenstrain_amplitude, stiffness_weight_derivative,
             eigenstrain_amplitude_derivative);
    return report;
  }

  /**
   * @brief \f$\int f_{\mathrm{el}}\,\mathrm dV\f$ over the whole domain.
   *
   * Reduced across `params().comm`, so every rank gets the same number.
   */
  [[nodiscard]] double total_elastic_energy() const {
    const auto &s = m_elastic_energy_density.spacing();
    const double cell = s[0] * s[1] * s[2];
    double local = 0.0;
    for (std::size_t i = 0; i < m_n_local; ++i)
      local += m_elastic_energy_density.data()[i];
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, m_params.comm);
    return global * cell;
  }

  /**
   * @brief \f$W_{*}=-\tfrac12\int\boldsymbol\sigma:\boldsymbol\varepsilon^{*}
   *        \,\mathrm{d}V\f$ from the last solve's stress and
   *        @p eigenstrain_amplitude.
   */
  [[nodiscard]] double
  transformation_work(const RealField &eigenstrain_amplitude) const {
    check_same_size(eigenstrain_amplitude, "eigenstrain_amplitude");
    const auto &s = m_elastic_energy_density.spacing();
    const double cell = s[0] * s[1] * s[2];
    const Sym3 &pattern = m_params.eigenstrain_pattern;
    double local = 0.0;
    for (std::size_t i = 0; i < m_n_local; ++i) {
      Sym3 estar;
      Sym3 sg;
      for (int c = 0; c < kSymComponents; ++c) {
        const auto ci = static_cast<std::size_t>(c);
        estar[c] = eigenstrain_amplitude.data()[i] * pattern[c];
        sg[c] = m_stress[ci].data()[i];
      }
      local += ddot(sg, estar);
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, m_params.comm);
    return -0.5 * global * cell;
  }

  [[nodiscard]] ElasticEnergyBalance
  energy_balance(const RealField &eigenstrain_amplitude) const {
    return elastic_energy_balance_from_integrals(
        total_elastic_energy(), transformation_work(eigenstrain_amplitude));
  }

  /// Local stiffness \f$w\mathbf C(1)+(1-w)\mathbf C(0)\f$.
  [[nodiscard]] Stiffness stiffness_at(double stiffness_weight) const noexcept {
    return Stiffness::blend(m_params.stiffness_at_one, stiffness_weight,
                            m_params.stiffness_at_zero, 1.0 - stiffness_weight);
  }

private:
  /**
   * @brief Neumann-series fixed point (Hu & Chen 2001).
   *
   * The stopping test is on \f$\boldsymbol\tau\f$, measured before the
   * transforms, so the pass that only confirms convergence costs no FFTs —
   * see the file-level note. `iterations` counts \f$\Gamma\f$ applications.
   */
  MicroelasticityReport run_basic(const RealField &h, const RealField &amp) {
    MicroelasticityReport report;
    for (int it = 1; it <= m_params.max_iterations; ++it) {
      build_polarisation(h, amp);
      if (it > 1) {
        const double res = relative_change();
        report.residual = res;
        report.residual_history.push_back(res);
        if (res < m_params.relative_tolerance) {
          report.converged = true;
          break;
        }
      }
      swap_tau();
      apply_green_operator(m_tau_prev, m_strain);
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
      report.converged = res < m_params.relative_tolerance;
    }
    return report;
  }

  /**
   * @brief Eyre–Milton accelerated fixed point.
   *
   * @details
   * Two conditions define the solution: a *local* one,
   * \f$\boldsymbol\sigma = \mathbf C:(\boldsymbol\varepsilon -
   * \boldsymbol\varepsilon^{*})\f$, and a *global* one,
   * \f$\boldsymbol\varepsilon - \bar{\boldsymbol\varepsilon}\f$ compatible and
   * \f$\boldsymbol\sigma\f$ divergence-free. Written in the variables
   *
   * \f[
   *   \mathbf y = \boldsymbol\sigma + \mathbf C_0:\boldsymbol\varepsilon,
   *   \qquad
   *   \mathbf z = \boldsymbol\sigma - \mathbf C_0:\boldsymbol\varepsilon,
   * \f]
   *
   * *each* condition becomes a reflection, and the scheme alternates them
   * (Eyre & Milton 1999; a Peaceman–Rachford splitting).
   *
   * **Global reflection.** \f$\mathcal K\f$ (compatible zero-mean strains) and
   * \f$\mathcal S\f$ (self-equilibrated stresses) are orthogonal complements
   * in the \f$\mathbf C_0\f$ inner product, and \f$\Gamma_0\mathbf C_0\f$ is
   * the projector onto \f$\mathcal K\f$. Splitting \f$\mathbf z\f$ along them
   * and reassembling gives
   * \f$\mathbf y = \mathbf z - 2\mathbf C_0\Gamma_0\mathbf z
   * + 2\mathbf C_0:\bar{\boldsymbol\varepsilon}
   * = \mathbf z + 2\,\mathbf C_0 : W(\mathbf z)\f$, where
   * \f$W = \bar{\boldsymbol\varepsilon} - \Gamma_0\f$ is exactly the Green
   * application the basic scheme already performs. So the accelerated scheme
   * reuses `apply_green_operator` unchanged and costs the same 12 transforms.
   *
   * **Local reflection.** \f$\mathbf y = (\mathbf C+\mathbf C_0):
   * \boldsymbol\varepsilon - \mathbf C:\boldsymbol\varepsilon^{*}\f$ inverts
   * pointwise, and \f$\mathbf z = \mathbf y - 2\mathbf C_0:
   * \boldsymbol\varepsilon\f$. Its gain per channel is
   * \f$(\lambda-\lambda_0)/(\lambda+\lambda_0)\f$ — the sum in the
   * denominator is where the square root comes from — while the global
   * reflection has unit norm, so the composition contracts at
   * \f$(\sqrt r-1)/(\sqrt r+1)\f$ with \f$\mathbf C_0\f$ the geometric mean.
   *
   * The fixed point is the same as the basic scheme's: substituting a true
   * solution reproduces \f$\mathbf y\f$ exactly. The state is \f$\mathbf z\f$,
   * which *is* the basic scheme's polarisation
   * \f$\boldsymbol\tau = \boldsymbol\sigma - \mathbf C_0:\boldsymbol
   * \varepsilon\f$, so the residual is the same quantity in both schemes and
   * the iteration counts are directly comparable. A homogeneous modulus makes
   * the local reflection identically zero, so \f$\mathbf z\f$ is already at
   * its fixed point after the initial polarisation and one pass converges.
   */
  MicroelasticityReport run_eyre_milton(const RealField &h, const RealField &amp) {
    MicroelasticityReport report;
    build_polarisation(h, amp); // m_tau = z^0
    for (int it = 1; it <= m_params.max_iterations; ++it) {
      swap_tau();                                   // m_tau_prev = z
      apply_green_operator(m_tau_prev, m_strain);   // m_strain = W(z)
      const double res = eyre_milton_local(h, amp); // -> m_strain = eps, m_tau = z'
      report.iterations = it;
      report.residual = res;
      report.residual_history.push_back(res);
      if (res < m_params.relative_tolerance) {
        report.converged = true;
        break;
      }
    }
    return report;
  }

  /**
   * @brief The local half of one Eyre–Milton pass; returns the relative change.
   *
   * Reads `m_tau_prev` (\f$\mathbf z\f$) and `m_strain` (\f$W(\mathbf z)\f$),
   * writes `m_strain` (\f$\boldsymbol\varepsilon\f$) and `m_tau`
   * (\f$\mathbf z'\f$). One cell at a time, so the aliasing on `m_strain` is
   * safe.
   */
  double eyre_milton_local(const RealField &h, const RealField &amp) {
    const double *hp = h.data();
    const double *ap = amp.data();
    const Sym3 &pattern = m_params.eigenstrain_pattern;
    std::array<const double *, kSymComponents> zin{};
    std::array<double *, kSymComponents> zout{};
    std::array<double *, kSymComponents> eps{};
    for (int c = 0; c < kSymComponents; ++c) {
      const auto ci = static_cast<std::size_t>(c);
      zin[ci] = m_tau_prev[ci].data();
      zout[ci] = m_tau[ci].data();
      eps[ci] = m_strain[ci].data();
    }
    double diff = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < m_n_local; ++i) {
      Sym3 z;
      Sym3 w;
      for (int c = 0; c < kSymComponents; ++c) {
        const auto ci = static_cast<std::size_t>(c);
        z[c] = zin[ci][i];
        w[c] = eps[ci][i]; // currently holds W(z)
      }
      const Stiffness ci_local = stiffness_at(hp[i]);
      const Sym3 c0w = m_reference_stiffness.contract(w);
      Sym3 y;
      for (int c = 0; c < kSymComponents; ++c) y[c] = z[c] + 2.0 * c0w[c];

      Sym3 estar;
      for (int c = 0; c < kSymComponents; ++c) estar[c] = ap[i] * pattern[c];
      const Sym3 c_estar = ci_local.contract(estar);
      Sym3 rhs;
      for (int c = 0; c < kSymComponents; ++c) rhs[c] = y[c] + c_estar[c];

      const Sym3 e =
          Stiffness::blend(ci_local, 1.0, m_reference_stiffness, 1.0).solve(rhs);
      const Sym3 c0e = m_reference_stiffness.contract(e);
      for (int c = 0; c < kSymComponents; ++c) {
        const auto ci = static_cast<std::size_t>(c);
        const double zn = y[c] - 2.0 * c0e[c];
        diff = std::max(diff, std::abs(zn - z[c]));
        scale = std::max(scale, std::abs(zn));
        zout[ci][i] = zn;
        eps[ci][i] = e[c];
      }
    }
    for (auto &f : m_tau) f.note_host_write();
    for (auto &f : m_strain) f.note_host_write();

    double local[2] = {diff, scale};
    double global[2] = {0.0, 0.0};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, m_params.comm);
    return (global[1] > 0.0) ? global[0] / global[1] : 0.0;
  }

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

    const double c12 = m_reference_stiffness.c12;
    const double c44 = m_reference_stiffness.c44;
    const double aniso = m_reference_stiffness.c11 - m_reference_stiffness.c12 -
                         2.0 * m_reference_stiffness.c44;

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
      const Sym3 s0 = m_reference_stiffness.contract(e);
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

  /**
   * @brief \f$W(\tau) = \bar{\boldsymbol\varepsilon} - \Gamma_0:\tau\f$.
   *
   * Six forward transforms, the Green multiply, six back. Both schemes call
   * this and nothing else touches the FFT, which is why they cost the same
   * per iteration.
   */
  void apply_green_operator(SymRealFields &in, SymRealFields &out) {
    for (int c = 0; c < kSymComponents; ++c) {
      m_fft.forward(in[static_cast<std::size_t>(c)].vec(),
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
      const auto gs = out[0].global_size();
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
                     out[static_cast<std::size_t>(c)].vec());
      out[static_cast<std::size_t>(c)].note_host_write();
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
    const Stiffness dc = Stiffness::blend(m_params.stiffness_at_one, 1.0,
                                          m_params.stiffness_at_zero, -1.0);

    std::array<const double *, kSymComponents> eps{};
    std::array<double *, kSymComponents> sig{};
    for (int c = 0; c < kSymComponents; ++c) {
      eps[static_cast<std::size_t>(c)] =
          m_strain[static_cast<std::size_t>(c)].data();
      sig[static_cast<std::size_t>(c)] =
          m_stress[static_cast<std::size_t>(c)].data();
    }
    double *fel = m_elastic_energy_density.data();
    double *dfel = m_elastic_energy_derivative.data();

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
    m_elastic_energy_density.note_host_write();
    m_elastic_energy_derivative.note_host_write();
  }

  pfc::fft::IHostFFT &m_fft;
  MicroelasticityParams m_params;
  Stiffness m_reference_stiffness;

  SymRealFields m_strain;
  SymRealFields m_stress;
  SymRealFields m_tau;
  SymRealFields m_tau_prev;
  RealField m_elastic_energy_density;
  RealField m_elastic_energy_derivative;
  std::array<ComplexField, kSymComponents> m_hat{};

  std::vector<double> m_kx, m_ky, m_kz;
  std::array<std::vector<double>, kSymComponents> m_g{};
  std::size_t m_zero_mode{static_cast<std::size_t>(-1)};
  std::size_t m_n_local{0};
  std::size_t m_n_outbox{0};
  bool m_has_solution{false};
};

} // namespace pfc::solvers
