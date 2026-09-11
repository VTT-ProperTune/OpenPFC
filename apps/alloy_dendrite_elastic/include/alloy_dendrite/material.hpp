// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file material.hpp
 * @brief Al-4.5 wt% Cu, in SI, with provenance -- and the arithmetic that
 *        turns it into the dimensionless inputs of equations (5)-(7).
 *
 * @details
 * ## Why a separate header with this much comment in it
 *
 * Every number below is either a measured property with a citation or a
 * derived quantity with the derivation written next to it. That is not
 * decoration. A coupled phase-field/elasticity run has exactly two ways to
 * produce a confident wrong answer: a sign error, and a parameter that was
 * "about right" and turned out to be off by the factor that decided the
 * conclusion. The first is caught by tests; the second is only caught by
 * making the provenance auditable. Where a value is *representative* rather
 * than assessed for this alloy, it says so in those words.
 *
 * ## The alloy
 *
 * Al-4.5 wt% Cu is the textbook dilute binary: it is the alloy the
 * quantitative dilute-alloy phase field was built around, its `k = 0.14` is
 * within 7% of the `k = 0.15` the Stage-1/Stage-2 cases already use, and its
 * solute misfit is large enough that the elastic coupling is not
 * automatically negligible. The phase-field runs keep `k = 0.15`; the
 * difference is 7% in one parameter and is not worth a second set of
 * verification numbers, but it is a difference and it is recorded here
 * rather than silently rounded away.
 *
 * ## The two conversions that matter
 *
 * **1. Stiffness.** Equation (2) is dimensionless, and the elastic term is
 * `- lambda_el (1-phi^2)^2 d f_el/d phi`. Matching it against the chemical
 * term `- lambda (1-phi^2)^2 U` (see `elasticity.hpp`) gives
 * `lambda_el = lambda / f_ref` with
 *
 *     f_ref = L dT_0 / T_M ,    dT_0 = |m| c_l^0 (1 - k)
 *
 * the physical free-energy density that corresponds to one unit of `U`.
 * Equivalently: divide every stiffness by `f_ref` and then `lambda_el =
 * lambda`. @ref kStiffnessScale is that divisor and @ref al_cu_solid_stiffness
 * applies it.
 *
 * **2. Eigenstrain.** `eps_c` is `d eps* / d U`, not `d eps* / d c`. One unit
 * of `U` is a composition change of `(1-k) c_l^0`, so
 *
 *     eps_c = (1/a) (da/dc) * (1 - k) c_l^0
 *
 * with `c` in the same units as `da/dc`. Getting this wrong by the factor
 * `(1-k) c_l^0` is the single easiest way to be off by two orders of
 * magnitude in the elastic energy, which is why it is spelled out.
 *
 * `eps_T` is `d eps* / d theta`, and `theta` is scaled by the hypercooling
 * `L / c_p` (equation (4)'s source is `(1/2) d_t phi`, so a full
 * transformation `phi: -1 -> +1` raises `theta` by exactly 1). Hence
 * `eps_T = alpha_L * L / c_p`.
 *
 * @see elasticity.hpp for the `lambda_el` derivation
 * @see MODEL_SPEC.md equations (5)-(7)
 */

#include <openpfc_apps/microelasticity.hpp>

namespace alloy_dendrite::material {

// ---------------------------------------------------------------------------
// Measured properties, SI
// ---------------------------------------------------------------------------

/// Melting point of pure aluminium, K. CRC Handbook, 97th ed.
inline constexpr double kMeltingPointAl = 933.47;

/// Latent heat of fusion of pure aluminium, J/kg. CRC Handbook, 97th ed.
/// (10.71 kJ/mol over 26.98 g/mol).
inline constexpr double kLatentHeatPerMass = 3.97e5;

/// Density of liquid aluminium near the melting point, kg/m^3.
/// Assael et al., *J. Phys. Chem. Ref. Data* **35**, 285 (2006).
inline constexpr double kDensity = 2375.0;

/// Latent heat per unit volume, J/m^3 -- the `L` of `f_ref`.
inline constexpr double kLatentHeatPerVolume = kLatentHeatPerMass * kDensity;

/// Specific heat of liquid Al near `T_M`, J/(kg K). Assael et al. (2006).
inline constexpr double kSpecificHeat = 1.18e3;

/// Equilibrium partition coefficient of Cu in Al. Al-Cu phase diagram;
/// the value used throughout the quantitative dilute-alloy phase-field
/// literature (Echebarria, Folch, Karma & Plapp, PRE 70, 061604 (2004)).
inline constexpr double kPartition = 0.14;

/// Liquidus slope, K per wt% Cu. Same source; magnitude only.
inline constexpr double kLiquidusSlope = 3.4;

/// Nominal alloy composition and the reference liquidus composition, wt% Cu.
inline constexpr double kCompositionWtPct = 4.5;

/// Atomic masses, g/mol (CRC Handbook), for the wt% -> at% conversion.
inline constexpr double kMolarMassCu = 63.546;
inline constexpr double kMolarMassAl = 26.9815;

/// Lattice parameter of pure fcc Al at 298 K, angstrom. CRC Handbook.
inline constexpr double kLatticeParamAl = 4.0496;

/**
 * @brief Vegard slope of the Al-rich fcc solid solution, angstrom per at% Cu.
 *
 * Cu is the smaller atom, so the lattice contracts: the slope is negative.
 * **Representative, not assessed here** -- reported values for the dilute
 * Al(Cu) solid solution cluster around `-0.017` to `-0.018` angstrom/at%, and
 * the exact figure depends on the temperature and on how the extrapolation
 * to the (metastable) full solubility range is done. The value below is the
 * middle of that band. The elastic energy scales as its square, so a 6%
 * uncertainty here is a 12% uncertainty in `f_el`; that is small next to the
 * order-of-magnitude questions this application is asking, and it is *not*
 * small enough to quote three significant figures of `f_el` from.
 */
inline constexpr double kVegardSlopePerAtPct = -0.0175;

/// Single-crystal cubic elastic constants of Al at 300 K, Pa.
/// Kamm & Alers, *J. Appl. Phys.* **35**, 327 (1964).
inline constexpr double kC11_300K = 108.2e9;
inline constexpr double kC12_300K = 61.3e9;
inline constexpr double kC44_300K = 28.5e9;

/**
 * @brief Fraction of the 300 K stiffness retained just below `T_M`.
 *
 * Aluminium's elastic constants fall roughly linearly with temperature and
 * have lost of order half their room-temperature value by the melting point
 * (Kamm & Alers 1964 measure to 800 K; the extrapolation to 933 K is
 * **representative, not measured here**). A solidification front is at `T_M`,
 * not at 300 K, so running with the room-temperature constants would
 * overstate the elastic energy by about a factor of four. The default below
 * applies this factor; `--el-soften=1` recovers the 300 K numbers for a
 * sensitivity run.
 */
inline constexpr double kSofteningAtTm = 0.5;

/// Linear thermal expansion coefficient of solid Al near `T_M`, 1/K.
/// **Representative**: the 300 K value is 23.1e-6 and it rises with
/// temperature; 3.0e-5 is the usual near-melting figure.
inline constexpr double kThermalExpansion = 3.0e-5;

// ---------------------------------------------------------------------------
// Derived dimensionless quantities
// ---------------------------------------------------------------------------

/// Nominal composition in at% Cu, from @ref kCompositionWtPct.
inline constexpr double kCompositionAtPct =
    100.0 * (kCompositionWtPct / kMolarMassCu) /
    (kCompositionWtPct / kMolarMassCu +
     (100.0 - kCompositionWtPct) / kMolarMassAl);

/// Freezing range `dT_0 = |m| c_l^0 (1 - k)`, K.
inline constexpr double kFreezingRange =
    kLiquidusSlope * kCompositionWtPct * (1.0 - kPartition);

/**
 * @brief `f_ref = L dT_0 / T_M`, J/m^3: the chemical free-energy density that
 *        one unit of `U` is worth. Divide stiffnesses by this.
 *
 * For Al-4.5 wt% Cu this is about 1.4e7 J/m^3, i.e. 0.014 GPa -- four orders
 * of magnitude below the elastic constants. That ratio is the reason the
 * dimensionless stiffnesses below are numbers in the thousands, and it is
 * also the reason the elastic coupling is not negligible despite the
 * eigenstrain being well under a percent: `f_el ~ C eps*^2 / 2` beats
 * `f_ref` whenever `eps* > sqrt(2 f_ref / C) ~ 5e-4`.
 */
inline constexpr double kStiffnessScale =
    kLatentHeatPerVolume * kFreezingRange / kMeltingPointAl;

/// Hypercooling `L / c_p`, K -- the unit of the dimensionless `theta`.
inline constexpr double kHypercooling =
    kLatentHeatPerMass / kSpecificHeat;

/**
 * @brief `eps_c = d eps* / d U` for Cu in Al.
 *
 * `(1/a)(da/dx_at)` times the composition change `(1-k) c_l^0` that one unit
 * of `U` represents. Negative: rejecting Cu into the liquid leaves solid that
 * is *less* Cu-rich than the interface liquid, and adding Cu contracts the
 * lattice, so a solute-enriched solid is a contracted solid.
 */
inline constexpr double kEpsC = (kVegardSlopePerAtPct / kLatticeParamAl) *
                                (1.0 - kPartition) * kCompositionAtPct;

/// `eps_T = d eps* / d theta = alpha_L * (L / c_p)`.
inline constexpr double kEpsT = kThermalExpansion * kHypercooling;

/**
 * @brief Solid stiffness in units of @ref kStiffnessScale.
 *
 * @param soften Multiplier on the 300 K constants; @ref kSofteningAtTm by
 *               default, 1 for the room-temperature values.
 */
[[nodiscard]] inline pfc::apps::Stiffness
al_cu_solid_stiffness(double soften = kSofteningAtTm) noexcept {
  const double s = soften / kStiffnessScale;
  return pfc::apps::Stiffness::cubic(kC11_300K * s, kC12_300K * s, kC44_300K * s);
}

} // namespace alloy_dendrite::material
