// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file coefficient_provenance.hpp
 * @brief Classification of every shipped aluminum material coefficient.
 *
 * This is a provenance table, not a material validation. No record cites a
 * paper, a fit, or a measurement. Reduced length, temperature, time, and
 * amplitude have no SI conversion in this repository.
 */

#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>

namespace aluminum {

enum class CoefficientStatus {
  /// Uncited number copied into the shipped presets.
  representative,
  /// Integrator or case control, not a material property.
  numerical,
  /// Produced by `AluminumParams::recompute_derived`.
  derived
};

/// Sentence shared by every representative coefficient.
inline constexpr std::string_view kUncitedSource =
    "Copied into the shipped presets. This repository does not cite a paper, "
    "a fit, or a measurement for this number.";

struct ModelCoefficient {
  std::string_view key;
  std::string_view meaning;
  std::string_view source;
  std::string_view reduced_unit;
  CoefficientStatus status;
  double value;
  /// When true, every maintained preset must carry this value.
  bool locks_every_preset;
  /// Remains false until a record names a source this repository can open.
  bool cites_external_source;
};

struct DerivedCoefficient {
  std::string_view key;
  std::string_view meaning;
  std::string_view formula;
  std::string_view reduced_unit;
};

struct DensityCoefficient {
  std::string_view key;
  std::string_view meaning;
  std::string_view source;
  std::string_view reduced_unit;
  double value;
  bool cites_external_source;
};

struct ReducedUnitConvention {
  bool length_to_si;
  bool temperature_to_si;
  bool time_to_si;
  bool amplitude_to_si;
  /// True only if this table were evidence that the model matches aluminum.
  bool claims_material_validation;
  std::string_view length;
  std::string_view temperature;
  std::string_view time;
  std::string_view amplitude;
};

inline constexpr std::array<ModelCoefficient, 19> kModelCoefficients{{
    {"T0", "scale that makes tau = T/T0 dimensionless", kUncitedSource,
     "reduced temperature", CoefficientStatus::representative, 89285.0, true, false},
    {"Bx", "prefactor of the FCC correlation peak", kUncitedSource,
     "reduced amplitude", CoefficientStatus::representative, 0.817900686921996, true,
     false},
    {"T_const", "temperature added to the imposed departure", kUncitedSource,
     "reduced temperature", CoefficientStatus::representative, 980.0, true, false},
    {"T_max", "upper clamp of T_const + T_var", kUncitedSource,
     "reduced temperature", CoefficientStatus::representative, 1280.0, true, false},
    {"T_min", "lower clamp of T_const + T_var", kUncitedSource,
     "reduced temperature", CoefficientStatus::representative, 780.0, true, false},
    {"G_grid", "imposed temperature slope; case-specific",
     "Case control. Isothermal demos use zero; other maintained cases may not.",
     "reduced temperature per reduced length", CoefficientStatus::numerical, 0.0,
     false, false},
    {"V_grid", "speed of imposed thermal profile; case-specific",
     "Case control. Isothermal demos use zero; a directional case may not.",
     "reduced length per reduced time", CoefficientStatus::numerical, 0.0, false,
     false},
    {"x_initial", "station where the imposed temperature departure is zero",
     "Case control. The demonstration presets use 130. A seeded preset sets it "
     "to that seed's X0.",
     "reduced length", CoefficientStatus::numerical, 130.0, false, false},
    {"alpha", "width of the FCC correlation peak", kUncitedSource,
     "reduced wavenumber", CoefficientStatus::representative, 0.20, true, false},
    {"lambda", "width of the mean-field filter", kUncitedSource,
     "reduced wavenumber", CoefficientStatus::representative, 0.22, true, false},
    {"stabP", "ETD stabilizer subtracted in the nonlinearity",
     "Numerical control. The shipped value is zero, so the stabilizer does not "
     "enter the free energy.",
     "reduced energy", CoefficientStatus::numerical, 0.0, true, false},
    {"p2_bar", "quadratic density coefficient after the offline vapour shift",
     kUncitedSource, "reduced free-energy coefficient",
     CoefficientStatus::representative, 0.8286531831, true, false},
    {"p3_bar", "cubic density coefficient after the offline vapour shift",
     kUncitedSource, "reduced free-energy coefficient",
     CoefficientStatus::representative, -0.04204863, true, false},
    {"p4_bar", "quartic density coefficient after the offline vapour shift",
     kUncitedSource, "reduced free-energy coefficient",
     CoefficientStatus::representative, 0.007533, true, false},
    {"q20_bar", "temperature-independent mean-field quadratic coefficient",
     kUncitedSource, "reduced free-energy coefficient",
     CoefficientStatus::representative, 0.016531729105214, true, false},
    {"q21_bar", "temperature slope of the mean-field quadratic coefficient",
     kUncitedSource, "reduced free-energy coefficient per reduced temperature",
     CoefficientStatus::representative, 5.467, true, false},
    {"q30_bar", "temperature-independent mean-field cubic coefficient",
     kUncitedSource, "reduced free-energy coefficient",
     CoefficientStatus::representative, 1.7152418049986, true, false},
    {"q31_bar", "temperature slope of the mean-field cubic coefficient",
     kUncitedSource, "reduced free-energy coefficient per reduced temperature",
     CoefficientStatus::representative, 0.45, true, false},
    {"q40_bar", "mean-field quartic coefficient", kUncitedSource,
     "reduced free-energy coefficient", CoefficientStatus::representative, 0.787482,
     true, false},
}};

inline constexpr std::array<DerivedCoefficient, 6> kDerivedCoefficients{{
    {"tau_const", "T_const measured in units of T0", "T_const / T0",
     "dimensionless"},
    {"q2_bar", "mean-field quadratic coefficient at tau_const",
     "q21_bar * tau_const + q20_bar", "reduced free-energy coefficient"},
    {"q2_bar_L", "q2_bar as it enters the linear symbol", "q2_bar",
     "reduced free-energy coefficient"},
    {"q3_bar", "mean-field cubic coefficient at tau_const",
     "q31_bar * tau_const + q30_bar", "reduced free-energy coefficient"},
    {"q4_bar", "mean-field quartic coefficient copied from q40_bar", "q40_bar",
     "reduced free-energy coefficient"},
    {"m_xpos", "front-tracker start taken from x_initial at load", "x_initial",
     "reduced length"},
}};

inline constexpr std::array<DensityCoefficient, 4> kDensityCoefficients{{
    {"n0", "uniform melt density written by the constant initial condition",
     kUncitedSource, "reduced density amplitude", -0.006, false},
    {"rho", "solid density written into an FCC seed", kUncitedSource,
     "reduced density amplitude", -0.036, false},
    {"rho_low", "reservoir density at the high-x end of the fixed band",
     kUncitedSource, "reduced density amplitude", -1.297, false},
    {"rho_high", "reservoir density matched to the melt side of the fixed band",
     kUncitedSource, "reduced density amplitude", -0.006, false},
}};

/// Keys some files still carry under model.params. The schema does not read them.
inline constexpr std::array<std::string_view, 7> kUnreadModelParamKeys{
    "n0", "n_sol", "n_vap", "alpha_farTol", "alpha_highOrd", "shift_u", "shift_s"};

[[nodiscard]] inline std::span<const ModelCoefficient> coefficient_provenance() {
  return kModelCoefficients;
}

[[nodiscard]] inline constexpr double coefficient_value(std::string_view key) {
  for (const ModelCoefficient &rec : kModelCoefficients) {
    if (rec.key == key) return rec.value;
  }
  throw std::invalid_argument("aluminum coefficient: unknown model key");
}

/**
 * @brief Eight cells on the FCC cube edge `a = 2 π √3`.
 *
 * The correlation peak treats wavenumber 1 as the first shell. Nothing
 * multiplies this spacing by a length in metres.
 */
[[nodiscard]] inline double reduced_fcc_cell_spacing() {
  return 2.0 * std::numbers::pi * std::sqrt(3.0) / 8.0;
}

[[nodiscard]] inline ReducedUnitConvention reduced_unit_convention() {
  return {.length_to_si = false,
          .temperature_to_si = false,
          .time_to_si = false,
          .amplitude_to_si = false,
          .claims_material_validation = false,
          .length = "reduced_fcc_cell_spacing() is 2*pi*sqrt(3)/8. No conversion "
                    "to metres is shipped.",
          .temperature =
              "T_const, T_min, T_max, and T0 are reduced temperatures, and "
              "tau = T/T0 is dimensionless. No conversion to kelvin is "
              "shipped.",
          .time = "The PFC step absorbs mobility into the reduced time. dt and "
                  "V_grid use that unit. No conversion to seconds is shipped.",
          .amplitude = "psi and the densities n0, rho, rho_low, and rho_high are "
                       "reduced amplitudes. No conversion to a number density is "
                       "shipped."};
}

} // namespace aluminum
