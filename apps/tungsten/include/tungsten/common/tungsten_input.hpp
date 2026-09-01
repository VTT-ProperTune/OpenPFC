// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_INPUT_HPP
#define TUNGSTEN_INPUT_HPP

#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/parameter_validator.hpp>
#include <tungsten/common/tungsten_params.hpp>
#include <tungsten/tungsten_physics.hpp>

using json = nlohmann::json;
using pfc::ui::ParameterMetadata;
using pfc::ui::ParameterValidator;

/**
 * @brief Create parameter validator with all Tungsten model parameter metadata
 *
 * Defines validation rules, bounds, and documentation for all 21 tungsten
 * parameters. This enables comprehensive validation with helpful error messages.
 *
 * @return Configured ParameterValidator
 */
inline ParameterValidator create_tungsten_validator() {
  ParameterValidator validator;
  validator.set_model_name("Tungsten PFC Model");

  // Thermodynamic parameters
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("n0")
                             .description("Average density of the metastable fluid")
                             .required(true)
                             .range(-1.0, 0.0)
                             .typical(-0.10)
                             .category("Thermodynamics")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("n_sol")
                             .description("Bulk density at solid coexistence")
                             .required(true)
                             .range(-1.0, 0.0)
                             .typical(-0.047)
                             .category("Thermodynamics")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("n_vap")
                             .description("Bulk density at vapor coexistence")
                             .required(true)
                             .range(-1.0, 0.0)
                             .typical(-0.464)
                             .category("Thermodynamics")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("T")
                             .description("Effective temperature")
                             .required(true)
                             .range(0.0, 10000.0)
                             .typical(3300.0)
                             .units("K")
                             .category("Thermodynamics")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("T0")
                             .description("Reference temperature")
                             .required(true)
                             .range(1.0, 1000000.0)
                             .typical(156000.0)
                             .units("K")
                             .category("Thermodynamics")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("Bx")
                             .description("Temperature-dependent coefficient")
                             .required(true)
                             .range(0.0, 2.0)
                             .typical(0.8582)
                             .category("Thermodynamics")
                             .build());

  // Correlation function parameters
  validator.add_metadata(
      ParameterMetadata<double>::builder()
          .name("alpha")
          .description("Width of C2's peak in correlation function")
          .required(true)
          .range(0.0, 2.0)
          .typical(0.50)
          .category("Correlation Function")
          .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("alpha_farTol")
                             .description("Tolerance for k=1 peak effect on k=0")
                             .required(true)
                             .range(0.0, 0.1)
                             .typical(0.001)
                             .category("Correlation Function")
                             .build());

  validator.add_metadata(ParameterMetadata<int>::builder()
                             .name("alpha_highOrd")
                             .description("Power of higher-order Gaussian component "
                                          "(multiple of 2, or 0 to disable)")
                             .required(true)
                             .range(0, 10)
                             .typical(4)
                             .category("Correlation Function")
                             .build());

  // Numerical parameters
  validator.add_metadata(
      ParameterMetadata<double>::builder()
          .name("lambda")
          .description("Strength of meanfield filter (avoid >0.28)")
          .required(true)
          .range(0.0, 0.5)
          .typical(0.22)
          .category("Numerical")
          .build());

  validator.add_metadata(
      ParameterMetadata<double>::builder()
          .name("stabP")
          .description("Numerical stability parameter for exponential integrator")
          .required(true)
          .range(0.0, 1.0)
          .typical(0.2)
          .category("Numerical")
          .build());

  // Vapor model shift parameters
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("shift_u")
                             .description("Vapor-model shift parameter u")
                             .required(true)
                             .typical(0.3341)
                             .category("Vapor Model")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("shift_s")
                             .description("Vapor-model shift parameter s")
                             .required(true)
                             .typical(0.1898)
                             .category("Vapor Model")
                             .build());

  // Vapor model polynomial coefficients
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("p2")
                             .description("Vapor-model polynomial coefficient p2")
                             .required(true)
                             .typical(1.0)
                             .category("Vapor Model")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("p3")
                             .description("Vapor-model polynomial coefficient p3")
                             .required(true)
                             .typical(-0.5)
                             .category("Vapor Model")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("p4")
                             .description("Vapor-model polynomial coefficient p4")
                             .required(true)
                             .typical(0.333333333)
                             .category("Vapor Model")
                             .build());

  // Vapor model q coefficients
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("q20")
                             .description("Vapor-model coefficient q20")
                             .required(true)
                             .typical(-0.0037)
                             .category("Vapor Model")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("q21")
                             .description("Vapor-model coefficient q21")
                             .required(true)
                             .typical(1.0)
                             .category("Vapor Model")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("q30")
                             .description("Vapor-model coefficient q30")
                             .required(true)
                             .typical(-12.4567)
                             .category("Vapor Model")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("q31")
                             .description("Vapor-model coefficient q31")
                             .required(true)
                             .typical(20.0)
                             .category("Vapor Model")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("q40")
                             .description("Vapor-model coefficient q40")
                             .required(true)
                             .typical(45.0)
                             .category("Vapor Model")
                             .build());

  return validator;
}

/**
 * @brief Validate and print summary of Tungsten model parameters
 *
 * Performs comprehensive validation of all 21 tungsten parameters against
 * metadata constraints. Prints validation summary if successful, or detailed
 * error report if validation fails.
 *
 * @param j JSON configuration
 * @throws std::invalid_argument if validation fails
 * @return ValidationResult with all validated parameters
 */
inline pfc::ui::ValidationResult validate_tungsten_params(const json &j) {
  auto validator = create_tungsten_validator();
  auto result = validator.validate(j);

  if (!result.is_valid()) {
    // Print detailed error report
    std::cerr << result.format_errors() << std::endl;
    throw std::invalid_argument("Tungsten model parameter validation failed");
  }

  // Print summary of validated parameters (for reproducibility)
  std::cout << result.format_summary("Tungsten PFC Model") << std::endl;

  return result;
}

/**
 * @brief Read model configuration from json file, under model/params.
 *
 * Validates all parameters before setting them on the model.
 * Prints validation summary for reproducibility.
 *
 * @param j json file
 * @param m model
 * @throws std::invalid_argument if validation fails
 */
void from_json(const json &j, TungstenParams &p) {
  validate_tungsten_params(j);
  tungsten::apply_tungsten_json(j, p);
}

#endif // TUNGSTEN_INPUT_HPP
