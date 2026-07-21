// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file adaptive_control_config.hpp
 * @brief Validated adaptive-control policy parameters (tolerances, dt bounds, mode)
 *
 * @details
 * Backend-neutral value type for external adaptive timestepping policy:
 * absolute/relative tolerances (scalar and/or per-field), safety factor,
 * growth/shrink limits, min/max dt, rejection cap, fixed-vs-adaptive mode,
 * and optional error weights / norm selection.
 *
 * Fixed mode is representable with defaults and does not require adaptive-only
 * domain checks. Adaptive mode aggregates all validation issues before use.
 *
 * Configuration identity (`make_identity`) exposes a semantic version plus a
 * deterministic parameter signature suitable as checkpoint *metadata only*
 * (not evolving runtime counters). Controllers, embedded RK steppers, and
 * simulator wiring are out of scope for this header.
 *
 * @note v1 has no alternate recovery mode for `shrink_max`: values must lie
 *       in the open unit interval `(0, 1)`.
 *
 * @see docs/development/time_integration_architecture.md
 */

#ifndef PFC_SIM_ADAPTIVE_CONTROL_CONFIG_HPP
#define PFC_SIM_ADAPTIVE_CONTROL_CONFIG_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc {
namespace sim {

/// Semantic version of the AdaptiveControlConfig parameter contract.
inline constexpr int k_adaptive_control_config_version_major = 1;
inline constexpr int k_adaptive_control_config_version_minor = 0;
inline constexpr int k_adaptive_control_config_version_patch = 0;

/**
 * @brief Whether the adaptive controller is active
 *
 * `fixed` keeps a constant dt (adaptive-only parameters need not be set).
 * `adaptive` requires the validated scale/bounds set.
 */
enum class AdaptiveControlMode : std::uint8_t { fixed, adaptive };

/**
 * @brief Norm used to reduce local error estimates across fields/components
 */
enum class AdaptiveErrorNorm : std::uint8_t { max_norm, weighted_l2 };

/**
 * @brief Adaptive-control policy parameters
 *
 * Empty `atol_per_field` / `rtol_per_field` means broadcast the scalar
 * `atol` / `rtol` to every controlled field. When `field_count > 0`, any
 * non-empty per-field vector must have that length.
 */
struct AdaptiveControlConfig {
  AdaptiveControlMode mode = AdaptiveControlMode::fixed;
  double atol = 0.0;
  double rtol = 0.0;
  std::vector<double> atol_per_field; ///< empty => use scalar atol
  std::vector<double> rtol_per_field; ///< empty => use scalar rtol
  std::size_t field_count = 0;        ///< 0 => do not enforce length vs model
  double safety_factor = 0.9;         ///< adaptive: (0, 1]
  double growth_max = 2.0;            ///< adaptive: >= 1
  double shrink_max = 0.5;            ///< adaptive: (0, 1); v1 no alternate recovery
  double min_dt = 1e-12;
  double max_dt = 1.0;
  int max_sequential_rejections = 10;
  std::vector<double> error_weights; ///< empty => treated as all-ones when adaptive
  AdaptiveErrorNorm error_norm = AdaptiveErrorNorm::weighted_l2;
};

/**
 * @brief One validation finding for an adaptive-control parameter
 */
struct AdaptiveConfigIssue {
  std::string parameter;     ///< e.g. "growth_max" or "atol_per_field[1]"
  std::string value;         ///< rendered provided value
  std::string rule;          ///< short rule id / sentence
  std::string allowed_range; ///< e.g. ">= 1" or "(0, 1)"
};

/**
 * @brief Aggregate of validation issues (empty => valid)
 */
struct AdaptiveConfigValidationResult {
  std::vector<AdaptiveConfigIssue> issues;

  [[nodiscard]] bool ok() const noexcept { return issues.empty(); }

  /**
   * @brief Multi-line aggregate suitable for exception messages
   */
  [[nodiscard]] std::string format() const {
    if (issues.empty()) {
      return {};
    }
    std::ostringstream oss;
    oss << "AdaptiveControlConfig validation failed (" << issues.size() << " issue"
        << (issues.size() == 1 ? "" : "s") << "):\n";
    for (const auto &issue : issues) {
      oss << "  - parameter='" << issue.parameter << "' value='" << issue.value
          << "' rule='" << issue.rule << "' allowed_range='" << issue.allowed_range
          << "'\n";
    }
    return oss.str();
  }
};

/**
 * @brief Stable configuration identity for checkpoint metadata
 *
 * Not evolving runtime state — only the parameter contract version and a
 * deterministic encoding of the configured values.
 */
struct AdaptiveConfigIdentity {
  int version_major = k_adaptive_control_config_version_major;
  int version_minor = k_adaptive_control_config_version_minor;
  int version_patch = k_adaptive_control_config_version_patch;
  std::string parameter_signature;

  [[nodiscard]] std::string semantic_version() const {
    return std::to_string(version_major) + '.' + std::to_string(version_minor) +
           '.' + std::to_string(version_patch);
  }
};

namespace detail {

[[nodiscard]] inline std::string format_double(double v) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<double>::max_digits10);
  oss << v;
  return oss.str();
}

[[nodiscard]] inline std::string format_int(int v) { return std::to_string(v); }

[[nodiscard]] inline std::string format_size(std::size_t v) {
  return std::to_string(v);
}

[[nodiscard]] inline bool is_finite(double v) noexcept { return std::isfinite(v); }

inline void push_issue(AdaptiveConfigValidationResult &result, std::string parameter,
                       std::string value, std::string rule, std::string allowed_range) {
  result.issues.push_back(AdaptiveConfigIssue{
      .parameter = std::move(parameter),
      .value = std::move(value),
      .rule = std::move(rule),
      .allowed_range = std::move(allowed_range),
  });
}

inline void require_finite(AdaptiveConfigValidationResult &result, const char *name,
                           double value) {
  if (!is_finite(value)) {
    push_issue(result, name, format_double(value), "must be finite", "finite double");
  }
}

[[nodiscard]] inline double effective_atol(const AdaptiveControlConfig &cfg,
                                           std::size_t i) {
  if (!cfg.atol_per_field.empty()) {
    return cfg.atol_per_field[i];
  }
  return cfg.atol;
}

[[nodiscard]] inline double effective_rtol(const AdaptiveControlConfig &cfg,
                                           std::size_t i) {
  if (!cfg.rtol_per_field.empty()) {
    return cfg.rtol_per_field[i];
  }
  return cfg.rtol;
}

inline void append_vector_signature(std::ostringstream &oss, const char *name,
                                    const std::vector<double> &values) {
  oss << name << "=[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ',';
    }
    oss << format_double(values[i]);
  }
  oss << ']';
}

} // namespace detail

/**
 * @brief Validate adaptive-control configuration (mode-aware, aggregates all issues)
 *
 * Fixed mode skips adaptive-only domain checks but still rejects non-finite
 * values when numeric fields are inspected. Adaptive mode enforces tolerances,
 * growth/shrink/dt bounds, rejection limit, safety factor, and error weights.
 */
[[nodiscard]] inline AdaptiveConfigValidationResult
validate(const AdaptiveControlConfig &cfg) {
  AdaptiveConfigValidationResult result;

  // Always reject non-finite scalars that are part of the stored config.
  detail::require_finite(result, "atol", cfg.atol);
  detail::require_finite(result, "rtol", cfg.rtol);
  detail::require_finite(result, "safety_factor", cfg.safety_factor);
  detail::require_finite(result, "growth_max", cfg.growth_max);
  detail::require_finite(result, "shrink_max", cfg.shrink_max);
  detail::require_finite(result, "min_dt", cfg.min_dt);
  detail::require_finite(result, "max_dt", cfg.max_dt);

  for (std::size_t i = 0; i < cfg.atol_per_field.size(); ++i) {
    if (!detail::is_finite(cfg.atol_per_field[i])) {
      detail::push_issue(result, "atol_per_field[" + detail::format_size(i) + "]",
                         detail::format_double(cfg.atol_per_field[i]), "must be finite",
                         "finite double");
    }
  }
  for (std::size_t i = 0; i < cfg.rtol_per_field.size(); ++i) {
    if (!detail::is_finite(cfg.rtol_per_field[i])) {
      detail::push_issue(result, "rtol_per_field[" + detail::format_size(i) + "]",
                         detail::format_double(cfg.rtol_per_field[i]), "must be finite",
                         "finite double");
    }
  }
  for (std::size_t i = 0; i < cfg.error_weights.size(); ++i) {
    if (!detail::is_finite(cfg.error_weights[i])) {
      detail::push_issue(result, "error_weights[" + detail::format_size(i) + "]",
                         detail::format_double(cfg.error_weights[i]), "must be finite",
                         "finite double");
    }
  }

  if (cfg.mode == AdaptiveControlMode::fixed) {
    return result;
  }

  // --- Adaptive mode ---

  const bool have_atol_vec = !cfg.atol_per_field.empty();
  const bool have_rtol_vec = !cfg.rtol_per_field.empty();

  if (have_atol_vec && have_rtol_vec &&
      cfg.atol_per_field.size() != cfg.rtol_per_field.size()) {
    detail::push_issue(result, "atol_per_field/rtol_per_field",
                       detail::format_size(cfg.atol_per_field.size()) + "/" +
                           detail::format_size(cfg.rtol_per_field.size()),
                       "per-field tolerance vectors must have equal size",
                       "equal lengths");
  }

  if (cfg.field_count > 0) {
    if (have_atol_vec && cfg.atol_per_field.size() != cfg.field_count) {
      detail::push_issue(result, "atol_per_field",
                         detail::format_size(cfg.atol_per_field.size()),
                         "length must equal field_count",
                         "size == field_count (" + detail::format_size(cfg.field_count) +
                             ")");
    }
    if (have_rtol_vec && cfg.rtol_per_field.size() != cfg.field_count) {
      detail::push_issue(result, "rtol_per_field",
                         detail::format_size(cfg.rtol_per_field.size()),
                         "length must equal field_count",
                         "size == field_count (" + detail::format_size(cfg.field_count) +
                             ")");
    }
    if (!cfg.error_weights.empty() && cfg.error_weights.size() != cfg.field_count) {
      detail::push_issue(result, "error_weights",
                         detail::format_size(cfg.error_weights.size()),
                         "length must equal field_count",
                         "size == field_count (" + detail::format_size(cfg.field_count) +
                             ")");
    }
  }

  // Walk effective fields when vector lengths are consistent enough to index.
  const bool length_mismatch =
      (have_atol_vec && have_rtol_vec &&
       cfg.atol_per_field.size() != cfg.rtol_per_field.size()) ||
      (cfg.field_count > 0 && have_atol_vec &&
       cfg.atol_per_field.size() != cfg.field_count) ||
      (cfg.field_count > 0 && have_rtol_vec &&
       cfg.rtol_per_field.size() != cfg.field_count);

  if (!length_mismatch) {
    std::size_t walk = 1;
    if (have_atol_vec || have_rtol_vec) {
      walk = have_atol_vec ? cfg.atol_per_field.size() : cfg.rtol_per_field.size();
    } else if (cfg.field_count > 0) {
      walk = cfg.field_count;
    }
    for (std::size_t i = 0; i < walk; ++i) {
      const double a = detail::effective_atol(cfg, i);
      const double r = detail::effective_rtol(cfg, i);
      const std::string a_name =
          have_atol_vec ? ("atol_per_field[" + detail::format_size(i) + "]") : "atol";
      const std::string r_name =
          have_rtol_vec ? ("rtol_per_field[" + detail::format_size(i) + "]") : "rtol";

      if (a < 0.0) {
        detail::push_issue(result, a_name, detail::format_double(a),
                           "absolute tolerance must be non-negative", ">= 0");
      }
      if (r < 0.0) {
        detail::push_issue(result, r_name, detail::format_double(r),
                           "relative tolerance must be non-negative", ">= 0");
      }
      if (a == 0.0 && r == 0.0) {
        detail::push_issue(result, a_name + "/" + r_name,
                           detail::format_double(a) + "/" + detail::format_double(r),
                           "at least one of absolute or relative tolerance must be "
                           "strictly positive",
                           "atol > 0 or rtol > 0");
      }
    }
  }

  if (!(cfg.growth_max >= 1.0)) {
    detail::push_issue(result, "growth_max", detail::format_double(cfg.growth_max),
                       "growth_max must be at least 1", ">= 1");
  }

  if (!(cfg.shrink_max > 0.0 && cfg.shrink_max < 1.0)) {
    detail::push_issue(result, "shrink_max", detail::format_double(cfg.shrink_max),
                       "shrink_max must lie in the open unit interval (v1: no alternate "
                       "recovery mode)",
                       "(0, 1)");
  }

  if (!(cfg.min_dt > 0.0)) {
    detail::push_issue(result, "min_dt", detail::format_double(cfg.min_dt),
                       "min_dt must be strictly positive", "> 0");
  }
  if (!(cfg.max_dt > 0.0)) {
    detail::push_issue(result, "max_dt", detail::format_double(cfg.max_dt),
                       "max_dt must be strictly positive", "> 0");
  }
  if (detail::is_finite(cfg.min_dt) && detail::is_finite(cfg.max_dt) &&
      cfg.min_dt > 0.0 && cfg.max_dt > 0.0 && cfg.min_dt > cfg.max_dt) {
    detail::push_issue(result, "min_dt/max_dt",
                       detail::format_double(cfg.min_dt) + "/" +
                           detail::format_double(cfg.max_dt),
                       "min_dt must not exceed max_dt", "min_dt <= max_dt");
  }

  if (!(cfg.max_sequential_rejections > 0)) {
    detail::push_issue(result, "max_sequential_rejections",
                       detail::format_int(cfg.max_sequential_rejections),
                       "rejection limit must be strictly positive", "> 0");
  }

  if (!(cfg.safety_factor > 0.0 && cfg.safety_factor <= 1.0)) {
    detail::push_issue(result, "safety_factor", detail::format_double(cfg.safety_factor),
                       "safety_factor must lie in (0, 1]", "(0, 1]");
  }

  if (!cfg.error_weights.empty()) {
    bool any_positive = false;
    for (double w : cfg.error_weights) {
      if (w > 0.0) {
        any_positive = true;
        break;
      }
    }
    if (!any_positive) {
      detail::push_issue(result, "error_weights", "all non-positive",
                         "at least one error weight must be strictly positive",
                         "exists i: error_weights[i] > 0");
    }
  }

  return result;
}

/**
 * @brief Build checkpoint-metadata identity for a configuration
 *
 * Signature is a portable textual canonicalization of mode and all numeric /
 * vector fields in a fixed order (max_digits10). Identical parameter sets yield
 * identical signatures.
 */
[[nodiscard]] inline AdaptiveConfigIdentity
make_identity(const AdaptiveControlConfig &cfg) {
  AdaptiveConfigIdentity id;
  std::ostringstream oss;
  oss.precision(std::numeric_limits<double>::max_digits10);
  oss << "mode=" << (cfg.mode == AdaptiveControlMode::fixed ? "fixed" : "adaptive");
  oss << ";atol=" << detail::format_double(cfg.atol);
  oss << ";rtol=" << detail::format_double(cfg.rtol);
  detail::append_vector_signature(oss, ";atol_per_field", cfg.atol_per_field);
  detail::append_vector_signature(oss, ";rtol_per_field", cfg.rtol_per_field);
  oss << ";field_count=" << cfg.field_count;
  oss << ";safety_factor=" << detail::format_double(cfg.safety_factor);
  oss << ";growth_max=" << detail::format_double(cfg.growth_max);
  oss << ";shrink_max=" << detail::format_double(cfg.shrink_max);
  oss << ";min_dt=" << detail::format_double(cfg.min_dt);
  oss << ";max_dt=" << detail::format_double(cfg.max_dt);
  oss << ";max_sequential_rejections=" << cfg.max_sequential_rejections;
  detail::append_vector_signature(oss, ";error_weights", cfg.error_weights);
  oss << ";error_norm="
      << (cfg.error_norm == AdaptiveErrorNorm::max_norm ? "max_norm" : "weighted_l2");
  id.parameter_signature = oss.str();
  return id;
}

/**
 * @brief Validate and return a copy; throws std::invalid_argument on failure
 */
[[nodiscard]] inline AdaptiveControlConfig
make_adaptive_control_config(AdaptiveControlConfig cfg) {
  const AdaptiveConfigValidationResult result = validate(cfg);
  if (!result.ok()) {
    throw std::invalid_argument(result.format());
  }
  return cfg;
}

} // namespace sim
} // namespace pfc

#endif // PFC_SIM_ADAPTIVE_CONTROL_CONFIG_HPP
