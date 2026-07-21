// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file method_composition.hpp
 * @brief Identifier-driven method selection and composition for steppers.
 *
 * @details
 * Drivers obtain a validated, driver-usable integrator via
 * `compose_scalar` / `compose_multi` from a stable method identifier plus
 * `IntegratorComposeConfig` — without switching on method-specific
 * construction code. Euler is registered through
 * `register_method_composer`; additional explicit methods can join the
 * same table later without editing call sites.
 *
 * This is distinct from the typed `steppers::create(Eval&, Model&, …)`
 * helpers in `euler.hpp` / `explicit_rk.hpp`, which bind a known method
 * type to a model + gradient evaluator. Composition here is
 * identifier + config + RHS → `IntegratorComposition<Stepper>` with
 * declared `WorkspaceOwnership` and optional `MethodStateCapability`
 * (empty for stateless Euler; field `save_state` / `restore_state` is
 * orthogonal and is not claimed as method-controller metadata).
 *
 * Identity tokens reuse `RKIntegratorMethod` / `to_string` /
 * `validate_method` from `integrator_method.hpp`. Adaptive capability
 * checks call `validate_method` rather than inventing a second rule.
 *
 * @see euler.hpp for typed `create` factories and `EulerStepper`
 * @see integrator_method.hpp for `RKIntegratorMethod` and `validate_method`
 * @see docs/user_guide/custom_stepper_integration.md
 */

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Who owns scratch / workspace buffers for the composed stepper.
 *
 * Euler composition always reports `MethodOwned` (owns `m_du` /
 * `m_u_checkpoint` scratch inside the stepper).
 */
enum class WorkspaceOwnership {
  MethodOwned, ///< Stepper owns its scratch buffers
  CallerLent,  ///< Caller supplies workspace for the method
  Pooled       ///< Shared / pooled workspace (future)
};

/**
 * @brief Optional checkpointable method-controller state metadata.
 *
 * Stateless Euler leaves `checkpointable == false` and `state_keys` empty.
 * Field rollback via `save_state` / `restore_state` is orthogonal and is
 * not claimed here.
 */
struct MethodStateCapability {
  bool checkpointable{false};
  std::vector<std::string> state_keys{};
};

/**
 * @brief Validated configuration for method composition.
 */
struct IntegratorComposeConfig {
  double dt{0.0};
  bool requires_adaptive{false};
};

/**
 * @brief Fail-fast diagnostics for method composition (before any step).
 */
class ComposeError final : public std::runtime_error {
public:
  enum class Kind {
    UnknownIdentifier,    ///< Method id not recognized
    InvalidConfiguration, ///< e.g. non-positive dt
    CapabilityMismatch    ///< Adaptive / registration / support mismatch
  };

  ComposeError(Kind kind, std::string message)
      : std::runtime_error(std::move(message)), m_kind(kind) {}

  [[nodiscard]] Kind kind() const noexcept { return m_kind; }

private:
  Kind m_kind;
};

/**
 * @brief Driver-usable result of successful method composition.
 *
 * @tparam Stepper Concrete stepper type (e.g. `EulerStepper<Rhs>`).
 */
template <class Stepper> struct IntegratorComposition {
  Stepper stepper;
  RKIntegratorMethod method{};
  WorkspaceOwnership workspace_ownership{WorkspaceOwnership::MethodOwned};
  MethodStateCapability method_state{};
};

/**
 * @brief One entry in the method-composer extension table.
 *
 * Register additional methods with `register_method_composer` so drivers
 * keep calling `compose_scalar` / `compose_multi` without method switches.
 */
struct MethodComposerEntry {
  std::string_view id;
  RKIntegratorMethod method{};
  bool supports_embedded_error{false};
  bool supports_scalar{false};
  bool supports_multi{false};
};

namespace detail {

inline std::vector<MethodComposerEntry> &composer_registry() {
  static std::vector<MethodComposerEntry> registry;
  return registry;
}

inline void ensure_builtin_composers() {
  static bool done = false;
  if (done) {
    return;
  }
  composer_registry().push_back(MethodComposerEntry{
      .id = "euler",
      .method = RKIntegratorMethod::Euler,
      .supports_embedded_error = false,
      .supports_scalar = true,
      .supports_multi = true,
  });
  done = true;
}

inline std::string format_registered_ids() {
  ensure_builtin_composers();
  std::ostringstream oss;
  const auto &reg = composer_registry();
  for (std::size_t i = 0; i < reg.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << reg[i].id;
  }
  return oss.str();
}

inline const MethodComposerEntry *
find_composer(RKIntegratorMethod method) noexcept {
  ensure_builtin_composers();
  for (const auto &entry : composer_registry()) {
    if (entry.method == method) {
      return &entry;
    }
  }
  return nullptr;
}

inline void throw_if_error(std::optional<ComposeError> err) {
  if (err) {
    throw ComposeError(err->kind(), err->what());
  }
}

} // namespace detail

/**
 * @brief Append a method composer entry (extension point).
 *
 * Duplicate ids with the same method and capability flags are ignored.
 * Conflicting re-registration throws `ComposeError::InvalidConfiguration`.
 */
inline void register_method_composer(MethodComposerEntry entry) {
  detail::ensure_builtin_composers();
  auto &reg = detail::composer_registry();
  for (const auto &existing : reg) {
    if (existing.id == entry.id) {
      if (existing.method == entry.method &&
          existing.supports_embedded_error == entry.supports_embedded_error &&
          existing.supports_scalar == entry.supports_scalar &&
          existing.supports_multi == entry.supports_multi) {
        return;
      }
      throw ComposeError(
          ComposeError::Kind::InvalidConfiguration,
          "register_method_composer: duplicate id \"" +
              std::string(entry.id) +
              "\" with conflicting capabilities (already registered)");
    }
  }
  reg.push_back(entry);
}

/**
 * @brief View of currently registered method composers.
 */
inline std::span<const MethodComposerEntry> registered_method_composers() {
  detail::ensure_builtin_composers();
  const auto &reg = detail::composer_registry();
  return std::span<const MethodComposerEntry>(reg.data(), reg.size());
}

/**
 * @brief Map a stable string token to `RKIntegratorMethod`.
 *
 * Tokens match `to_string` (`"euler"`, `"rk4_classical"`, …). Unknown →
 * `nullopt`.
 */
inline std::optional<RKIntegratorMethod>
resolve_method_id(std::string_view id) {
  static constexpr RKIntegratorMethod kAll[] = {
      RKIntegratorMethod::Euler,
      RKIntegratorMethod::RK2_Midpoint,
      RKIntegratorMethod::RK2_Heun,
      RKIntegratorMethod::RK4_Classical,
      RKIntegratorMethod::BogackiShampine32,
  };
  for (RKIntegratorMethod m : kAll) {
    if (to_string(m) == id) {
      return m;
    }
  }
  return std::nullopt;
}

/**
 * @brief Validate compose config against method capabilities.
 *
 * @return empty optional on success; otherwise a `ComposeError` describing
 *         the failure (caller may throw it).
 */
inline std::optional<ComposeError>
validate_compose_config(RKIntegratorMethod method,
                        const IntegratorComposeConfig &cfg) {
  if (!(cfg.dt > 0.0)) {
    std::ostringstream oss;
    oss << "dt must be > 0 (got " << cfg.dt
        << "); set IntegratorComposeConfig::dt to a positive time step";
    return ComposeError(ComposeError::Kind::InvalidConfiguration, oss.str());
  }
  if (cfg.requires_adaptive) {
    if (auto msg = validate_method(method, true)) {
      return ComposeError(ComposeError::Kind::CapabilityMismatch, *msg);
    }
  }
  return std::nullopt;
}

namespace detail {

inline RKIntegratorMethod
resolve_or_throw(std::string_view id) {
  ensure_builtin_composers();
  auto method = resolve_method_id(id);
  if (!method) {
    throw ComposeError(
        ComposeError::Kind::UnknownIdentifier,
        "unknown method identifier \"" + std::string(id) +
            "\"; registered composers: " + format_registered_ids() +
            " (known tokens also include rk2_midpoint, rk2_heun, "
            "rk4_classical, bogacki_shampine32 — register a composer to "
            "construct them)");
  }
  return *method;
}

inline const MethodComposerEntry &
require_composer(std::string_view id, RKIntegratorMethod method,
                 bool need_scalar, bool need_multi) {
  const MethodComposerEntry *entry = find_composer(method);
  if (!entry) {
    throw ComposeError(
        ComposeError::Kind::CapabilityMismatch,
        "no composer registered for \"" + std::string(id) +
            "\"; registered: " + format_registered_ids() +
            " — call register_method_composer to add this method");
  }
  if (need_scalar && !entry->supports_scalar) {
    throw ComposeError(
        ComposeError::Kind::CapabilityMismatch,
        "method \"" + std::string(id) +
            "\" does not support scalar composition; use compose_multi "
            "or register a scalar-capable composer");
  }
  if (need_multi && !entry->supports_multi) {
    throw ComposeError(
        ComposeError::Kind::CapabilityMismatch,
        "method \"" + std::string(id) +
            "\" does not support multi-field composition; use "
            "compose_scalar or register a multi-capable composer");
  }
  return *entry;
}

} // namespace detail

/**
 * @brief Compose a scalar stepper from method id + config + RHS.
 *
 * Fail-fast: throws `ComposeError` before constructing a stepper when the
 * id is unknown, config is invalid, or no composer is registered.
 */
template <class Rhs>
  requires StageFunction<Rhs>
IntegratorComposition<EulerStepper<Rhs>>
compose_scalar(std::string_view id, const IntegratorComposeConfig &cfg,
               std::size_t local_size, Rhs rhs) {
  detail::ensure_builtin_composers();
  const RKIntegratorMethod method = detail::resolve_or_throw(id);
  const MethodComposerEntry &entry =
      detail::require_composer(id, method, /*need_scalar=*/true,
                               /*need_multi=*/false);
  detail::throw_if_error(validate_compose_config(method, cfg));
  if (entry.method != RKIntegratorMethod::Euler) {
    throw ComposeError(
        ComposeError::Kind::CapabilityMismatch,
        "compose_scalar: registered composer for \"" + std::string(id) +
            "\" is not Euler; only Euler construction is implemented in "
            "this seam");
  }
  return IntegratorComposition<EulerStepper<Rhs>>{
      .stepper = EulerStepper<Rhs>(cfg.dt, local_size, std::move(rhs)),
      .method = RKIntegratorMethod::Euler,
      .workspace_ownership = WorkspaceOwnership::MethodOwned,
      .method_state = {},
  };
}

/**
 * @brief Compose a scalar stepper from `RKIntegratorMethod` + config + RHS.
 */
template <class Rhs>
  requires StageFunction<Rhs>
IntegratorComposition<EulerStepper<Rhs>>
compose_scalar(RKIntegratorMethod method, const IntegratorComposeConfig &cfg,
               std::size_t local_size, Rhs rhs) {
  const std::string id = to_string(method);
  return compose_scalar(std::string_view{id}, cfg, local_size, std::move(rhs));
}

/**
 * @brief Compose a multi-field stepper from method id + config + RHS.
 */
template <class Rhs, std::size_t N>
IntegratorComposition<MultiEulerStepper<Rhs, N>>
compose_multi(std::string_view id, const IntegratorComposeConfig &cfg,
              const std::array<std::size_t, N> &local_sizes, Rhs rhs) {
  detail::ensure_builtin_composers();
  const RKIntegratorMethod method = detail::resolve_or_throw(id);
  const MethodComposerEntry &entry =
      detail::require_composer(id, method, /*need_scalar=*/false,
                               /*need_multi=*/true);
  detail::throw_if_error(validate_compose_config(method, cfg));
  if (entry.method != RKIntegratorMethod::Euler) {
    throw ComposeError(
        ComposeError::Kind::CapabilityMismatch,
        "compose_multi: registered composer for \"" + std::string(id) +
            "\" is not Euler; only Euler construction is implemented in "
            "this seam");
  }
  return IntegratorComposition<MultiEulerStepper<Rhs, N>>{
      .stepper =
          MultiEulerStepper<Rhs, N>(cfg.dt, local_sizes, std::move(rhs)),
      .method = RKIntegratorMethod::Euler,
      .workspace_ownership = WorkspaceOwnership::MethodOwned,
      .method_state = {},
  };
}

/**
 * @brief Compose a multi-field stepper from `RKIntegratorMethod` + config.
 */
template <class Rhs, std::size_t N>
IntegratorComposition<MultiEulerStepper<Rhs, N>>
compose_multi(RKIntegratorMethod method, const IntegratorComposeConfig &cfg,
              const std::array<std::size_t, N> &local_sizes, Rhs rhs) {
  const std::string id = to_string(method);
  return compose_multi(std::string_view{id}, cfg, local_sizes, std::move(rhs));
}

} // namespace pfc::sim::steppers
