// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_world_time.hpp
 * @brief `from_json` specializations for `Domain` and `Time`
 */

#ifndef PFC_UI_FROM_JSON_WORLD_TIME_HPP
#define PFC_UI_FROM_JSON_WORLD_TIME_HPP

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/frontend/ui/from_json_integrator_method.hpp>
#include <openpfc/frontend/ui/json_helpers.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::ui {

namespace detail {

/// Keys of the removed `Lx/dx/origin` domain format.
inline constexpr std::array<const char *, 8> legacy_domain_keys{
    "Lx", "Ly", "Lz", "dx", "dy", "dz", "origin", "origo"};

/**
 * @brief `{lower, upper}` with `(upper - lower) / n == d` bit for bit when one
 *        exists (symmetric first), otherwise the closest. Mirrors
 *        `scripts/migrate_domain_block.py`.
 */
inline std::pair<double, double> exact_box_limits(int n, double d, double lower) {
  std::vector<double> starts;
  if (lower < 0.0) starts.push_back(-lower);
  starts.push_back(lower + n * d);
  double best = starts.back();
  double best_err = std::numeric_limits<double>::infinity();
  for (double start : starts) {
    double lo = start, hi = start;
    for (int k = 0; k < 64; ++k) {
      for (double u : {lo, hi}) {
        if (!(u > lower)) continue;
        const double err = std::abs((u - lower) / n - d);
        if (err == 0.0) return {lower, u};
        if (err < best_err) {
          best_err = err;
          best = u;
        }
      }
      lo = std::nextafter(lo, -std::numeric_limits<double>::infinity());
      hi = std::nextafter(hi, std::numeric_limits<double>::infinity());
    }
  }
  return {lower, best};
}

/// The `size/lower/upper` block equivalent to a legacy block, or null.
inline json legacy_domain_equivalent(const json &d) {
  try {
    const int nx = d.at("Lx").get<int>();
    const int ny = d.value("Ly", 1);
    const int nz = d.value("Lz", 1);
    const double dx = d.at("dx").get<double>();
    const std::array<int, 3> n{nx, ny, nz};
    const std::array<double, 3> h{dx, d.value("dy", dx), d.value("dz", dx)};
    const std::string origin =
        d.value("origin", d.value("origo", std::string("corner")));
    if (origin != "center" && origin != "corner") return nullptr;
    json lower = json::array(), upper = json::array();
    for (int a = 0; a < 3; ++a) {
      const double x0 = origin == "center" ? -0.5 * h[a] * n[a] : 0.0;
      const auto [lo, up] = exact_box_limits(n[a], h[a], x0);
      lower.push_back(lo);
      upper.push_back(up);
    }
    return json{{"size", n}, {"lower", lower}, {"upper", upper}};
  } catch (const std::exception &) {
    return nullptr;
  }
}

/// Throw, naming the replacement, when a removed domain key is present.
inline void reject_legacy_domain_keys(const json &root, const json &d) {
  std::vector<std::string> found;
  for (const char *k : legacy_domain_keys) {
    if (d.contains(k) || root.contains(k)) found.emplace_back(k);
  }
  if (found.empty()) return;
  std::ostringstream oss;
  oss << "Invalid configuration: the domain keys";
  for (const auto &k : found) oss << " '" << k << "'";
  oss << " were removed. A domain is now its grid size and the lower and upper "
         "limits of the box, \"domain\": {\"size\": [Nx, Ny, Nz], \"lower\": "
         "[x0, y0, z0], \"upper\": [x1, y1, z1]}, with spacing (upper - lower) "
         "/ size on each periodic axis.";
  const json legacy = d.contains("Lx") ? d : root;
  if (const json eq = legacy_domain_equivalent(legacy); !eq.is_null()) {
    oss << " This input is equivalent to \"domain\": " << eq.dump() << ".";
  }
  oss << " scripts/migrate_domain_block.py rewrites input files in place.";
  throw std::invalid_argument(oss.str());
}

/// A JSON array of exactly three numbers that pass @p ok.
template <class T, class Ok>
std::array<T, 3> read_triple(const json &d, const char *key, const char *what,
                             const char *expected, const char *example, Ok ok) {
  const auto fail = [&](const std::string &got) {
    throw std::invalid_argument(
        format_config_error(std::string("domain.") + key, what, expected, got, {},
                            std::string("\"") + key + "\": " + example));
  };
  if (!d.contains(key)) fail("missing");
  const json &v = d.at(key);
  if (!v.is_array() || v.size() != 3) fail(v.dump());
  std::array<T, 3> out{};
  for (std::size_t a = 0; a < 3; ++a) {
    const json &e = v[a];
    if constexpr (std::is_same_v<T, bool>) {
      if (!e.is_boolean()) fail(v.dump());
    } else if constexpr (std::is_integral_v<T>) {
      if (!e.is_number_integer()) fail(v.dump());
    } else {
      if (!e.is_number()) fail(v.dump());
    }
    out[a] = e.get<T>();
    if (!ok(out[a])) fail(v.dump());
  }
  return out;
}

} // namespace detail

/**
 * Creates a Domain from the `domain` block of an input:
 *
 * @code{.json}
 * "domain": {"size": [256, 256, 256],
 *            "lower": [-142.2, -142.2, -142.2],
 *            "upper": [142.2, 142.2, 142.2]}
 * @endcode
 *
 * - `size`: grid points per axis, positive integers.
 * - `lower`, `upper`: limits of the box, finite, `upper > lower` per axis.
 *
 * `periodic` is an optional array of three booleans (default: all true).
 * Periodic axes use `(upper - lower) / size`; bounded axes include both
 * endpoints and use `(upper - lower) / (size - 1)`, requiring size >= 2.
 * This metadata does not give Fourier operators a non-periodic basis.
 *
 * @param j The input root (with a `domain` object) or the domain object.
 * @throws std::invalid_argument on a missing or invalid key, and on the
 *         removed `Lx/Ly/Lz/dx/dy/dz/origin/origo` keys, with the equivalent
 *         block in the message.
 */
template <> [[nodiscard]] inline Domain from_json<Domain>(const json &j) {
  const json &d =
      (j.contains("domain") && j["domain"].is_object()) ? j["domain"] : j;
  detail::reject_legacy_domain_keys(j, d);
  const auto size = detail::read_triple<int>(
      d, "size", "grid points per axis", "three positive integers",
      "[256, 256, 256]", [](int n) { return n > 0; });
  const auto finite = [](double x) { return std::isfinite(x); };
  const auto lower =
      detail::read_triple<double>(d, "lower", "lower limits of the box",
                                  "three finite numbers", "[0.0, 0.0, 0.0]", finite);
  const auto upper = detail::read_triple<double>(
      d, "upper", "upper limits of the box", "three finite numbers",
      "[64.0, 64.0, 64.0]", finite);
  for (int a = 0; a < 3; ++a) {
    if (!(upper[a] > lower[a])) {
      throw std::invalid_argument(format_config_error(
          "domain.upper", "upper limits of the box", "each above its lower limit",
          d.at("upper").dump() + " with lower " + d.at("lower").dump(), {},
          "\"lower\": [0.0, 0.0, 0.0], \"upper\": [64.0, 64.0, 64.0]"));
    }
  }
  const auto periodic = d.contains("periodic")
      ? detail::read_triple<bool>(d, "periodic", "per-axis periodicity",
            "three booleans", "[true, true, false]", [](bool) { return true; })
      : pfc::Bool3{true, true, true};
  for (int a = 0; a < 3; ++a) {
    if (!periodic[a] && size[a] < 2) {
      throw std::invalid_argument("domain.size: non-periodic axes require at least two grid points");
    }
  }
  return pfc::domain::from_bounds({size[0], size[1], size[2]},
                                  {lower[0], lower[1], lower[2]},
                                  {upper[0], upper[1], upper[2]}, periodic);
}

template <> [[nodiscard]] inline Time from_json<Time>(const json &j) {
  // Support both flat and nested structures
  auto t0_val = get_json_value(j, "t0", "timestepping");
  auto t1_val = get_json_value(j, "t1", "timestepping");
  auto dt_val = get_json_value(j, "dt", "timestepping");
  auto saveat_val = get_json_value(j, "saveat", "timestepping");

  if (t0_val.is_null() || t1_val.is_null() || dt_val.is_null() ||
      saveat_val.is_null()) {
    throw std::invalid_argument(
        "Missing required time stepping parameters (t0, t1, dt, saveat)");
  }

  double t0 = t0_val;
  double t1 = t1_val;
  double dt = dt_val;
  double saveat = saveat_val;

  if (!std::isfinite(t0)) {
    throw std::invalid_argument(
        format_config_error("t0", "simulation start time", "finite float",
                            get_json_value_string(j, "t0"), {}, "\"t0\": 0.0"));
  }

  if (!std::isfinite(t1)) {
    throw std::invalid_argument(
        format_config_error("t1", "simulation end time", "finite float",
                            get_json_value_string(j, "t1"), {}, "\"t1\": 100.0"));
  }

  if (t1 <= t0) {
    throw std::invalid_argument(
        format_config_error("t1", "simulation end time", "value greater than t0",
                            get_json_value_string(j, "t1"), {}, "\"t1\": 100.0"));
  }

  if (!std::isfinite(dt)) {
    throw std::invalid_argument(
        format_config_error("dt", "timestep size", "finite float",
                            get_json_value_string(j, "dt"), {}, "\"dt\": 0.01"));
  }
  if (dt <= 0.0) {
    throw std::invalid_argument(
        format_config_error("dt", "timestep size", "positive float",
                            get_json_value_string(j, "dt"), {}, "\"dt\": 0.01"));
  }

  if (!std::isfinite(saveat)) {
    throw std::invalid_argument(format_config_error(
        "saveat", "snapshot output interval", "finite float",
        get_json_value_string(j, "saveat"), {}, "\"saveat\": 1.0"));
  }
  // `saveat <= 0` disables periodic saves (`Time::do_save`); I/O-off
  // campaign inputs use -1. Positive values must still be finite.

  // Parse integrator method (optional, defaults to Euler).
  pfc::sim::steppers::RKIntegratorMethod method =
      pfc::sim::steppers::RKIntegratorMethod::Euler;
  if (j.contains("timestepping") && j["timestepping"].contains("integrator")) {
    const auto &integrator = j["timestepping"]["integrator"];
    if (integrator.contains("method")) {
      try {
        method =
            from_json<pfc::sim::steppers::RKIntegratorMethod>(integrator["method"]);
      } catch (const std::exception &ex) {
        throw std::invalid_argument(ex.what());
      }
    }
  }

  Time time({t0, t1, dt}, saveat, method);
  return time;
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_WORLD_TIME_HPP
