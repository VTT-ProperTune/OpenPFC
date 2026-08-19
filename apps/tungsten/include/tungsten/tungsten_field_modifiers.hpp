// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_field_modifiers.hpp
 * @brief Host-buffer ICs and fixed BC for tungsten ETD sessions.
 *
 * Formulas match Gen-1 `Constant`, `SingleSeed`, and `FixedBC`. Callers pass
 * a host-accessible buffer (host `Field::data()` or a device field's host
 * mirror). Layout is x-fastest, halo 0, local owned box.
 */

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>

namespace tungsten {

inline std::size_t local_idx(const pfc::Box3i &box, int i, int j, int k) {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(box.size[0]) +
         static_cast<std::size_t>(k) * static_cast<std::size_t>(box.size[0]) *
             static_cast<std::size_t>(box.size[1]);
}

inline pfc::Real3 local_coords(const pfc::Domain &domain, const pfc::Box3i &box,
                               int i, int j, int k) {
  const auto &o = pfc::domain::get_origin(domain);
  const auto &s = pfc::domain::get_spacing(domain);
  return {o[0] + static_cast<double>(box.low[0] + i) * s[0],
          o[1] + static_cast<double>(box.low[1] + j) * s[1],
          o[2] + static_cast<double>(box.low[2] + k) * s[2]};
}

inline void fill_constant(double *data, std::size_t n, double n0) {
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = n0;
  }
}

inline void fill_single_seed(const pfc::Domain &domain, const pfc::Box3i &box,
                             double *data, double amp, double rho) {
  const double s = 1.0 / std::sqrt(2.0);
  const double q[6][3] = {{s, s, 0}, {s, 0, s},  {0, s, s},
                          {s, 0, -s}, {s, -s, 0}, {0, s, -s}};
  const double r2 = 64.0 * 64.0;
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const auto c = local_coords(domain, box, i, j, k);
        const double x = c[0];
        const double y = c[1];
        const double z = c[2];
        if (x * x + y * y + z * z >= r2) {
          data[local_idx(box, i, j, k)] = 0.0;
          continue;
        }
        double u = rho;
        for (int qi = 0; qi < 6; ++qi) {
          u += 2.0 * amp * std::cos(q[qi][0] * x + q[qi][1] * y + q[qi][2] * z);
        }
        data[local_idx(box, i, j, k)] = u;
      }
    }
  }
}

inline void apply_ics_from_json(const nlohmann::json &settings,
                                const pfc::Domain &domain, const pfc::Box3i &box,
                                double *data, std::size_t n) {
  if (!settings.contains("initial_conditions") ||
      !settings["initial_conditions"].is_array()) {
    return;
  }
  for (const auto &ic : settings["initial_conditions"]) {
    const std::string type = ic.value("type", std::string{});
    if (type == "constant") {
      fill_constant(data, n, ic.at("n0").get<double>());
    } else if (type == "single_seed") {
      fill_single_seed(domain, box, data, ic.at("amp_eq").get<double>(),
                       ic.at("rho_seed").get<double>());
    } else {
      throw std::invalid_argument(
          "tungsten: unsupported initial condition type '" + type + "'");
    }
  }
}

struct FixedBc {
  double rho_low{};
  double rho_high{};
};

inline std::optional<FixedBc> parse_fixed_bc(const nlohmann::json &settings) {
  if (!settings.contains("boundary_conditions") ||
      !settings["boundary_conditions"].is_array()) {
    return std::nullopt;
  }
  std::optional<FixedBc> out;
  for (const auto &bc : settings["boundary_conditions"]) {
    if (bc.value("type", std::string{}) != "fixed") {
      throw std::invalid_argument(
          "tungsten: unsupported boundary condition type '" +
          bc.value("type", std::string{}) + "'");
    }
    out = FixedBc{bc.at("rho_low").get<double>(),
                  bc.at("rho_high").get<double>()};
  }
  return out;
}

inline void apply_fixed_bc(const pfc::Domain &domain, const pfc::Box3i &box,
                           double *data, const FixedBc &bc) {
  const auto n = pfc::domain::get_size(domain);
  const auto dx = pfc::domain::get_spacing(domain);
  const double xwidth = 20.0;
  const double alpha = 1.0;
  const double xpos = static_cast<double>(n[0]) * dx[0] - xwidth;
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const double x = local_coords(domain, box, i, j, k)[0];
        if (std::abs(x - xpos) < xwidth) {
          const double S = 1.0 / (1.0 + std::exp(-alpha * (x - xpos)));
          data[local_idx(box, i, j, k)] =
              (bc.rho_low * S) + (bc.rho_high * (1.0 - S));
        }
      }
    }
  }
}

} // namespace tungsten
