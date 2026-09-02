// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_field_modifiers.hpp
 * @brief Host-buffer ICs and fixed BC for aluminum ETD sessions.
 *
 * Constant IC and `FixedBC` match Gen-1. Callers pass a host-accessible
 * buffer (host `Field::data()`). Layout is x-fastest, halo 0.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "SeedFCC.hpp"

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>

namespace aluminum {

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

/** Host-buffer SeedGridFCC; RNG and SeedFCC match Gen-1. Does not write outside
 * seeds. */
inline void fill_seed_grid_fcc(const pfc::Domain &domain, const pfc::Box3i &box,
                               double *data, int Ny, int Nz, double X0,
                               double radius, double amplitude, double rho,
                               double rseed) {
  const auto size = pfc::domain::get_size(domain);
  const auto spacing = pfc::domain::get_spacing(domain);
  const double Dy =
      spacing[1] * static_cast<double>(size[1]) / static_cast<double>(Ny);
  const double Dz =
      spacing[2] * static_cast<double>(size[2]) / static_cast<double>(Nz);
  const double Y0 = Dy / 2.0;
  const double Z0 = Dz / 2.0;
  std::mt19937_64 re(static_cast<std::mt19937_64::result_type>(rseed));
  std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
  std::uniform_real_distribution<double> rr(0.0, 2.0 * std::numbers::pi);
  std::vector<SeedFCC> seeds;
  seeds.reserve(static_cast<std::size_t>(Ny * Nz));
  for (int j = 0; j < Ny; ++j) {
    for (int k = 0; k < Nz; ++k) {
      const std::array<double, 3> location = {
          X0 + rt(re), Y0 + Dy * static_cast<double>(j) + rt(re),
          Z0 + Dz * static_cast<double>(k) + rt(re)};
      const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
      seeds.emplace_back(location, orientation, radius, rho, amplitude);
    }
  }
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const auto c = local_coords(domain, box, i, j, k);
        const std::array<double, 3> X = {c[0], c[1], c[2]};
        for (const auto &seed : seeds) {
          if (seed.is_inside(X)) {
            data[local_idx(box, i, j, k)] = seed.get_value(X);
            break;
          }
        }
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
    } else if (type == "seed_grid_fcc") {
      const double rseed = ic.contains("rseed") && ic["rseed"].is_number()
                               ? ic["rseed"].get<double>()
                               : 0.0;
      fill_seed_grid_fcc(
          domain, box, data, ic.at("Ny").get<int>(), ic.at("Nz").get<int>(),
          ic.at("X0").get<double>(), ic.at("radius").get<double>(),
          ic.at("amplitude").get<double>(), ic.at("rho").get<double>(), rseed);
    } else {
      throw std::invalid_argument("aluminum: unsupported initial condition type '" +
                                  type + "'");
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
      throw std::invalid_argument("aluminum: unsupported boundary condition type '" +
                                  bc.value("type", std::string{}) + "'");
    }
    out = FixedBc{bc.at("rho_low").get<double>(), bc.at("rho_high").get<double>()};
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

} // namespace aluminum
