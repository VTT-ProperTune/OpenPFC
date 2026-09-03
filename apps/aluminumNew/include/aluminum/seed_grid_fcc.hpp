// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file seed_grid_fcc.hpp
 * @brief `seed_grid_fcc` initial condition: a Ny×Nz grid of FCC seeds at X0.
 *
 * @details
 * A catalog `FieldModifier` (JSON `"type": "seed_grid_fcc"`) so the aluminum
 * binaries and tests use the same IC path as every other 0.2 session. RNG and
 * `SeedFCC` geometry match Gen-1 `SeedGridFCC` (pinned by the aluminum 5-step
 * CPU checksum). Cells outside every seed are left untouched, so it composes
 * with a preceding `constant` IC.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include <SeedFCC.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace aluminum {

class SeedGridFCC : public pfc::FieldModifier {
public:
  void set_Ny(int Ny) { m_Ny = Ny; }
  void set_Nz(int Nz) { m_Nz = Nz; }
  void set_X0(double X0) { m_X0 = X0; }
  void set_radius(double radius) { m_radius = radius; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_density(double rho) { m_rho = rho; }
  void set_rseed(double rseed) { m_rseed = rseed; }

  const std::string &get_modifier_name() const override {
    static const std::string k{"SeedGridFCC"};
    return k;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double /*time*/) override {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const auto origin = pfc::domain::get_origin(domain);
    const double Dy =
        spacing[1] * static_cast<double>(size[1]) / static_cast<double>(m_Ny);
    const double Dz =
        spacing[2] * static_cast<double>(size[2]) / static_cast<double>(m_Nz);
    const double Y0 = Dy / 2.0;
    const double Z0 = Dz / 2.0;
    std::mt19937_64 re(static_cast<std::mt19937_64::result_type>(m_rseed));
    std::uniform_real_distribution<double> rt(-0.2 * m_radius, 0.2 * m_radius);
    std::uniform_real_distribution<double> rr(0.0, 2.0 * std::numbers::pi);
    std::vector<SeedFCC> seeds;
    seeds.reserve(static_cast<std::size_t>(m_Ny * m_Nz));
    for (int j = 0; j < m_Ny; ++j) {
      for (int k = 0; k < m_Nz; ++k) {
        const std::array<double, 3> location = {
            m_X0 + rt(re), Y0 + Dy * static_cast<double>(j) + rt(re),
            Z0 + Dz * static_cast<double>(k) + rt(re)};
        const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
        seeds.emplace_back(location, orientation, m_radius, m_rho, m_amplitude);
      }
    }
    const int nx = box.size[0];
    const int ny = box.size[1];
    const int nz = box.size[2];
    double *data = field.data();
    std::size_t idx = 0;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const std::array<double, 3> X = {
              origin[0] + static_cast<double>(box.low[0] + i) * spacing[0],
              origin[1] + static_cast<double>(box.low[1] + j) * spacing[1],
              origin[2] + static_cast<double>(box.low[2] + k) * spacing[2]};
          for (const auto &seed : seeds) {
            if (seed.is_inside(X)) {
              data[idx] = seed.get_value(X);
              break;
            }
          }
          ++idx;
        }
      }
    }
  }

private:
  int m_Ny{1};
  int m_Nz{1};
  double m_X0{0.0};
  double m_radius{1.0};
  double m_amplitude{0.0};
  double m_rho{0.0};
  double m_rseed{0.0};
};

inline void from_json(const nlohmann::json &j, SeedGridFCC &ic) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "seed_grid_fcc",
      "Invalid JSON input: missing or incorrect 'type' field.");
  for (const char *key : {"Ny", "Nz", "X0", "radius", "amplitude", "rho"}) {
    if (!j.contains(key) || !j[key].is_number()) {
      throw std::invalid_argument(std::string("seed_grid_fcc: missing or invalid '") +
                                  key + "' field.");
    }
  }
  ic.set_Ny(j["Ny"].get<int>());
  ic.set_Nz(j["Nz"].get<int>());
  ic.set_X0(j["X0"].get<double>());
  ic.set_radius(j["radius"].get<double>());
  ic.set_amplitude(j["amplitude"].get<double>());
  ic.set_density(j["rho"].get<double>());
  if (j.contains("rseed") && j["rseed"].is_number()) {
    ic.set_rseed(j["rseed"].get<double>());
  }
}

} // namespace aluminum
