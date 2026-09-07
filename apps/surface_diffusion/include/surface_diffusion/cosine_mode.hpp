// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cosine_mode.hpp
 * @brief One or more Fourier modes on a mean height.
 *
 * JSON `"type": "cosine_mode"`. Single-mode form: `h0`, `amplitude`, `nx`,
 * `ny`, `nz`. Multi-mode form: `h0` plus `modes` array of
 * `{nx, ny, nz?, amplitude}`.
 */

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace surface_diffusion {

struct CosineTerm {
  int nx{0};
  int ny{0};
  int nz{0};
  double amplitude{0.0};
};

class CosineMode : public pfc::FieldModifier {
public:
  void set_h0(double h0) { m_h0 = h0; }
  void set_terms(std::vector<CosineTerm> terms) { m_terms = std::move(terms); }

  [[nodiscard]] double h0() const { return m_h0; }
  [[nodiscard]] const std::vector<CosineTerm> &terms() const { return m_terms; }

  const std::string &get_modifier_name() const override {
    static const std::string k{"CosineMode"};
    return k;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double /*time*/) override {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const double Lx = spacing[0] * static_cast<double>(size[0]);
    const double Ly = spacing[1] * static_cast<double>(size[1]);
    const double Lz = spacing[2] * static_cast<double>(size[2]);
    const double twopi = 2.0 * std::numbers::pi;
    const double h0 = m_h0;
    const auto terms = m_terms;
    pfc::field::apply(field, domain, box, [=](const pfc::Real3 &x) {
      double v = h0;
      for (const auto &t : terms) {
        const double kx = (Lx > 0.0) ? twopi * static_cast<double>(t.nx) / Lx : 0.0;
        const double ky = (Ly > 0.0) ? twopi * static_cast<double>(t.ny) / Ly : 0.0;
        const double kz = (Lz > 0.0) ? twopi * static_cast<double>(t.nz) / Lz : 0.0;
        v += t.amplitude * std::cos(kx * x[0] + ky * x[1] + kz * x[2]);
      }
      return v;
    });
  }

private:
  double m_h0{0.0};
  std::vector<CosineTerm> m_terms{};
};

inline void from_json(const nlohmann::json &j, CosineMode &ic) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "cosine_mode", "Invalid JSON input: missing or incorrect 'type' field.");
  if (!j.contains("h0") || !j["h0"].is_number()) {
    throw std::invalid_argument("cosine_mode: missing or invalid 'h0' field.");
  }
  ic.set_h0(j["h0"].get<double>());
  std::vector<CosineTerm> terms;
  if (j.contains("modes")) {
    if (!j["modes"].is_array() || j["modes"].empty()) {
      throw std::invalid_argument("cosine_mode: 'modes' must be a non-empty array.");
    }
    for (const auto &m : j["modes"]) {
      if (!m.contains("amplitude") || !m["amplitude"].is_number() ||
          !m.contains("nx") || !m["nx"].is_number()) {
        throw std::invalid_argument(
            "cosine_mode: each mode needs integer nx and numeric amplitude.");
      }
      CosineTerm t;
      t.nx = m["nx"].get<int>();
      t.amplitude = m["amplitude"].get<double>();
      if (m.contains("ny") && m["ny"].is_number()) {
        t.ny = m["ny"].get<int>();
      }
      if (m.contains("nz") && m["nz"].is_number()) {
        t.nz = m["nz"].get<int>();
      }
      terms.push_back(t);
    }
  } else {
    if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
      throw std::invalid_argument(
          "cosine_mode: missing or invalid 'amplitude' field.");
    }
    CosineTerm t;
    t.amplitude = j["amplitude"].get<double>();
    if (j.contains("nx") && j["nx"].is_number()) {
      t.nx = j["nx"].get<int>();
    }
    if (j.contains("ny") && j["ny"].is_number()) {
      t.ny = j["ny"].get<int>();
    }
    if (j.contains("nz") && j["nz"].is_number()) {
      t.nz = j["nz"].get<int>();
    }
    terms.push_back(t);
  }
  ic.set_terms(std::move(terms));
}

} // namespace surface_diffusion
