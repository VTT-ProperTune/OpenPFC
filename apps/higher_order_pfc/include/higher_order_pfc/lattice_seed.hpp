// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file lattice_seed.hpp
 * @brief Controlled single-crystal initial condition: a sum of plane waves.
 *
 * @details
 * `cosine_mode.hpp` seeds exactly one Fourier mode. `#118` also asks for a
 * *controlled single-crystal seed* -- an initial condition that starts the
 * run already at the target symmetry, so a clean run can confirm the kernel
 * merely *preserves* that symmetry rather than having to *select* it out of
 * noise. A single lattice is a small sum of plane waves, not one:
 *
 * \f[
 *   \psi(\mathbf x) = \psi_0 + \frac{A}{M}\sum_{m=1}^{M}
 *     \cos\Bigl(2\pi\bigl(\tfrac{n_x^{(m)}x}{L_x}+\tfrac{n_y^{(m)}y}{L_y}
 *               +\tfrac{n_z^{(m)}z}{L_z}\bigr)\Bigr) .
 * \f]
 *
 * Two modes at \f$90^\circ\f$ (e.g. \f$(n,0,0)\f$ and \f$(0,n,0)\f$) seed a
 * square lattice; three at \f$120^\circ\f$ seed a triangular one. Integer
 * mode counts keep every term exactly periodic, which is what "commensurate
 * box" means in the case JSON comments -- the mode list is chosen so that
 * `2*pi*n/L` lands exactly on the kernel's unstable wavenumber(s) for the box
 * being used; see `apps/higher_order_pfc/README.md`.
 *
 * JSON `"type": "lattice_seed"`, `"modes": [[nx,ny,nz], ...]`.
 */

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace higher_order_pfc {

class LatticeSeed : public pfc::FieldModifier {
public:
  struct Mode {
    int nx{0}, ny{0}, nz{0};
  };

  void set_psi0(double psi0) { m_psi0 = psi0; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_modes(std::vector<Mode> modes) { m_modes = std::move(modes); }

  [[nodiscard]] double psi0() const { return m_psi0; }
  [[nodiscard]] double amplitude() const { return m_amplitude; }
  [[nodiscard]] const std::vector<Mode> &modes() const { return m_modes; }

  const std::string &get_modifier_name() const override {
    static const std::string k{"LatticeSeed"};
    return k;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double /*time*/) override {
    if (m_modes.empty())
      throw std::invalid_argument("lattice_seed: need at least one mode");
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const double Lx = spacing[0] * static_cast<double>(size[0]);
    const double Ly = spacing[1] * static_cast<double>(size[1]);
    const double Lz = spacing[2] * static_cast<double>(size[2]);
    const double twopi = 2.0 * std::numbers::pi;
    std::vector<std::array<double, 3>> k;
    k.reserve(m_modes.size());
    for (const auto &m : m_modes) {
      k.push_back({(Lx > 0.0) ? twopi * static_cast<double>(m.nx) / Lx : 0.0,
                   (Ly > 0.0) ? twopi * static_cast<double>(m.ny) / Ly : 0.0,
                   (Lz > 0.0) ? twopi * static_cast<double>(m.nz) / Lz : 0.0});
    }
    const double psi0 = m_psi0;
    const double amp = m_amplitude / static_cast<double>(k.size());
    pfc::field::apply(field, domain, box, [=](const pfc::Real3 &x) {
      double s = 0.0;
      for (const auto &kv : k) s += std::cos(kv[0] * x[0] + kv[1] * x[1] + kv[2] * x[2]);
      return psi0 + amp * s;
    });
  }

private:
  double m_psi0{0.0};
  double m_amplitude{0.1};
  std::vector<Mode> m_modes{};
};

inline void from_json(const nlohmann::json &j, LatticeSeed &ic) {
  if (j.value("type", "") != "lattice_seed")
    throw std::invalid_argument("lattice_seed: incorrect or missing 'type' field.");
  if (!j.contains("psi0") || !j["psi0"].is_number())
    throw std::invalid_argument("lattice_seed: missing or invalid 'psi0' field.");
  if (!j.contains("amplitude") || !j["amplitude"].is_number())
    throw std::invalid_argument("lattice_seed: missing or invalid 'amplitude' field.");
  if (!j.contains("modes") || !j["modes"].is_array() || j["modes"].empty())
    throw std::invalid_argument("lattice_seed: 'modes' must be a nonempty array "
                                "of [nx, ny, nz] triples.");
  ic.set_psi0(j["psi0"].get<double>());
  ic.set_amplitude(j["amplitude"].get<double>());
  std::vector<LatticeSeed::Mode> modes;
  modes.reserve(j["modes"].size());
  for (const auto &mode : j["modes"]) {
    if (!mode.is_array() || mode.size() != 3)
      throw std::invalid_argument("lattice_seed: each mode must be [nx, ny, nz].");
    modes.push_back({mode.at(0).get<int>(), mode.at(1).get<int>(), mode.at(2).get<int>()});
  }
  ic.set_modes(std::move(modes));
}

} // namespace higher_order_pfc
