// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <string>

#include <inverse_homogenization/homogenization.hpp>
#include <inverse_homogenization/inverse_convergence.hpp>
#include <nlohmann/json.hpp>

namespace pfc::apps::inverse {

/// Diagnostics of an already computed, unpenalized endpoint tensor. This does
/// not declare inverse convergence: MAX_STEPS remains a nonconverged endpoint.
inline nlohmann::json material_report(const std::string &field, int accepted_step,
                                      TerminationReason reason,
                                      const std::array<int, 3> &grid, double dx,
                                      const HomogenizationResult &result,
                                      const Voigt6 &target) {
  using Json = nlohmann::json;
  auto matrix = [](const Voigt6 &C) {
    Json rows = Json::array();
    for (int i = 0; i < 6; ++i) {
      Json row = Json::array();
      for (int j = 0; j < 6; ++j)
        row.push_back(std::isfinite(C(i, j)) ? Json(C(i, j)) : Json(nullptr));
      rows.push_back(row);
    }
    return rows;
  };
  bool finite = true;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      finite = finite && std::isfinite(result.stiffness(i, j));
  Json solves = Json::array();
  bool solve_valid = result.n_loads == 6 && result.all_converged();
  for (const auto &report : result.reports) {
    solve_valid =
        solve_valid && std::isfinite(report.residual) && report.residual >= 0;
    solves.push_back(
        {{"converged", report.converged},
         {"iterations", report.iterations},
         {"residual",
          std::isfinite(report.residual) ? Json(report.residual) : Json(nullptr)}});
  }
  Json out = {{"schema_version", 1},
              {"field", field},
              {"accepted_step", accepted_step},
              {"termination", termination_name(reason)},
              {"inverse_converged", reason == TerminationReason::Converged},
              {"grid", grid},
              {"spacing", dx},
              {"interpolation", "unpenalized linear material interpolation"},
              {"voigt_order", {"xx", "yy", "zz", "yz", "xz", "xy"}},
              {"strain_convention", "engineering shear gamma=2*epsilon"},
              {"stiffness_raw", matrix(result.stiffness)},
              {"finite_stiffness", finite},
              {"volume_fraction", result.volume_fraction},
              {"load_count", result.n_loads},
              {"elasticity_solves", solves},
              {"elasticity_converged", result.all_converged()},
              {"diagnostics_valid", false},
              {"diagnostics", nullptr},
              {"compliance", nullptr},
              {"poisson", nullptr}};
  if (!finite) return out;
  const auto d = diagnose_stiffness(result.stiffness, &target);
  out["stiffness_symmetric"] = matrix(d.C);
  out["diagnostics"] = {{"spd", d.spd},
                        {"invertible", d.invertible},
                        {"minimum_eigenvalue", d.min_eig},
                        {"relative_target_frobenius", d.rel_frobenius},
                        {"spread_C11", d.spread_C11},
                        {"spread_C12", d.spread_C12},
                        {"spread_C44", d.spread_C44}};
  bool valid = solve_valid && d.spd && d.invertible && std::isfinite(d.min_eig) &&
               std::isfinite(d.rel_frobenius) &&
               std::isfinite(result.volume_fraction) &&
               result.volume_fraction >= 0 && result.volume_fraction <= 1;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) valid = valid && std::isfinite(d.S(i, j));
  for (int i = 0; i < 3; ++i) valid = valid && d.S(i, i) > 1.e-30;
  out["diagnostics_valid"] = valid;
  if (valid) {
    out["compliance"] = matrix(d.S);
    out["poisson"] = {{"xy", d.nu_xy}, {"xz", d.nu_xz}, {"yx", d.nu_yx},
                      {"yz", d.nu_yz}, {"zx", d.nu_zx}, {"zy", d.nu_zy}};
  }
  return out;
}

} // namespace pfc::apps::inverse
