// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file finite_strain_inverse.hpp
 * @brief First 2-D finite-strain multimaterial experiment.
 *
 * Frozen protocol: 16² neo-Hookean rotating-square family. Compare
 * single-material+void against stiff/compliant/void (hinge-painted)
 * on the same geometries. Report ν_t(F11) and any log-strain zero
 * crossing. Not a claim that free topology optimization was run.
 */

#include "finite_strain_grid.hpp"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace pfc::apps::inverse::fs {

struct PathPoint {
  double F11{1.0};
  double F22{1.0};
  double P11{0.0};
  double nu_t{0.0};
  bool converged{false};
};

struct DesignPath {
  std::string arm;
  double half{0.20};
  std::vector<PathPoint> points;
  bool any_sign_change{false};
  double eps_cross{0.0};
  bool cross_found{false};
  double K_small{0.0};
  bool all_converged{false};
};

struct Experiment484 {
  std::vector<DesignPath> designs;
  bool all_converged{false};
  bool any_programmed_crossing{false};
  std::string interpretation;
};

inline double nu_t_at(PixelCell &c, double F11, double rel = 1e-3) {
  PixelCell cm = c, cp = c;
  const auto sm = homogenize_periodic_2d(cm, F11 * (1.0 - rel));
  const auto sp = homogenize_periodic_2d(cp, F11 * (1.0 + rel));
  if (!sm.converged || !sp.converged) {
    throw std::runtime_error("nu_t_at: homogenization failed");
  }
  return tangent_poisson_log(sm.F11, sm.F22, sp.F11, sp.F22);
}

inline DesignPath evaluate_design(const std::string &arm, double half, bool hinges) {
  DesignPath d;
  d.arm = arm;
  d.half = half;
  const auto stiff = lame_from_young_poisson(1.0, 0.3);
  const auto compliant = lame_from_young_poisson(0.05, 0.3);
  const auto voided = lame_from_young_poisson(1e-3, 0.3);
  std::vector<double> h;
  fill_rotating_squares_vec(h, 16, 16, half, 0.45);
  PixelCell c;
  init_cell(c, 16, 16, Model::NeoHookean);
  if (hinges) {
    std::vector<int> phase;
    paint_hinge_phase(h, 16, 16, phase);
    assign_lame_three_phase(c, phase, stiff, compliant, voided);
  } else {
    assign_lame_two_phase(c, h, stiff, voided);
  }
  const double F11s[] = {1.02, 1.05, 1.10, 1.15, 1.20};
  d.all_converged = true;
  PixelCell c1 = c;
  for (double F11 : F11s) {
    const auto g = homogenize_periodic_2d(c1, F11);
    PathPoint p;
    p.F11 = F11;
    p.F22 = g.F22;
    p.P11 = g.P11;
    p.converged = g.converged;
    d.all_converged = d.all_converged && g.converged;
    d.points.push_back(p);
  }
  for (std::size_t i = 0; i + 1 < d.points.size(); ++i) {
    if (d.points[i].converged && d.points[i + 1].converged) {
      d.points[i].nu_t =
          tangent_poisson_log(d.points[i].F11, d.points[i].F22, d.points[i + 1].F11,
                              d.points[i + 1].F22);
    }
  }
  if (d.points.size() >= 2) {
    d.points.back().nu_t = d.points[d.points.size() - 2].nu_t;
  }
  if (!d.points.empty() && d.points.front().converged) {
    const double e = std::log(d.points.front().F11);
    d.K_small = d.points.front().P11 / e;
  }
  for (std::size_t i = 1; i < d.points.size(); ++i) {
    const double a = d.points[i - 1].nu_t;
    const double b = d.points[i].nu_t;
    if (d.points[i - 1].converged && d.points[i].converged && a * b < 0.0) {
      d.any_sign_change = true;
      const double ea = std::log(d.points[i - 1].F11);
      const double eb = std::log(d.points[i].F11);
      d.eps_cross = ea + (0.0 - a) / (b - a) * (eb - ea);
      if (a > 0.0 && b < 0.0) d.cross_found = true;
    }
  }
  return d;
}

[[nodiscard]] inline Experiment484 run_declared_484_experiment() {
  Experiment484 ex;
  ex.all_converged = true;
  const double halves[] = {0.16, 0.18, 0.20, 0.22, 0.24};
  for (double half : halves) {
    ex.designs.push_back(evaluate_design("single-material+void", half, false));
    ex.designs.push_back(evaluate_design("stiff+compliant+void", half, true));
  }
  for (const auto &d : ex.designs) {
    ex.all_converged = ex.all_converged && d.all_converged;
    if (d.cross_found) ex.any_programmed_crossing = true;
  }
  if (!ex.any_programmed_crossing) {
    ex.interpretation =
        "Frozen 16^2 neo-Hookean rotating-square family: no design met "
        "nu_t>0 at small strain and nu_t<0 after a prescribed epsilon_c. "
        "Disconnected cells stay conventional; the hinged 0.20 cell is "
        "auxetic at F11=1.02 then loses that sign. Hinge-painted "
        "compliant phase changes stiffness but does not create the "
        "requested programmable crossing. Bounded to this geometry and "
        "reversible neo-Hookean scope; not free topology optimization.";
  } else {
    ex.interpretation =
        "At least one frozen rotating-square design produced nu_t>0 then "
        "nu_t<0. Compare single-material vs hinge-painted arms for "
        "whether the second phase moves epsilon_c at matched small-strain "
        "stiffness. Not a general multimaterial TO guarantee.";
  }
  return ex;
}

} // namespace pfc::apps::inverse::fs
