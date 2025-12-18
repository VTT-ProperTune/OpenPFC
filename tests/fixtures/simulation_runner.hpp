// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <openpfc/model.hpp>
#include <vector>

namespace pfc {
namespace test {

struct SimulationRunner {
  Model &model;
  explicit SimulationRunner(Model &m) : model(m) {}

  void run_steps(int steps, double t = 0.0) {
    for (int s = 0; s < steps; ++s) model.step(t);
  }
};

inline double compute_mean(const std::vector<double> &field) {
  if (field.empty()) return 0.0;
  double sum = 0.0;
  for (double v : field) sum += v;
  return sum / static_cast<double>(field.size());
}

inline double compute_l2(const std::vector<double> &field) {
  double l2 = 0.0;
  for (double v : field) l2 += v * v;
  return l2;
}

inline double compute_max(const std::vector<double> &field) {
  double m = field.empty() ? 0.0 : field[0];
  for (double v : field)
    if (v > m) m = v;
  return m;
}

} // namespace test
} // namespace pfc
