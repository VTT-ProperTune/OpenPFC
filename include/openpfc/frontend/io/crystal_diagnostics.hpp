// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file crystal_diagnostics.hpp
 * @brief Write crystal-front and orientation observables to a DiagnosticsSeries.
 *
 * Column order is `front_position`, `front_velocity`, `front_valid`,
 * `wave_x`, `wave_y`, `wave_z`, `orientation_angle`, `orientation_valid`,
 * `voting_cells`, then `grain_0` ... `grain_{n-1}`. Invalid samples are
 * quiet NaN in the numeric columns and 0 in the flag columns. The series
 * has to be constructed with `crystal_diagnostic_columns(candidate count)`.
 */

#include <openpfc/frontend/io/diagnostics_series.hpp>
#include <openpfc/kernel/field/crystal_observables.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc::io {

[[nodiscard]] inline std::vector<std::string>
crystal_diagnostic_columns(std::size_t grain_count) {
  std::vector<std::string> names = {
      "front_position", "front_velocity", "front_valid",       "wave_x",
      "wave_y",         "wave_z",         "orientation_angle", "orientation_valid",
      "voting_cells"};
  names.reserve(names.size() + grain_count);
  for (std::size_t g = 0; g < grain_count; ++g) {
    names.push_back("grain_" + std::to_string(g));
  }
  return names;
}

/// One collective front, one orientation, and one grain vote, then one CSV row.
inline void record_crystal_diagnostics(
    DiagnosticsSeries &series, int step, double time,
    const pfc::observables::OwnedDensity &density, MPI_Comm comm,
    const pfc::observables::PlanarFrontOptions &front,
    const pfc::observables::FrontSample *earlier, double earlier_time,
    pfc::observables::CrystalSymmetry symmetry, pfc::observables::IndexWindow window,
    std::span<const std::array<double, 3>> candidates) {
  const pfc::observables::FrontSample sample =
      pfc::observables::planar_front_position(density, comm, front);
  pfc::observables::FrontVelocity speed;
  if (earlier != nullptr) {
    speed = pfc::observables::front_velocity(*earlier, earlier_time, sample, time);
  }
  const pfc::observables::OrientationSample orientation =
      pfc::observables::region_orientation(density, comm, window, symmetry);
  const pfc::observables::GrainStatistics grains =
      pfc::observables::label_grains(density, comm, window, symmetry, candidates);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> row = {
      sample.validity == pfc::observables::FrontValidity::valid ? sample.position
                                                                : nan,
      speed.valid ? speed.velocity : nan,
      sample.validity == pfc::observables::FrontValidity::valid ? 1.0 : 0.0,
      orientation.valid ? orientation.wavevector[0] : nan,
      orientation.valid ? orientation.wavevector[1] : nan,
      orientation.valid ? orientation.wavevector[2] : nan,
      orientation.valid ? orientation.angle_xy : nan,
      orientation.valid ? 1.0 : 0.0,
      static_cast<double>(grains.voting_cells)};
  if (grains.fraction.size() != candidates.size()) {
    throw std::invalid_argument(
        "crystal diagnostics: grain fractions do not match the candidates");
  }
  for (double fraction : grains.fraction) row.push_back(fraction);
  series.write(step, time, row);
}

} // namespace pfc::io
