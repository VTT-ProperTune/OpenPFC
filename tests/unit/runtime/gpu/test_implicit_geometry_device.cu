// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// Device parity for implicit classification and cut-cell measures.
// Built by nvcc/hipcc together with implicit_geometry.cu.

#include <openpfc/kernel/geometry/cut_cell.hpp>
#include <openpfc/runtime/gpu/implicit_geometry.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
  using namespace pfc::geometry;
  pfc::Domain domain;
  domain.size = {8, 8, 8};
  domain.spacing = {0.125, 0.15, 0.1};
  domain.origin = {-0.2, 0.3, 1.0};
  domain.periodic = {false, false, false};
  const CellLayout layout = layout_from_domain(domain);
  const auto rotation = rotation_about({0.3, 0.5, 0.8}, 0.9);
  const Implicit shape =
      make_ellipsoid({0.3, 0.8, 1.4}, {0.22, 0.14, 0.1}, rotation);
  Policy policy;
  policy.max_depth = 3;
  std::vector<std::uint8_t> mask;
  std::vector<double> inside, outside, area;
  DeviceProfile profile;
  try {
    evaluate_on_device(layout, shape, policy, mask, inside, outside, area, profile);
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  const CutList host = classify_layout(shape, layout, policy);
  int host_cut = 0;
  double host_inside = 0;
  double device_inside = 0;
  for (int id = 0; id < layout.size(); ++id) {
    const int i = id % layout.count[0];
    const int jk = id / layout.count[0];
    const int j = jk % layout.count[1];
    const int k = jk / layout.count[1];
    const Verdict verdict = classify_box(shape, layout.box(i, j, k), policy);
    const unsigned char expect = verdict.kind == CellKind::Interior   ? 0
                                 : verdict.kind == CellKind::Exterior ? 1
                                                                      : 2;
    if (mask[static_cast<std::size_t>(id)] != expect) {
      std::cerr << "class mismatch at " << id << "\n";
      return 1;
    }
    if (expect == 2) {
      const CutMeasure measured =
          measure_box(shape, layout.box(i, j, k), layout.dimension(), policy);
      if (host_cut >= static_cast<int>(inside.size())) return 1;
      const double scale = std::max(1.0, measured.inside_volume);
      if (std::abs(inside[static_cast<std::size_t>(host_cut)] -
                   measured.inside_volume) > 1e-10 * scale ||
          std::abs(outside[static_cast<std::size_t>(host_cut)] -
                   measured.outside_volume) > 1e-10 * scale) {
        std::cerr << "measure mismatch at cut " << host_cut << "\n";
        return 1;
      }
      host_inside += measured.inside_volume;
      device_inside += inside[static_cast<std::size_t>(host_cut)];
      ++host_cut;
    }
  }
  if (host_cut != profile.cut_cells ||
      host_cut != static_cast<int>(host.index.size())) {
    std::cerr << "cut count " << host_cut << " device " << profile.cut_cells
              << " host list " << host.index.size() << "\n";
    return 1;
  }
  std::cout << "cells " << profile.cells << " cut " << profile.cut_cells
            << " classify_s " << profile.classify_seconds << " measure_s "
            << profile.measure_seconds << " device_bytes " << profile.device_bytes
            << " to_device " << profile.bytes_to_device << " to_host "
            << profile.bytes_to_host << " inside " << device_inside << "\n";
  if (!(profile.device_bytes > 0) || !(host_inside > 0)) return 1;
  return 0;
}
