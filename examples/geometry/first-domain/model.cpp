// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <filesystem>
#include <fstream>
#include <iostream>
#include <openpfc/kernel/data/domain.hpp>

int main() {
  try {
    // start snippet domain
    const auto domain = pfc::domain::create(
        pfc::GridSize{{16, 8, 1}}, pfc::PhysicalOrigin{{-1, -2, 0}},
        pfc::GridSpacing{{.125, .5, 1}}, pfc::Bool3{true, true, false});
    const pfc::Int3 index{3, 5, 0};
    const auto x = pfc::domain::to_coords(domain, index);
    const auto restored = pfc::domain::to_indices(domain, x);
    if (restored != index) throw std::runtime_error("Coordinate round trip failed");
    // end snippet domain
    std::filesystem::create_directories("results");
    std::ofstream output("results/domain.csv");
    output << "axis,samples,origin,spacing,last_sample,upper,periodic\n";
    const auto size = pfc::domain::get_size(domain);
    const auto origin = pfc::domain::get_origin(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    for (int axis = 0; axis < 3; ++axis)
      output << axis << ',' << size[axis] << ',' << origin[axis] << ','
             << spacing[axis] << ','
             << origin[axis] + (size[axis] - 1) * spacing[axis] << ','
             << pfc::domain::upper(domain)[axis] << ','
             << pfc::domain::is_periodic(domain, axis) << '\n';
    if (!output) throw std::runtime_error("Cannot write domain CSV");
    std::cout << "Index (3,5,0) has coordinates (" << x[0] << ',' << x[1] << ','
              << x[2] << ")\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
