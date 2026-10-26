// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int failed = 0;
  try {
    if (argc != 3)
      throw std::invalid_argument("usage: vtk_consumer output-directory split");
    const int split = std::stoi(argv[2]);
    std::array<int, 3> dims{1, 1, 1};
    if (split >= 0 && split < 3) {
      dims[split] = ranks;
    } else if (ranks == 4 && split >= 3 && split < 6) {
      constexpr std::array<std::array<int, 2>, 3> axes{{{0, 1}, {0, 2}, {1, 2}}};
      dims[axes[split - 3][0]] = dims[axes[split - 3][1]] = 2;
    } else {
      throw std::invalid_argument("unsupported decomposition");
    }
    constexpr std::array<int, 3> size{11, 9, 7};
    const auto domain = pfc::domain::create(pfc::GridSize(size),
                                            pfc::PhysicalOrigin({0.3, 0.125, 0.55}),
                                            pfc::GridSpacing({0.6, 0.25, 1.1}));
    std::array<int, 3> low{}, high{};
    int remainder = rank;
    for (int d = 0; d < 3; ++d) {
      const int coordinate = remainder % dims[d];
      remainder /= dims[d];
      low[d] = size[d] * coordinate / dims[d];
      high[d] = size[d] * (coordinate + 1) / dims[d] - 1;
    }
    pfc::data::Field<double> field(domain, pfc::Box3i::from_bounds(low, high), 0);
    field.for_each_owned([&](int i, int j, int k) {
      const auto x = field.coords(i, j, k);
      field(i, j, k) = x[0] + 0.2 * x[1] - 0.1 * x[2];
    });
    const std::filesystem::path output(argv[1]);
    std::filesystem::create_directories(output);
    pfc::VTKWriter writer((output / "field_%04d.vti").string());
    pfc::apply_writer_domain(writer, field);
    writer.set_field_name("plane");
    auto status = writer.write(1, field.view());
    int count = 0;
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    if (count != static_cast<int>(field.view().size()))
      throw std::runtime_error("writer status must count owned samples");
  } catch (const std::exception &error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    failed = 1;
  }
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return any_failed;
}
