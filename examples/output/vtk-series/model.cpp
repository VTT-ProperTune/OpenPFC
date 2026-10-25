// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <openpfc/frontend/io/snapshot_series.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto domain = output_lesson::domain(argc, argv);
    const auto partition = pfc::decomposition::create(domain, comm.size());
    // One padded storage layer demonstrates that snapshots pack owned values.
    auto field = pfc::data::field_from_subdomain<double>(partition, comm.rank(), 1);
    std::ranges::fill(field.vec(), -9999);
    // start snippet series
    pfc::io::SnapshotSeries series(
        domain, field.box(),
        {.directory = "results", .prefix = "lesson", .comm = comm});
    series.add_field("density", field, "results/density_%04d.vti",
                     pfc::io::SnapshotFormat::Vtk);
    series.add_field("density_raw", field);
    for (int frame = 0; frame < 3; ++frame) {
      const double t = .25 * frame;
      field.apply([&](double x, double y, double z) {
        return output_lesson::value(x, y, z, t);
      });
      series.write(4 * frame, t);
    }
    series.close();
    // end snippet series
    if (comm.rank() == 0) std::cout << "Wrote " << series.frames() << " frames\n";
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
