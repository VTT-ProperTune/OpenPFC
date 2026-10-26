// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/ui/simulation_wiring_writers.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    MPI_Finalize();
    return 1;
  }

  // Shared unique relative directory, absent when rank one enters wiring.
  long long id = std::chrono::steady_clock::now().time_since_epoch().count();
  MPI_Bcast(&id, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  const auto root = std::filesystem::path("writer-directory-" + std::to_string(id));
  const auto output = root / "nested" / ("rank-" + std::to_string(rank) + ".bin");
  int failed = 0;
  int ready = 1;
  if (rank == 0) {
    MPI_Recv(&ready, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  try {
    const nlohmann::json settings = {
        {"fields", {{{"name", "psi"}, {"data", output.string()}}}}};
    const pfc::ui::JsonWiringContext ctx{MPI_COMM_WORLD, rank, rank == 0};
    const auto writers = pfc::ui::parse_result_writers_from_json(
        settings, ctx, pfc::ui::make_builtin_results_writer_catalog());
    std::ofstream file(output);
    file << rank << '\n';
    file.close();
    failed = writers.size() != 1 || !file || !std::filesystem::exists(output);
    // Repeated preparation of an existing directory is harmless.
    failed |= pfc::ui::ensure_results_parent_dir_for_writer(output.string(), rank);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    failed = 1;
  }
  if (rank == 1) {
    MPI_Send(&ready, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  int global_failed = 0;
  MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) std::filesystem::remove_all(root);
  MPI_Finalize();
  return global_failed;
}
