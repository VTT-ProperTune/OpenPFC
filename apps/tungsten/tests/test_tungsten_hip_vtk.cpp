// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_hip_vtk.cpp
 * @brief Test Tungsten HIP model with initial/boundary conditions and VTK output
 *
 * This program runs a tungsten simulation with:
 * - HIP/GPU acceleration (AMD ROCm)
 * - Initial conditions (constant + single seed)
 * - Boundary conditions (fixed)
 * - VTK output for visualization
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "This test requires HIP support. Enable with -DOpenPFC_ENABLE_HIP=ON"
#endif

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/ui/simulation_wiring_conditions.hpp>
#include <openpfc/frontend/ui/ui.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <tungsten/common/tungsten_input.hpp>
#include <tungsten/hip/tungsten_model.hpp>

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  pfc::MPI_Worker worker(argc, argv);
  int rank = worker.get_rank();
  int num_ranks = worker.get_num_ranks();
  bool rank0 = (rank == 0);

  std::string config_file = "tungsten_single_seed_256_hip.json";
  if (argc > 1) {
    config_file = argv[1];
  }

  json settings;
  std::ifstream input_file(config_file);
  if (!input_file) {
    if (rank0) {
      std::cerr << "Error: Cannot open configuration file: " << config_file
                << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  input_file >> settings;

  if (rank0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Tungsten HIP Test with VTK Output" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration: " << config_file << std::endl;
    std::cout << "MPI ranks: " << num_ranks << std::endl;
    std::cout << "========================================" << std::endl;
  }

  pfc::World world(pfc::ui::from_json<pfc::World>(settings));
  if (rank0) {
    std::cout << "World: " << world << std::endl;
  }

  auto decomp = pfc::decomposition::create(world, num_ranks);
  heffte::plan_options options = heffte::default_options<heffte::backend::rocfft>();
  if (settings.contains("plan_options")) {
    options = pfc::ui::from_json<heffte::plan_options>(settings["plan_options"]);
  }
  auto fft_layout = pfc::fft::layout::create(decomp, 0);

  auto dummy_fft = pfc::fft::create(fft_layout, rank, options);
  TungstenHIP<double> model(dummy_fft, world);

  if (settings.contains("model") && settings["model"].contains("params")) {
    from_json(settings["model"]["params"], model);
  }

  pfc::Time time(pfc::ui::from_json<pfc::Time>(settings));

  if (rank0) std::cout << "Initializing model..." << std::endl;
  model.initialize(time.get_dt());

  pfc::Simulator simulator(model, time);

  if (rank0) std::cout << "Adding initial conditions..." << std::endl;
  pfc::ui::add_initial_conditions_from_json(simulator, settings,
                                            simulator.mpi_comm(), rank, rank0);

  if (rank0) std::cout << "Adding boundary conditions..." << std::endl;
  pfc::ui::add_boundary_conditions_from_json(simulator, settings,
                                             simulator.mpi_comm(), rank, rank0);

  if (rank0) std::cout << "Adding VTK writer..." << std::endl;
  if (settings.contains("fields") && settings["saveat"] > 0) {
    for (const auto &field : settings["fields"]) {
      std::string name = field["name"];
      std::string data = field["data"];

      if (data.find(".vti") != std::string::npos ||
          data.find(".vtk") != std::string::npos) {
        if (rank0) {
          std::cout << "Creating VTK writer for field '" << name << "' -> " << data
                    << std::endl;
          std::filesystem::path results_dir(data);
          if (results_dir.has_filename()) results_dir = results_dir.parent_path();
          if (!std::filesystem::exists(results_dir)) {
            std::filesystem::create_directories(results_dir);
          }
        }

        auto vtk_writer = std::make_unique<pfc::VTKWriter>(data);

        auto &hip_fft = model.get_hip_fft();
        auto inbox = pfc::fft::get_inbox(hip_fft);

        auto [Lx, Ly, Lz] = pfc::world::get_size(world);
        std::array<int, 3> global_size = {Lx, Ly, Lz};
        std::array<int, 3> local_size = {inbox.high[0] - inbox.low[0] + 1,
                                         inbox.high[1] - inbox.low[1] + 1,
                                         inbox.high[2] - inbox.low[2] + 1};
        std::array<int, 3> local_offset = {inbox.low[0], inbox.low[1], inbox.low[2]};

        auto [ox, oy, oz] = pfc::world::get_origin(world);
        auto [dx, dy, dz] = pfc::world::get_spacing(world);
        vtk_writer->set_domain(global_size, local_size, local_offset);
        vtk_writer->set_origin({ox, oy, oz});
        vtk_writer->set_spacing({dx, dy, dz});
        vtk_writer->set_field_name(name);

        simulator.add_results_writer(name, std::move(vtk_writer));
      } else {
        if (rank0)
          std::cout << "Creating binary writer for field '" << name << "' -> "
                    << data << std::endl;
        auto binary_writer = std::make_unique<pfc::BinaryWriter>(data);
        simulator.add_results_writer(name, std::move(binary_writer));
      }
    }
  }

  if (rank0) std::cout << "Applying initial conditions..." << std::endl;
  model.prepare_for_field_modifiers();
  simulator.apply_initial_conditions();
  model.finalize_after_field_modifiers();

  if (rank0) std::cout << "Writing initial state..." << std::endl;
  simulator.write_results();

  if (rank0) std::cout << "Starting simulation..." << std::endl;

  while (!time.done()) {
    time.next();

    model.prepare_for_field_modifiers();
    simulator.apply_boundary_conditions();
    model.finalize_after_field_modifiers();

    double t = time.get_current();
    model.step(t);

    double saveat = time.get_saveat();
    double dt = time.get_dt();
    if (saveat > 0.0 && dt > 0.0) {
      int save_interval = static_cast<int>(std::round(saveat / dt));
      if (save_interval > 0 && time.get_increment() % save_interval == 0) {
        if (rank0)
          std::cout << "Step " << time.get_increment() << ", t = " << t
                    << ", writing results..." << std::endl;
        simulator.write_results();
      }
    }

    if (rank0 && time.get_increment() % 10 == 0) {
      std::cout << "Step " << time.get_increment() << ", t = " << t << std::endl;
    }
  }

  if (rank0) std::cout << "Simulation completed!" << std::endl;

  return 0;
}
