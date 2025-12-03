// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_cuda_vtk.cpp
 * @brief Test Tungsten CUDA model with initial/boundary conditions and VTK output
 *
 * This program runs a tungsten simulation with:
 * - CUDA/GPU acceleration
 * - Initial conditions (constant + single seed)
 * - Boundary conditions (fixed)
 * - VTK output for visualization
 */

#if !defined(OpenPFC_ENABLE_CUDA)
#error "This test requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include "tungsten_cuda_model.hpp"
#include "tungsten_input.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <openpfc/results_writer.hpp>
#include <openpfc/results_writers/vtk_writer.hpp>
#include <openpfc/simulator.hpp>
#include <openpfc/ui.hpp>

using namespace pfc;
using namespace pfc::ui;
using json = nlohmann::json;

int main(int argc, char *argv[]) {
  MPI_Worker worker(argc, argv);
  int rank = worker.get_rank();
  int num_ranks = worker.get_num_ranks();
  bool rank0 = (rank == 0);

  // Parse command line arguments
  std::string config_file = "tungsten_single_seed_256_cuda.json";
  if (argc > 1) {
    config_file = argv[1];
  }

  // Read configuration
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
    std::cout << "Tungsten CUDA Test with VTK Output" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration: " << config_file << std::endl;
    std::cout << "MPI ranks: " << num_ranks << std::endl;
    std::cout << "========================================" << std::endl;
  }

  // Create world from configuration
  World world(ui::from_json<World>(settings));
  if (rank0) {
    std::cout << "World: " << world << std::endl;
  }

  // Create decomposition and FFT layout
  auto decomp = decomposition::create(world, num_ranks);
  heffte::plan_options options = heffte::default_options<heffte::backend::cufft>();
  if (settings.contains("plan_options")) {
    options = ui::from_json<heffte::plan_options>(settings["plan_options"]);
  }
  auto fft_layout = fft::layout::create(decomp, 0);

  // Create CUDA model
  TungstenCUDA<double> model(world);

  // Set CUDA FFT (this creates the CUDA FFT internally)
  // We need to pass a dummy CPU FFT for interface compatibility
  auto dummy_fft = fft::create(fft_layout, rank, options);
  model.set_fft(dummy_fft);

  // Load model parameters
  if (settings.contains("model") && settings["model"].contains("params")) {
    from_json(settings["model"]["params"], model);
  }

  // Create time stepper
  Time time(ui::from_json<Time>(settings));

  // Initialize model (allocates GPU memory, precomputes operators)
  if (rank0) std::cout << "Initializing model..." << std::endl;
  model.initialize(time.get_dt());

  // Create simulator
  Simulator simulator(model, time);

  // Add initial conditions
  if (rank0) std::cout << "Adding initial conditions..." << std::endl;
  if (settings.contains("initial_conditions")) {
    for (const json &params : settings["initial_conditions"]) {
      std::string type = params["type"];
      auto field_modifier = create_field_modifier(type, params);
      if (params.contains("target")) {
        field_modifier->set_field_name(params["target"]);
      }
      simulator.add_initial_conditions(std::move(field_modifier));
    }
  }

  // Add boundary conditions
  if (rank0) std::cout << "Adding boundary conditions..." << std::endl;
  if (settings.contains("boundary_conditions")) {
    for (const json &params : settings["boundary_conditions"]) {
      std::string type = params["type"];
      auto field_modifier = create_field_modifier(type, params);
      if (params.contains("target")) {
        field_modifier->set_field_name(params["target"]);
      }
      simulator.add_boundary_conditions(std::move(field_modifier));
    }
  }

  // Add VTK writer
  if (rank0) std::cout << "Adding VTK writer..." << std::endl;
  if (settings.contains("fields") && settings["saveat"] > 0) {
    for (const auto &field : settings["fields"]) {
      std::string name = field["name"];
      std::string data = field["data"];

      // Check if it's a VTK file (.vti extension)
      if (data.find(".vti") != std::string::npos ||
          data.find(".vtk") != std::string::npos) {
        if (rank0) {
          std::cout << "Creating VTK writer for field '" << name << "' -> " << data
                    << std::endl;
          // Create results directory
          std::filesystem::path results_dir(data);
          if (results_dir.has_filename()) results_dir = results_dir.parent_path();
          if (!std::filesystem::exists(results_dir)) {
            std::filesystem::create_directories(results_dir);
          }
        }

        auto vtk_writer = std::make_unique<VTKWriter>(data);

        // Get inbox from CUDA FFT using helper function
        auto &cuda_fft = model.get_cuda_fft();
        auto inbox = fft::get_inbox(cuda_fft);

        auto [Lx, Ly, Lz] = get_size(world);
        std::array<int, 3> global_size = {Lx, Ly, Lz};
        std::array<int, 3> local_size = {inbox.high[0] - inbox.low[0] + 1,
                                         inbox.high[1] - inbox.low[1] + 1,
                                         inbox.high[2] - inbox.low[2] + 1};
        std::array<int, 3> local_offset = {inbox.low[0], inbox.low[1], inbox.low[2]};

        auto [ox, oy, oz] = get_origin(world);
        auto [dx, dy, dz] = get_spacing(world);
        vtk_writer->set_domain(global_size, local_size, local_offset);
        vtk_writer->set_origin({ox, oy, oz});
        vtk_writer->set_spacing({dx, dy, dz});
        vtk_writer->set_field_name(name);

        simulator.add_results_writer(name, std::move(vtk_writer));
      } else {
        // Binary writer for non-VTK files
        if (rank0)
          std::cout << "Creating binary writer for field '" << name << "' -> "
                    << data << std::endl;
        auto binary_writer = std::make_unique<BinaryWriter>(data);
        simulator.add_results_writer(name, std::move(binary_writer));
      }
    }
  }

  // Apply initial conditions (with GPU sync)
  if (rank0) std::cout << "Applying initial conditions..." << std::endl;
  model.prepare_for_field_modifiers();
  simulator.apply_initial_conditions();
  model.finalize_after_field_modifiers();

  // Apply initial conditions (before first time step)
  if (rank0) std::cout << "Applying initial conditions..." << std::endl;
  model.prepare_for_field_modifiers();
  simulator.apply_initial_conditions();
  model.finalize_after_field_modifiers();

  // Write initial state (after applying initial conditions)
  if (rank0) std::cout << "Writing initial state..." << std::endl;
  simulator.write_results();

  // Main simulation loop
  if (rank0) std::cout << "Starting simulation..." << std::endl;

  while (!time.done()) {
    time.next();

    // Apply boundary conditions (with GPU sync)
    model.prepare_for_field_modifiers();
    simulator.apply_boundary_conditions();
    model.finalize_after_field_modifiers();

    // Time step
    double t = time.get_current();
    model.step(t);

    // Write results if needed (check saveat interval)
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
