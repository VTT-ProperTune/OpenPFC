// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fixtures/diffusion_model.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/results_writers/vtk_writer.hpp>

using namespace pfc;
using namespace pfc::test;
namespace fs = std::filesystem;

TEST_CASE("VTK roundtrip write", "[integration][io][vtk]") {
  auto tmpdir = fs::path(".temp/tests/integration/vtk");
  fs::create_directories(tmpdir);

  auto world = world::uniform(16, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  DiffusionModel model(fft, world);
  model.initialize(1.0e-3);

  // Write a single snapshot
  auto writer = std::make_unique<VTKWriter>((tmpdir / "output_%04d.vti").string());
  // Single-rank domain: local == global, offset == 0
  auto global_size = world::get_size(world);
  auto local_size = global_size;
  std::array<int, 3> offset{0, 0, 0};
  writer->set_domain(global_size, local_size, offset);

  // Acquire field and write
  auto &psi = model.get_field();
  writer->write(0, psi);

  // Expect exact file and non-zero size
  auto out = tmpdir / "output_0000.vti";
  REQUIRE(fs::exists(out));
  REQUIRE(fs::file_size(out) > 0);
}
