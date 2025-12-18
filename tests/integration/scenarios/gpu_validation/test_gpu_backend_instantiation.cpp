// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>

using namespace pfc;
using namespace pfc::fft;

TEST_CASE("GPU backend instantiation smoke", "[integration][gpu][cuda]") {
  auto world = world::uniform(16, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);

#if defined(OpenPFC_ENABLE_CUDA)
  auto cpu_fft = create_with_backend(decomp, /*rank*/ 0, Backend::FFTW);
  auto gpu_fft = create_with_backend(decomp, /*rank*/ 0, Backend::CUDA);
  REQUIRE(cpu_fft.get() != nullptr);
  REQUIRE(gpu_fft.get() != nullptr);
  REQUIRE(cpu_fft->size_inbox() == gpu_fft->size_inbox());
  REQUIRE(cpu_fft->size_outbox() == gpu_fft->size_outbox());
#else
  SUCCEED("CUDA disabled - skipping GPU backend instantiation test");
#endif
}
