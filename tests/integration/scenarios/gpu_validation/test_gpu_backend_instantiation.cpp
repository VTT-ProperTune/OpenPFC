// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/hip/fft_hip.hpp>
#endif

using namespace pfc;
using namespace pfc::fft;

TEST_CASE("CUDA backend instantiation smoke", "[integration][gpu][cuda]") {
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

TEST_CASE("HIP backend instantiation smoke", "[integration][gpu][hip]") {
  auto world = world::uniform(16, 1.0);
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto decomp = decomposition::create(world, size);

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
  auto cpu_fft = create(decomp);
  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  auto gpu_fft = create_hip(decomp, rank, MPI_COMM_WORLD);
  REQUIRE(cpu_fft.size_inbox() == gpu_fft.size_inbox());
  REQUIRE(cpu_fft.size_outbox() == gpu_fft.size_outbox());
#else
  SUCCEED("HIP spectral not enabled - skipping GPU backend instantiation test");
#endif
}
