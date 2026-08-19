// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_GPU_STACK_HIP) &&                                    \
    !defined(OPENPFC_TEST_GPU_STACK_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>

#if defined(OPENPFC_TEST_GPU_STACK_HIP)
using Space = pfc::HipSpace;
#elif defined(OPENPFC_TEST_GPU_STACK_CUDA)
using Space = pfc::CudaSpace;
#endif

TEST_CASE("GpuSpectralStack inbox field matches device FFT",
          "[gpu][spectral_stack]") {
#if defined(OPENPFC_TEST_GPU_STACK_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
#endif

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  constexpr int N = 8;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  pfc::sim::stacks::GpuSpectralStack<Space> stack(domain, rank, mpi_size,
                                                  MPI_COMM_WORLD);
  REQUIRE(stack.fft().size_inbox() == stack.u().size());
  REQUIRE(stack.fft().size_outbox() > 0);
  REQUIRE(stack.rank() == rank);
  REQUIRE(stack.nproc() == mpi_size);
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  const int result = Catch::Session().run(argc, argv);
  return result;
}

#endif
