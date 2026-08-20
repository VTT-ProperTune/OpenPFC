// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_FD_GPU_STACK_HIP) &&                                      \
    !defined(OPENPFC_TEST_FD_GPU_STACK_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/runtime/gpu/fd_gpu_stack.hpp>

#if defined(OPENPFC_TEST_FD_GPU_STACK_HIP)
using Space = pfc::HIPSpace;
#elif defined(OPENPFC_TEST_FD_GPU_STACK_CUDA)
using Space = pfc::CUDASpace;
#endif

TEST_CASE("FDGPUStack padded field and extra-field factory", "[gpu][fd_stack]") {
#if defined(OPENPFC_TEST_FD_GPU_STACK_HIP)
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
  auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  pfc::comm::HaloExchangeOptions opt;
  opt.directions = pfc::halo::presets::Axes2D();
  pfc::sim::stacks::FDGPUStack<Space> stack(domain, 1, rank, mpi_size,
                                            MPI_COMM_WORLD, opt);
  REQUIRE(stack.halo_width() == 1);
  REQUIRE(stack.u().storage_halo() == 1);
  REQUIRE(stack.u().local_size()[0] == N);
  REQUIRE(stack.rank() == rank);
  REQUIRE(stack.nproc() == mpi_size);

  auto extra = stack.make_field();
  REQUIRE(extra.size() == stack.u().size());
  REQUIRE(extra.storage_halo() == 1);

  auto group = stack.make_exchange({&stack.u(), &extra}, opt);
  stack.exchange_halos();
  group.exchange();
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  return Catch::Session().run(argc, argv);
}

#endif
