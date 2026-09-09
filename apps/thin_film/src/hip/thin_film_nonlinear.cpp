// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file thin_film_nonlinear.cpp
 * @brief `thin_film_nonlinear_hip` main: `run_thin_film_nonlinear` on HIP.
 *
 * Same JSON schema and observables as `thin_film_nonlinear`; see
 * `thin_film/nonlinear_driver.hpp` for the shared, `MemorySpace`-templated
 * physics and `apps/thin_film/src/gpu/thin_film_pointwise.inc` for the device
 * instantiations (`ThinFilmPointwise`, `PotentialPointwise`, and
 * `pfc::apps::MobilityGradPointwise<CubicMobility>`) this binary links
 * against.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "thin_film_nonlinear_hip requires HIP spectral support (rocFFT HeFFTe)"
#endif

#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <thin_film/nonlinear_driver.hpp>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: thin_film_nonlinear_hip CASE.json");
    status = thin_film::run_thin_film_nonlinear<
        pfc::HIPSpace, pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>(
        rank, nproc, MPI_COMM_WORLD, argv[1]);
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "thin_film_nonlinear_hip: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
