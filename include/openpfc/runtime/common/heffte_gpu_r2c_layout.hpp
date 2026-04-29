// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heffte_gpu_r2c_layout.hpp
 * @brief Shared MPI + HeFFTe box layout helpers for CUDA/HIP GPU FFT factories
 *
 * @details
 * `fft_cuda.cpp` and `fft_hip.cpp` share the same decomposition checks and
 * default r2c layout construction; keep that logic in one place under
 * `runtime/common/`.
 */

#pragma once

#include <heffte.h>
#include <mpi.h>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>

namespace pfc::runtime::heffte_gpu {

inline int mpi_comm_rank(MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

inline int mpi_comm_size(MPI_Comm comm) {
  int size = 0;
  MPI_Comm_size(comm, &size);
  return size;
}

inline heffte::box3d<int> to_heffte_box(const pfc::fft::Box3i &b) {
  return heffte::box3d<int>(b.low, b.high);
}

struct DefaultR2cBoxes {
  heffte::box3d<int> real_inbox;
  heffte::box3d<int> complex_outbox;
  int r2c_direction = 0;
};

inline DefaultR2cBoxes
make_default_r2c_boxes(const pfc::decomposition::Decomposition &decomposition,
                       int rank_id) {
  using namespace pfc::fft::layout;
  constexpr int r2c_dir = 0;
  auto fft_layout = create(decomposition, r2c_dir);
  const auto &inbox = get_real_box(fft_layout, rank_id);
  const auto &outbox = get_complex_box(fft_layout, rank_id);
  return {to_heffte_box(inbox), to_heffte_box(outbox),
          get_r2c_direction(fft_layout)};
}

[[gnu::cold]] inline void throw_if_mpi_decomposition_mismatch(
    MPI_Comm comm, const pfc::decomposition::Decomposition &decomposition,
    const char *use_overload_with_rank_id) {
  const int mpi_sz = mpi_comm_size(comm);
  const int domains = pfc::decomposition::get_num_domains(decomposition);
  if (mpi_sz != domains) {
    throw std::logic_error(
        "Mismatch between MPI communicator size and domain decomposition size: " +
        std::to_string(mpi_sz) + " != " + std::to_string(domains) +
        ". This indicates that the number of MPI ranks does not match the number of "
        "domains in the decomposition. To resolve this issue, you can manually "
        "specify the rank by calling " +
        std::string(use_overload_with_rank_id) + " instead.");
  }
}

} // namespace pfc::runtime::heffte_gpu
