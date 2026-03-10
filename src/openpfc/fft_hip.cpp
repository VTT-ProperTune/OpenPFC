// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_HIP)

#include "openpfc/fft_hip.hpp"
#include "openpfc/core/decomposition.hpp"
#include "openpfc/fft.hpp"

#include <heffte.h>
#include <mpi.h>
#include <stdexcept>

namespace pfc {
namespace fft {

using heffte::plan_options;
using layout::FFTLayout;
using layout::get_complex_box;
using layout::get_r2c_direction;
using layout::get_real_box;
using pfc::decomposition::get_num_domains;
using pfc::fft::FFT_Impl;

static auto get_comm() { return MPI_COMM_WORLD; }

static int get_mpi_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

static int get_mpi_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

FFT_HIP create_hip(const Decomposition &decomposition, int rank_id) {
  auto options = heffte::default_options<heffte::backend::rocfft>();
  auto r2c_dir = 0;
  auto fft_layout = layout::create(decomposition, r2c_dir);

  auto inbox = get_real_box(fft_layout, rank_id);
  auto outbox = get_complex_box(fft_layout, rank_id);
  auto r2c_direction = get_r2c_direction(fft_layout);
  auto comm = get_comm();

  using fft_r2c_hip_type = heffte::fft3d_r2c<heffte::backend::rocfft>;
  fft_r2c_hip_type fft_hip(inbox, outbox, r2c_direction, comm, options);

  return FFT_HIP(std::move(fft_hip));
}

FFT_HIP create_hip(const Decomposition &decomposition) {
  auto comm = get_comm();
  auto mpi_comm_size = get_mpi_size(comm);
  auto rank_id = get_mpi_rank(comm);
  auto decomposition_size = pfc::decomposition::get_num_domains(decomposition);

  if (mpi_comm_size != decomposition_size) {
    throw std::logic_error(
        "Mismatch between MPI communicator size and domain decomposition size: " +
        std::to_string(mpi_comm_size) + " != " + std::to_string(decomposition_size) +
        ". This indicates that the number of MPI ranks does not match the number of "
        "domains in the decomposition. To resolve this issue, you can manually "
        "specify the rank by calling fft::create_hip(decomposition, rank_id) "
        "instead.");
  }

  return create_hip(decomposition, rank_id);
}

} // namespace fft
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
