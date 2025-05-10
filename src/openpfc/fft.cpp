// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/fft.hpp"

namespace pfc {
namespace fft {
namespace layout {

const FFTLayout create(const Decomposition &decomposition, int r2c_direction,
                       int num_domains) {
  if (num_domains <= 0) {
    throw std::logic_error("Cannot construct domain decomposition: !(nprocs > 0)");
  }
  auto world = get_world(decomposition);
  auto [N1r, N2r, N3r] = get_size(world); // real size
  auto [N1c, N2c, N3c] = get_size(world); // complex size
  if (r2c_direction == 0) {
    N1c = N1c / 2 + 1;
  } else if (r2c_direction == 1) {
    N2c = N2c / 2 + 1;
  } else if (r2c_direction == 2) {
    N3c = N3c / 2 + 1;
  } else {
    throw std::logic_error("Invalid r2c_direction: " +
                           std::to_string(r2c_direction));
  }
  box3di real_indices({0, 0, 0}, {N1r - 1, N2r - 1, N3r - 1});
  box3di complex_indices({0, 0, 0}, {N1c - 1, N2c - 1, N3c - 1});
  Int3 grid = heffte::proc_setup_min_surface(real_indices, num_domains);
  std::vector<box3di> real_boxes = heffte::split_world(real_indices, grid);
  std::vector<box3di> complex_boxes = heffte::split_world(complex_indices, grid);
  return FFTLayout{decomposition, r2c_direction, real_boxes, complex_boxes};
}

} // namespace layout

auto get_comm() { return MPI_COMM_WORLD; }

int get_mpi_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int get_mpi_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

FFT create(const Decomposition &decomposition, MPI_Comm comm,
           heffte::plan_options options) {
  int rank = get_mpi_rank(comm);
  int mpi_num_ranks = get_mpi_size(comm);
  if (mpi_num_ranks <= 0) {
    throw std::logic_error("Cannot construct domain decomposition: !(nprocs > 0)");
  }
  auto r2c_dir = 0;
  auto fft_layout = fft::layout::create(decomposition, r2c_dir, mpi_num_ranks);
  auto inbox = get_real_box(fft_layout, rank);
  auto outbox = get_complex_box(fft_layout, rank);
  using fft_r2c = heffte::fft3d_r2c<heffte::backend::fftw>;
  return FFT(fft_r2c(inbox, outbox, r2c_dir, comm, options));
}

FFT create(const Decomposition &decomposition) {
  auto comm = get_comm();
  auto options = heffte::default_options<heffte::backend::fftw>();
  return create(decomposition, comm, options);
}

} // namespace fft
} // namespace pfc
