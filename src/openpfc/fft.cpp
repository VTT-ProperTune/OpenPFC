// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/fft.hpp"

namespace pfc {
namespace fft {
namespace layout {

#include <array>
#include <iostream>
#include <sstream>

// Helper function to print std::array
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const std::array<T, N> &arr) {
  os << "{";
  for (std::size_t i = 0; i < N; ++i) {
    os << arr[i];
    if (i < N - 1) {
      os << ", ";
    }
  }
  os << "}";
  return os;
}

using heffte::split_world;

auto get_real_indices(const Decomposition &decomposition) {
  auto world = get_global_world(decomposition);
  auto [N1, N2, N3] = get_size(world);
  return heffte::box3d<int>({0, 0, 0}, {N1 - 1, N2 - 1, N3 - 1});
}

auto get_complex_indices(const Decomposition &decomposition, int r2c_direction) {
  auto [N1, N2, N3] = get_size(get_global_world(decomposition));
  if (r2c_direction == 0) {
    return heffte::box3d<int>({0, 0, 0}, {N1 / 2, N2 - 1, N3 - 1});
  } else if (r2c_direction == 1) {
    return heffte::box3d<int>({0, 0, 0}, {N1 - 1, N2 / 2, N3 - 1});
  } else if (r2c_direction == 2) {
    return heffte::box3d<int>({0, 0, 0}, {N1 - 1, N2 - 1, N3 / 2});
  } else {
    throw std::logic_error("Invalid r2c_direction: " +
                           std::to_string(r2c_direction));
  }
}

const FFTLayout create(const Decomposition &decomposition, int r2c_direction) {
  auto real_indices = get_real_indices(decomposition);
  auto complex_indices = get_complex_indices(decomposition, r2c_direction);
  auto grid = get_grid(decomposition);
  auto real_boxes = split_world(real_indices, grid);
  auto complex_boxes = split_world(complex_indices, grid);
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

using heffte::plan_options;
using layout::FFTLayout;
using fft_r2c = heffte::fft3d_r2c<heffte::backend::fftw>;

FFT create(const FFTLayout &fft_layout, int rank_id, plan_options options) {
  auto inbox = get_real_box(fft_layout, rank_id);
  auto outbox = get_complex_box(fft_layout, rank_id);
  auto r2c_dir = get_r2c_direction(fft_layout);
  auto comm = get_comm();
  return FFT(fft_r2c(inbox, outbox, r2c_dir, comm, options));
}

FFT create(const Decomposition &decomposition, int rank_id) {
  auto options = heffte::default_options<heffte::backend::fftw>();
  auto r2c_dir = 0;
  auto fft_layout = layout::create(decomposition, r2c_dir);
  return create(fft_layout, rank_id, options);
}

// Runtime backend selection - returns IFFT interface
std::unique_ptr<IFFT> create_with_backend(const FFTLayout &fft_layout, int rank_id,
                                          plan_options options, Backend backend) {
  auto inbox = get_real_box(fft_layout, rank_id);
  auto outbox = get_complex_box(fft_layout, rank_id);
  auto r2c_dir = get_r2c_direction(fft_layout);
  auto comm = get_comm();

  switch (backend) {
  case Backend::FFTW: {
    using fft_type = heffte::fft3d_r2c<heffte::backend::fftw>;
    return std::make_unique<FFT_Impl<heffte::backend::fftw>>(
        fft_type(inbox, outbox, r2c_dir, comm, options));
  }
#if defined(OpenPFC_ENABLE_CUDA)
  case Backend::CUDA: {
    using fft_type = heffte::fft3d_r2c<heffte::backend::cufft>;
    return std::make_unique<FFT_Impl<heffte::backend::cufft>>(
        fft_type(inbox, outbox, r2c_dir, comm, options));
  }
#endif
  default: throw std::runtime_error("Unsupported FFT backend requested");
  }
}

std::unique_ptr<IFFT> create_with_backend(const Decomposition &decomposition,
                                          int rank_id, Backend backend) {
  auto r2c_dir = 0;
  auto fft_layout = layout::create(decomposition, r2c_dir);

  // Get default options for the selected backend
  switch (backend) {
  case Backend::FFTW: {
    auto options = heffte::default_options<heffte::backend::fftw>();
    return create_with_backend(fft_layout, rank_id, options, backend);
  }
#if defined(OpenPFC_ENABLE_CUDA)
  case Backend::CUDA: {
    auto options = heffte::default_options<heffte::backend::cufft>();
    return create_with_backend(fft_layout, rank_id, options, backend);
  }
#endif
  default: throw std::runtime_error("Unsupported FFT backend requested");
  }
}

FFT create(const Decomposition &decomposition) {
  auto comm = get_comm();
  auto mpi_comm_size = get_mpi_size(comm);
  auto rank_id = get_mpi_rank(comm);
  auto decomposition_size = get_num_domains(decomposition);
  if (mpi_comm_size != decomposition_size) {
    throw std::logic_error(
        "Mismatch between MPI communicator size and domain decomposition size: " +
        std::to_string(mpi_comm_size) + " != " + std::to_string(decomposition_size) +
        ". This indicates that the number of MPI ranks does not match the number of "
        "domains in the decomposition. To resolve this issue, you can manually "
        "specify the rank by calling fft::create(decomposition, rank_id) instead.");
  }
  // if mpi communicator size matches decomposition size, we can safely assume
  // that the intention is to decompose the whole communicator into the
  // decomposition
  return create(decomposition, rank_id);
}

} // namespace fft
} // namespace pfc
