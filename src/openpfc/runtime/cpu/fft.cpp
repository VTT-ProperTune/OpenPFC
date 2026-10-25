// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/complex_outbox.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

namespace pfc::fft::layout {

namespace {

[[nodiscard]] Box3i box3i_from_heffte(const heffte::box3d<int> &b) {
  return Box3i{b.low, b.high, b.size};
}

[[nodiscard]] std::vector<Box3i>
boxes_from_heffte(const std::vector<heffte::box3d<int>> &boxes) {
  std::vector<Box3i> out;
  out.reserve(boxes.size());
  for (const auto &bx : boxes) {
    out.push_back(box3i_from_heffte(bx));
  }
  return out;
}

} // namespace

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
  auto dom = decomposition::domain(decomposition);
  auto [N1, N2, N3] = pfc::domain::get_size(dom);
  return heffte::box3d<int>({0, 0, 0}, {N1 - 1, N2 - 1, N3 - 1});
}

auto get_complex_indices(const Decomposition &decomposition, int r2c_direction) {
  auto [N1, N2, N3] = pfc::domain::get_size(decomposition::domain(decomposition));
  if (r2c_direction == 0) {
    return heffte::box3d<int>({0, 0, 0}, {N1 / 2, N2 - 1, N3 - 1});
  }
  if (r2c_direction == 1) {
    return heffte::box3d<int>({0, 0, 0}, {N1 - 1, N2 / 2, N3 - 1});
  }
  if (r2c_direction == 2) {
    return heffte::box3d<int>({0, 0, 0}, {N1 - 1, N2 - 1, N3 / 2});
  }
  throw std::logic_error("Invalid r2c_direction: " + std::to_string(r2c_direction));
}

[[nodiscard]] FFTLayout create(const Decomposition &decomposition, int r2c_direction,
                               const heffte::plan_options &options) {
  auto real_indices = get_real_indices(decomposition);
  auto complex_indices = get_complex_indices(decomposition, r2c_direction);
  auto grid = get_grid(decomposition);
  auto real_boxes = boxes_from_heffte(split_world(real_indices, grid));
  const pfc::Int3 cgrid = complex_proc_grid_for_r2c(
      grid, real_indices, complex_indices, r2c_direction, options);
  auto complex_boxes = boxes_from_heffte(split_world(complex_indices, cgrid));
  return FFTLayout{
      decomposition, r2c_direction, std::move(real_boxes), std::move(complex_boxes),
      grid,          cgrid};
}

[[nodiscard]] FFTLayout create(const Decomposition &decomposition,
                               int r2c_direction) {
  return create(decomposition, r2c_direction,
                heffte::default_options<heffte::backend::fftw>());
}

} // namespace pfc::fft::layout

namespace pfc::fft {

using pfc::decomposition::get_num_domains;

namespace {

[[nodiscard]] heffte::box3d<int> heffte_box_from_box3i(const Box3i &b) {
  return heffte::box3d<int>(b.low, b.high);
}

} // namespace

int get_mpi_rank(MPI_Comm comm) {
  int rank;
  int err = MPI_Comm_rank(comm, &rank);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Comm_rank");
  return rank;
}

int get_mpi_size(MPI_Comm comm) {
  int size;
  int err = MPI_Comm_size(comm, &size);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Comm_size");
  return size;
}

using layout::FFTLayout;
using fft_r2c = heffte::fft3d_r2c<heffte::backend::fftw>;

namespace {

// Validate before entering HeFFTe's collective plan creation. All overloads
// share this admission rule, including the explicit-rank interfaces.
void validate_plan_participation(const FFTLayout &layout, int rank, MPI_Comm comm) {
  if (get_mpi_size(comm) != get_num_domains(layout.m_decomposition)) {
    throw std::logic_error("FFT plan: communicator size differs from decomposition");
  }
  if (rank != get_mpi_rank(comm)) {
    throw std::invalid_argument("FFT plan: rank_id must be the communicator rank");
  }
}

void log_r2c_layout(int rank_id, const pfc::Int3 &real_grid,
                    const pfc::Int3 &complex_grid) {
  if (rank_id != 0 || std::getenv("OPENPFC_FFT_LOG_LAYOUT") == nullptr) {
    return;
  }
  std::cerr << "FFT_R2C_LAYOUT real_grid=" << layout::format_proc_grid(real_grid)
            << " complex_grid=" << layout::format_proc_grid(complex_grid) << '\n';
}

} // namespace

[[nodiscard]] CPUFFT create(const FFTLayout &fft_layout, int rank_id,
                            const heffte::plan_options &options, MPI_Comm comm) {
  validate_plan_participation(fft_layout, rank_id, comm);
  const auto &inbox = get_real_box(fft_layout, rank_id);
  const auto &outbox = get_complex_box(fft_layout, rank_id);
  auto r2c_dir = get_r2c_direction(fft_layout);
  log_r2c_layout(rank_id, get_real_proc_grid(fft_layout),
                 get_complex_proc_grid(fft_layout));
  return {fft_r2c(heffte_box_from_box3i(inbox), heffte_box_from_box3i(outbox),
                  r2c_dir, comm, options)};
}

[[nodiscard]] CPUFFT create(const Decomposition &decomposition, int rank_id,
                            MPI_Comm comm, int r2c_direction,
                            const heffte::plan_options &options) {
  const auto plan_opts =
      layout::effective_r2c_plan_options<heffte::backend::fftw>(options);
  auto fft_layout = layout::create(decomposition, r2c_direction, plan_opts);
  return create(fft_layout, rank_id, options, comm);
}

[[nodiscard]] CPUFFT create(const Decomposition &decomposition, int rank_id,
                            MPI_Comm comm, int r2c_direction) {
  return create(decomposition, rank_id, comm, r2c_direction,
                heffte::default_options<heffte::backend::fftw>());
}

[[nodiscard]] std::unique_ptr<IHostFFT>
create_with_backend(const FFTLayout &fft_layout, int rank_id,
                    const heffte::plan_options &options, Backend backend,
                    MPI_Comm comm) {
  validate_plan_participation(fft_layout, rank_id, comm);
  auto inbox = get_real_box(fft_layout, rank_id);
  auto outbox = get_complex_box(fft_layout, rank_id);
  auto r2c_dir = get_r2c_direction(fft_layout);

  switch (backend) {
  case Backend::FFTW: {
    using fft_type = heffte::fft3d_r2c<heffte::backend::fftw>;
    return std::make_unique<FFT_Impl<heffte::backend::fftw>>(
        fft_type(heffte_box_from_box3i(inbox), heffte_box_from_box3i(outbox),
                 r2c_dir, comm, options));
  }
  case Backend::CUDA:
    throw std::invalid_argument(
        "fft::create_with_backend: Backend::CUDA is a device FFT; "
        "use fft::create_cuda instead of IHostFFT");
  case Backend::HIP:
    throw std::invalid_argument(
        "fft::create_with_backend: Backend::HIP is a device FFT; "
        "use fft::create_hip instead of IHostFFT");
  default: throw std::runtime_error("Unsupported FFT backend requested");
  }
}

[[nodiscard]] std::unique_ptr<IHostFFT>
create_with_backend(const Decomposition &decomposition, int rank_id, Backend backend,
                    MPI_Comm comm, int r2c_direction) {
  switch (backend) {
  case Backend::FFTW: {
    auto options = heffte::default_options<heffte::backend::fftw>();
    const auto plan_opts =
        layout::effective_r2c_plan_options<heffte::backend::fftw>(options);
    auto fft_layout = layout::create(decomposition, r2c_direction, plan_opts);
    return create_with_backend(fft_layout, rank_id, options, backend, comm);
  }
  case Backend::CUDA:
    throw std::invalid_argument(
        "fft::create_with_backend: Backend::CUDA is a device FFT; "
        "use fft::create_cuda instead of IHostFFT");
  case Backend::HIP:
    throw std::invalid_argument(
        "fft::create_with_backend: Backend::HIP is a device FFT; "
        "use fft::create_hip instead of IHostFFT");
  default: throw std::runtime_error("Unsupported FFT backend requested");
  }
}

[[nodiscard]] CPUFFT create(const Decomposition &decomposition, MPI_Comm comm) {
  const int mpi_comm_size = get_mpi_size(comm);
  const int rank_id = get_mpi_rank(comm);
  const auto decomposition_size = get_num_domains(decomposition);
  if (mpi_comm_size != decomposition_size) {
    throw std::logic_error(
        "Mismatch between MPI communicator size and domain decomposition size: " +
        std::to_string(mpi_comm_size) + " != " + std::to_string(decomposition_size) +
        ". Build the decomposition for this communicator before creating the plan.");
  }
  return create(decomposition, rank_id, comm);
}

} // namespace pfc::fft
