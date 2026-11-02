// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/distributed_complex_line_fft.hpp>

#include <openpfc/kernel/fft/complex_line_fft.hpp>

#include <heffte_fft3d.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pfc::fft {
namespace {
void require_collective(int invalid, MPI_Comm comm, const char *message) {
  int any = 0;
  MPI_Allreduce(&invalid, &any, 1, MPI_INT, MPI_MAX, comm);
  if (any) throw std::invalid_argument(message);
}

[[nodiscard]] int bytes_of(int count) {
  return count * static_cast<int>(sizeof(std::complex<double>));
}
} // namespace

struct DistributedComplexLineFFT::Impl {
  int length = 0;
  int offset = 0;
  int local = 0;
  int ranks = 1;
  MPI_Comm comm = MPI_COMM_NULL;
  std::unique_ptr<heffte::fft3d<heffte::backend::fftw>> plan;
  std::unique_ptr<ComplexLineFFT> serial;
  std::vector<std::complex<double>> stage;
  std::vector<std::complex<double>> work;
  std::vector<std::complex<double>> gathered;
  std::vector<std::complex<double>> full;
  std::vector<int> counts;
  std::vector<int> displs;
  std::size_t workspace = 0;
  std::size_t communication = 0;

  Impl(int n, MPI_Comm c) : length(n), comm(c) {
    if (c == MPI_COMM_NULL) {
      throw std::invalid_argument("DistributedComplexLineFFT: null communicator");
    }
    int rank = 0;
    MPI_Comm_rank(c, &rank);
    MPI_Comm_size(c, &ranks);
    const int unit = static_cast<int>(sizeof(std::complex<double>));
    int invalid = n < 2 || n < ranks || n > std::numeric_limits<int>::max() / unit;
    require_collective(invalid, c,
                       "DistributedComplexLineFFT: length must cover every rank");
    int lo = n;
    int hi = n;
    MPI_Allreduce(&n, &lo, 1, MPI_INT, MPI_MIN, c);
    MPI_Allreduce(&n, &hi, 1, MPI_INT, MPI_MAX, c);
    require_collective(lo != hi, c,
                       "DistributedComplexLineFFT: ranks disagree on the length");
    const auto block = line_block(n, rank, ranks);
    offset = block.offset;
    local = block.count;
    stage.resize(static_cast<std::size_t>(local));
    if (ranks == 1) {
      const auto world = heffte::box3d<>({0, 0, 0}, {n - 1, 0, 0});
      const auto box = heffte::split_world(world, {1, 1, 1})[0];
      auto options = heffte::default_options<heffte::backend::fftw>();
      plan = std::make_unique<heffte::fft3d<heffte::backend::fftw>>(box, box, c,
                                                                    options);
      const auto scratch = std::max<std::size_t>(plan->size_workspace(), 1);
      work.resize(scratch);
      workspace = plan->size_workspace() * sizeof(std::complex<double>);
      communication = plan->size_comm_buffers() * sizeof(std::complex<double>);
      return;
    }
    // HeFFTe's process grid cannot sit on two axes of length one.
    serial = std::make_unique<ComplexLineFFT>(n);
    gathered.resize(static_cast<std::size_t>(n));
    full.resize(static_cast<std::size_t>(n));
    counts.resize(static_cast<std::size_t>(ranks));
    displs.resize(static_cast<std::size_t>(ranks));
    for (int peer = 0; peer < ranks; ++peer) {
      const auto peer_block = line_block(n, peer, ranks);
      counts[static_cast<std::size_t>(peer)] = bytes_of(peer_block.count);
      displs[static_cast<std::size_t>(peer)] = bytes_of(peer_block.offset);
    }
    workspace = 2 * static_cast<std::size_t>(n) * sizeof(std::complex<double>);
    communication =
        static_cast<std::size_t>(n - local) * sizeof(std::complex<double>);
  }

  void execute(std::span<const std::complex<double>> in,
               std::span<std::complex<double>> out, bool backward) {
    int bad = in.size() != static_cast<std::size_t>(local) ||
              out.size() != static_cast<std::size_t>(local);
    require_collective(bad, comm,
                       "DistributedComplexLineFFT: span has the wrong length");
    std::copy(in.begin(), in.end(), stage.begin());
    if (plan) {
      if (backward) {
        plan->backward(stage.data(), out.data(), work.data(), heffte::scale::none);
      } else {
        plan->forward(stage.data(), out.data(), work.data(), heffte::scale::none);
      }
      return;
    }
    MPI_Allgatherv(stage.data(), bytes_of(local), MPI_BYTE, gathered.data(),
                   counts.data(), displs.data(), MPI_BYTE, comm);
    if (backward) {
      serial->backward(gathered, full);
    } else {
      serial->forward(gathered, full);
    }
    std::copy_n(full.begin() + offset, local, out.begin());
  }
};

DistributedComplexLineFFT::DistributedComplexLineFFT(int global_length,
                                                     MPI_Comm comm)
    : m_impl(std::make_unique<Impl>(global_length, comm)) {}
DistributedComplexLineFFT::~DistributedComplexLineFFT() = default;
DistributedComplexLineFFT::DistributedComplexLineFFT(
    DistributedComplexLineFFT &&) noexcept = default;
DistributedComplexLineFFT &DistributedComplexLineFFT::operator=(
    DistributedComplexLineFFT &&) noexcept = default;

void DistributedComplexLineFFT::forward(std::span<const std::complex<double>> in,
                                        std::span<std::complex<double>> out) {
  m_impl->execute(in, out, false);
}
void DistributedComplexLineFFT::backward(std::span<const std::complex<double>> in,
                                         std::span<std::complex<double>> out) {
  m_impl->execute(in, out, true);
}
int DistributedComplexLineFFT::global_length() const { return m_impl->length; }
int DistributedComplexLineFFT::local_length() const { return m_impl->local; }
int DistributedComplexLineFFT::local_offset() const { return m_impl->offset; }
std::size_t DistributedComplexLineFFT::workspace_bytes() const {
  return m_impl->workspace;
}
std::size_t DistributedComplexLineFFT::comm_bytes() const {
  return m_impl->communication;
}

} // namespace pfc::fft
