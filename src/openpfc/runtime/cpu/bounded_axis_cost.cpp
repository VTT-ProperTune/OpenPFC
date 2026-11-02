// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/bounded_axis_cost.hpp>

#include <openpfc/kernel/fft/chebyshev.hpp>
#include <openpfc/kernel/fft/complex_line_fft.hpp>
#include <openpfc/kernel/fft/distributed_complex_line_fft.hpp>
#include <openpfc/kernel/field/fd_apply.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace pfc::fft {
namespace {
void require_collective(int invalid, MPI_Comm comm, const char *message) {
  int any = 0;
  MPI_Allreduce(&invalid, &any, 1, MPI_INT, MPI_MAX, comm);
  if (any) throw std::invalid_argument(message);
}

[[nodiscard]] double max_rank(double value, MPI_Comm comm) {
  double reduced = 0;
  MPI_Allreduce(&value, &reduced, 1, MPI_DOUBLE, MPI_MAX, comm);
  return reduced;
}

[[nodiscard]] std::vector<double> fornberg_second(double origin,
                                                  const std::vector<double> &nodes) {
  const int n = static_cast<int>(nodes.size()) - 1;
  constexpr int derivative = 2;
  std::vector<std::vector<double>> weight(static_cast<std::size_t>(derivative + 1),
                                          std::vector<double>(nodes.size(), 0.0));
  weight[0][0] = 1.0;
  double product = 1.0;
  double previous_offset = nodes[0] - origin;
  for (int i = 1; i <= n; ++i) {
    const int highest = std::min(i, derivative);
    double node_product = 1.0;
    const double saved_offset = previous_offset;
    previous_offset = nodes[static_cast<std::size_t>(i)] - origin;
    for (int j = 0; j < i; ++j) {
      const double separation =
          nodes[static_cast<std::size_t>(i)] - nodes[static_cast<std::size_t>(j)];
      node_product *= separation;
      if (j == i - 1) {
        for (int k = highest; k >= 1; --k) {
          const auto row = static_cast<std::size_t>(k);
          const auto col = static_cast<std::size_t>(i);
          const auto prev = static_cast<std::size_t>(i - 1);
          weight[row][col] = product *
                             (static_cast<double>(k) * weight[row - 1][prev] -
                              saved_offset * weight[row][prev]) /
                             node_product;
        }
        weight[0][static_cast<std::size_t>(i)] =
            -product * saved_offset * weight[0][static_cast<std::size_t>(i - 1)] /
            node_product;
      }
      for (int k = highest; k >= 1; --k) {
        const auto row = static_cast<std::size_t>(k);
        const auto col = static_cast<std::size_t>(j);
        weight[row][col] = (previous_offset * weight[row][col] -
                            static_cast<double>(k) * weight[row - 1][col]) /
                           separation;
      }
      weight[0][static_cast<std::size_t>(j)] =
          previous_offset * weight[0][static_cast<std::size_t>(j)] / separation;
    }
    product = node_product;
  }
  return weight[static_cast<std::size_t>(derivative)];
}

template <int Order>
[[nodiscard]] double central_weight(int index, int node, double spacing) {
  using Stencil = pfc::field::fd::EvenCentralD2<Order>;
  std::vector<double> field(
      static_cast<std::size_t>(index + Stencil::half_width + 1), 0.0);
  field[static_cast<std::size_t>(node)] = 1.0;
  const double unscaled =
      pfc::field::fd::apply_d2_along<0, Stencil>(field.data(), index, 1, 0, 0);
  return unscaled / (static_cast<double>(Stencil::denom) * spacing * spacing);
}

template <int Order> [[nodiscard]] double uniform_dirichlet_error(int intervals) {
  using Stencil = pfc::field::fd::EvenCentralD2<Order>;
  constexpr int half = Stencil::half_width;
  const double spacing = 2.0 / static_cast<double>(intervals);
  const int unknown = intervals - 1;
  std::vector<double> matrix(static_cast<std::size_t>(unknown * unknown), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(unknown), 0.0);
  const double at_plus = std::exp(1.0);
  const double at_minus = std::exp(-1.0);
  for (int index = 1; index <= unknown; ++index) {
    const int equation = index - 1;
    const double node = 1.0 - static_cast<double>(index) * spacing;
    rhs[static_cast<std::size_t>(equation)] = std::exp(node);
    if (index >= half && index <= intervals - half) {
      for (int neighbor = index - half; neighbor <= index + half; ++neighbor) {
        const double value = central_weight<Order>(index, neighbor, spacing);
        if (neighbor == 0) {
          rhs[static_cast<std::size_t>(equation)] -= value * at_plus;
        } else if (neighbor == intervals) {
          rhs[static_cast<std::size_t>(equation)] -= value * at_minus;
        } else {
          matrix[static_cast<std::size_t>(equation * unknown + neighbor - 1)] +=
              value;
        }
      }
    } else {
      const int width = Order + 2;
      int start = index - (width / 2 - 1);
      if (start < 0) start = 0;
      const int last = intervals - width + 1;
      if (start > last) start = last;
      std::vector<double> nodes(static_cast<std::size_t>(width));
      for (int j = 0; j < width; ++j) {
        nodes[static_cast<std::size_t>(j)] =
            1.0 - static_cast<double>(start + j) * spacing;
      }
      const auto weight =
          fornberg_second(1.0 - static_cast<double>(index) * spacing, nodes);
      for (int k = 0; k < width; ++k) {
        const int node_index = start + k;
        const double value = weight[static_cast<std::size_t>(k)];
        if (node_index == 0) {
          rhs[static_cast<std::size_t>(equation)] -= value * at_plus;
        } else if (node_index == intervals) {
          rhs[static_cast<std::size_t>(equation)] -= value * at_minus;
        } else {
          matrix[static_cast<std::size_t>(equation * unknown + node_index - 1)] +=
              value;
        }
      }
    }
  }
  for (int column = 0; column < unknown; ++column) {
    int pivot = column;
    double largest =
        std::abs(matrix[static_cast<std::size_t>(column * unknown + column)]);
    for (int row = column + 1; row < unknown; ++row) {
      const double value =
          std::abs(matrix[static_cast<std::size_t>(row * unknown + column)]);
      if (value > largest) {
        largest = value;
        pivot = row;
      }
    }
    if (!(largest > 0.0)) {
      throw std::runtime_error("bounded axis: singular uniform stencil");
    }
    if (pivot != column) {
      for (int j = column; j < unknown; ++j) {
        std::swap(matrix[static_cast<std::size_t>(column * unknown + j)],
                  matrix[static_cast<std::size_t>(pivot * unknown + j)]);
      }
      std::swap(rhs[static_cast<std::size_t>(column)],
                rhs[static_cast<std::size_t>(pivot)]);
    }
    const double diagonal =
        matrix[static_cast<std::size_t>(column * unknown + column)];
    for (int row = column + 1; row < unknown; ++row) {
      const double factor =
          matrix[static_cast<std::size_t>(row * unknown + column)] / diagonal;
      matrix[static_cast<std::size_t>(row * unknown + column)] = 0.0;
      for (int j = column + 1; j < unknown; ++j) {
        matrix[static_cast<std::size_t>(row * unknown + j)] -=
            factor * matrix[static_cast<std::size_t>(column * unknown + j)];
      }
      rhs[static_cast<std::size_t>(row)] -=
          factor * rhs[static_cast<std::size_t>(column)];
    }
  }
  std::vector<double> solution(static_cast<std::size_t>(unknown));
  for (int row = unknown - 1; row >= 0; --row) {
    double sum = rhs[static_cast<std::size_t>(row)];
    for (int column = row + 1; column < unknown; ++column) {
      sum -= matrix[static_cast<std::size_t>(row * unknown + column)] *
             solution[static_cast<std::size_t>(column)];
    }
    solution[static_cast<std::size_t>(row)] =
        sum / matrix[static_cast<std::size_t>(row * unknown + row)];
  }
  double peak = 0.0;
  for (int index = 1; index <= unknown; ++index) {
    const double node = 1.0 - static_cast<double>(index) * spacing;
    peak = std::max(peak, std::abs(solution[static_cast<std::size_t>(index - 1)] -
                                   std::exp(node)));
  }
  return peak;
}

template <class Function> [[nodiscard]] double seconds_of(Function &&function) {
  constexpr int repeats = 4;
  const auto start = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < repeats; ++repeat) function();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

struct PreparedLine {
  LineBlock block;
  std::vector<double> gathered;
};

[[nodiscard]] PreparedLine
prepare_line(std::span<const double> local_forcing, int degree, MPI_Comm comm,
             std::span<const double> ends, const char *degree_message,
             const char *degree_disagree, const char *end_disagree,
             const char *span_message) {
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  const int count = degree + 1;
  int invalid = degree < 1 || count < ranks;
  for (const double end : ends) invalid |= !std::isfinite(end);
  require_collective(invalid, comm, degree_message);
  int lo = degree;
  int hi = degree;
  MPI_Allreduce(&degree, &lo, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&degree, &hi, 1, MPI_INT, MPI_MAX, comm);
  require_collective(lo != hi, comm, degree_disagree);
  std::vector<double> end_lo(ends.begin(), ends.end());
  auto end_hi = end_lo;
  if (!end_lo.empty()) {
    const int end_count = static_cast<int>(end_lo.size());
    MPI_Allreduce(MPI_IN_PLACE, end_lo.data(), end_count, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, end_hi.data(), end_count, MPI_DOUBLE, MPI_MAX, comm);
  }
  int mismatch = 0;
  for (std::size_t index = 0; index < end_lo.size(); ++index) {
    mismatch |= end_lo[index] != end_hi[index];
  }
  require_collective(mismatch, comm, end_disagree);
  const auto block = line_block(count, rank, ranks);
  int bad = static_cast<int>(local_forcing.size()) != block.count;
  for (const double sample : local_forcing) bad |= !std::isfinite(sample);
  require_collective(bad, comm, span_message);
  std::vector<int> counts(static_cast<std::size_t>(ranks));
  std::vector<int> displs(static_cast<std::size_t>(ranks));
  for (int peer = 0; peer < ranks; ++peer) {
    const auto peer_block = line_block(count, peer, ranks);
    counts[static_cast<std::size_t>(peer)] = peer_block.count;
    displs[static_cast<std::size_t>(peer)] = peer_block.offset;
  }
  std::vector<double> gathered(static_cast<std::size_t>(count));
  MPI_Allgatherv(local_forcing.data(), block.count, MPI_DOUBLE, gathered.data(),
                 counts.data(), displs.data(), MPI_DOUBLE, comm);
  return {block, std::move(gathered)};
}

[[nodiscard]] std::vector<double> owned_of(const std::vector<double> &solved,
                                           LineBlock block) {
  return {solved.begin() + block.offset,
          solved.begin() + block.offset + block.count};
}
} // namespace

std::vector<double>
distributed_chebyshev_dirichlet_poisson(std::span<const double> local_forcing,
                                        int degree, double value_at_plus,
                                        double value_at_minus, MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("distributed Chebyshev Poisson: null communicator");
  }
  const double ends[] = {value_at_plus, value_at_minus};
  const auto line = prepare_line(
      local_forcing, degree, comm, ends,
      "distributed Chebyshev Poisson: unusable degree",
      "distributed Chebyshev Poisson: ranks disagree on the degree",
      "distributed Chebyshev Poisson: ranks disagree on the ends",
      "distributed Chebyshev Poisson: local forcing has the wrong span");
  const auto solved =
      chebyshev_dirichlet_poisson(line.gathered, value_at_plus, value_at_minus);
  return owned_of(solved, line.block);
}

std::vector<double>
distributed_chebyshev_neumann_poisson(std::span<const double> local_forcing,
                                      int degree, double slope_at_plus,
                                      double slope_at_minus, MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("distributed Chebyshev Neumann: null communicator");
  }
  const double ends[] = {slope_at_plus, slope_at_minus};
  const auto line = prepare_line(
      local_forcing, degree, comm, ends,
      "distributed Chebyshev Neumann: unusable degree",
      "distributed Chebyshev Neumann: ranks disagree on the degree",
      "distributed Chebyshev Neumann: ranks disagree on the slopes",
      "distributed Chebyshev Neumann: local forcing has the wrong span");
  const auto solved =
      chebyshev_neumann_poisson(line.gathered, slope_at_plus, slope_at_minus);
  return owned_of(solved, line.block);
}

std::vector<double> distributed_chebyshev_robin_poisson(
    std::span<const double> local_forcing, int degree, double value_weight_plus,
    double slope_weight_plus, double data_plus, double value_weight_minus,
    double slope_weight_minus, double data_minus, MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("distributed Chebyshev Robin: null communicator");
  }
  const double ends[] = {value_weight_plus,  slope_weight_plus,  data_plus,
                         value_weight_minus, slope_weight_minus, data_minus};
  const auto line =
      prepare_line(local_forcing, degree, comm, ends,
                   "distributed Chebyshev Robin: unusable degree",
                   "distributed Chebyshev Robin: ranks disagree on the degree",
                   "distributed Chebyshev Robin: ranks disagree on the ends",
                   "distributed Chebyshev Robin: local forcing has the wrong span");
  const auto solved = chebyshev_robin_poisson(
      line.gathered, value_weight_plus, slope_weight_plus, data_plus,
      value_weight_minus, slope_weight_minus, data_minus);
  return owned_of(solved, line.block);
}

BoundedAxisComparison compare_bounded_axis(int degree, MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("bounded-axis comparison: null communicator");
  }
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  const int points = degree + 1;
  const int extended = 2 * degree;
  int invalid = degree < 16 || degree > 64 || points < ranks || extended < ranks;
  require_collective(invalid, comm,
                     "bounded-axis comparison: degree is outside 16..64");
  int lo = degree;
  int hi = degree;
  MPI_Allreduce(&degree, &lo, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&degree, &hi, 1, MPI_INT, MPI_MAX, comm);
  require_collective(lo != hi, comm, "bounded-axis comparison: ranks disagree");

  const auto nodes = chebyshev_lobatto(degree);
  std::vector<double> forcing(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) forcing[i] = std::exp(nodes[i]);
  const double at_plus = std::exp(1.0);
  const double at_minus = std::exp(-1.0);
  std::vector<double> solved;
  const double chebyshev_seconds = seconds_of(
      [&] { solved = chebyshev_dirichlet_poisson(forcing, at_plus, at_minus); });
  double chebyshev_error = 0;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    chebyshev_error =
        std::max(chebyshev_error, std::abs(solved[i] - std::exp(nodes[i])));
  }

  const auto owned = line_block(points, rank, ranks);
  const std::span<const double> local(forcing.data() + owned.offset, owned.count);
  double distributed_poisson_seconds = 0;
  distributed_poisson_seconds = seconds_of([&] {
    (void)distributed_chebyshev_dirichlet_poisson(local, degree, at_plus, at_minus,
                                                  comm);
  });

  double fd4_error = 0;
  double fd8_error = 0;
  const double fd4_seconds =
      seconds_of([&] { fd4_error = uniform_dirichlet_error<4>(degree); });
  const double fd8_seconds =
      seconds_of([&] { fd8_error = uniform_dirichlet_error<8>(degree); });

  if (solved.size() != nodes.size()) {
    throw std::runtime_error("bounded axis: Chebyshev solve changed the grid");
  }
  std::vector<std::complex<double>> extension(static_cast<std::size_t>(extended));
  extension[0] = std::complex<double>(solved[0], 0.0);
  for (int j = 1; j < degree; ++j) {
    const double sample = solved[static_cast<std::size_t>(j)];
    extension[static_cast<std::size_t>(j)] = std::complex<double>(sample, 0.0);
    extension[static_cast<std::size_t>(extended - j)] =
        std::complex<double>(sample, 0.0);
  }
  extension[static_cast<std::size_t>(degree)] =
      std::complex<double>(solved[static_cast<std::size_t>(degree)], 0.0);
  const auto line = line_block(extended, rank, ranks);
  std::vector<std::complex<double>> local_in(
      extension.begin() + line.offset, extension.begin() + line.offset + line.count);
  std::vector<std::complex<double>> spectrum(local_in.size());
  std::vector<std::complex<double>> back(local_in.size());
  DistributedComplexLineFFT distributed(extended, comm);
  double roundtrip = 0;
  const double distributed_line_seconds = seconds_of([&] {
    distributed.forward(local_in, spectrum);
    distributed.backward(spectrum, back);
  });
  for (std::size_t i = 0; i < local_in.size(); ++i) {
    roundtrip = std::max(
        roundtrip, std::abs(back[i] / static_cast<double>(extended) - local_in[i]));
  }

  std::vector<std::complex<double>> serial_out(extension.size());
  ComplexLineFFT serial(extended);
  const double local_line_seconds =
      seconds_of([&] { serial.forward(extension, serial_out); });

  BoundedAxisComparison result;
  result.ranks = ranks;
  result.points = points;
  result.chebyshev_error = max_rank(chebyshev_error, comm);
  result.fd4_error = max_rank(fd4_error, comm);
  result.fd8_error = max_rank(fd8_error, comm);
  result.chebyshev_seconds = max_rank(chebyshev_seconds, comm);
  result.distributed_poisson_seconds = max_rank(distributed_poisson_seconds, comm);
  result.local_line_seconds = max_rank(local_line_seconds, comm);
  result.distributed_line_seconds = max_rank(distributed_line_seconds, comm);
  result.fd4_seconds = max_rank(fd4_seconds, comm);
  result.fd8_seconds = max_rank(fd8_seconds, comm);
  result.distributed_workspace_bytes = static_cast<std::size_t>(
      max_rank(static_cast<double>(distributed.workspace_bytes()), comm));
  result.distributed_comm_bytes = static_cast<std::size_t>(
      max_rank(static_cast<double>(distributed.comm_bytes()), comm));
  const auto unknowns = static_cast<std::size_t>(degree - 1);
  result.fd_matrix_bytes = unknowns * unknowns * sizeof(double);
  result.distributed_roundtrip_peak = max_rank(roundtrip, comm);
  result.prefer_chebyshev = result.chebyshev_error < result.fd4_error &&
                            result.chebyshev_error < result.fd8_error;
  return result;
}

} // namespace pfc::fft
