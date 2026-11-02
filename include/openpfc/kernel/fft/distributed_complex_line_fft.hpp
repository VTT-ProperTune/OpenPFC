// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file distributed_complex_line_fft.hpp
 * @brief MPI block distribution of the unnormalized complex line DFT.
 *
 * The mathematical result is `ComplexLineFFT` of the whole line.
 * `backward(forward(x))` equals `length * x`. One rank plans that line
 * with HeFFTe as a complex box whose other two axes have length one.
 * Several ranks cannot: HeFFTe refuses a process grid on those unit
 * axes. They gather the line and reuse `ComplexLineFFT`. This is not
 * `IHostFFT`.
 */

#include <complex>
#include <memory>
#include <mpi.h>
#include <span>

namespace pfc::fft {

/** @brief Contiguous block of a global line. Rank `r` owns
 * `offset` through `offset + count`. The first `length % ranks`
 * ranks own one extra sample.
 */
struct LineBlock {
  int offset = 0;
  int count = 0;
};

[[nodiscard]] inline LineBlock line_block(int length, int rank, int ranks) {
  const int base = length / ranks;
  const int extra = length % ranks;
  LineBlock block;
  block.count = base + (rank < extra ? 1 : 0);
  block.offset = rank * base + (rank < extra ? rank : extra);
  return block;
}

/** @brief Collective unnormalized complex DFT of one distributed line.
 *
 * Every rank passes the same length and a live communicator, and
 * destroys the plan before `MPI_Finalize`. A length shorter than 2, a
 * length shorter than the communicator, or disagreeing ranks are
 * rejected together. Spans must have `local_length()` entries. The
 * caller spans are preserved. One instance is not reentrant.
 */
class DistributedComplexLineFFT {
public:
  DistributedComplexLineFFT(int global_length, MPI_Comm comm);
  ~DistributedComplexLineFFT();
  DistributedComplexLineFFT(const DistributedComplexLineFFT &) = delete;
  DistributedComplexLineFFT &operator=(const DistributedComplexLineFFT &) = delete;
  DistributedComplexLineFFT(DistributedComplexLineFFT &&) noexcept;
  DistributedComplexLineFFT &operator=(DistributedComplexLineFFT &&) noexcept;

  void forward(std::span<const std::complex<double>> in,
               std::span<std::complex<double>> out);
  void backward(std::span<const std::complex<double>> in,
                std::span<std::complex<double>> out);

  [[nodiscard]] int global_length() const;
  [[nodiscard]] int local_length() const;
  [[nodiscard]] int local_offset() const;
  /// Transform scratch on this rank, in bytes. One rank reports the HeFFTe
  /// workspace. Several ranks report the gathered line and its image.
  [[nodiscard]] std::size_t workspace_bytes() const;
  /// Bytes moved for this rank. One rank reports the HeFFTe communication
  /// scratch. Several ranks report the samples this rank does not own.
  [[nodiscard]] std::size_t comm_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace pfc::fft
