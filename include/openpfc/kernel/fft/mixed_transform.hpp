// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file mixed_transform.hpp
 * @brief Periodic x/y and bounded uniform z, using existing heFFTe reshapes.
 */

#include <complex>
#include <mpi.h>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/real_transform.hpp>

namespace pfc::fft {

/** @brief Planned double-precision host mixed transform.
 * All ranks supply the same geometry and a live communicator. Construct,
 * execute and destroy before MPI_Finalize. Collectively plans a heFFTe
 * batched 2D r2c FFT; z transforms stay rank-local. The y slab is partitioned
 * by heFFTe, x/z are complete. At most ny ranks are supported.
 * Physical data are x-fastest, then local y, then z. Spectral data have
 * x=0..nx/2, local y, and all z modes. Forward is raw FFTW DCT/DST times
 * unscaled Fourier; backward normalizes by nx*ny*mirrored_size(z).
 * No host/device transfers. One instance is not reentrant. Buffer extents
 * must be correct on every rank before entering collective execution.
 */
class MixedTransform {
public:
  MixedTransform(int nx, int ny, double period_x, double period_y, BoundedAxis z,
                 MPI_Comm comm = MPI_COMM_WORLD);
  ~MixedTransform();
  MixedTransform(MixedTransform &&) noexcept;
  MixedTransform &operator=(MixedTransform &&) noexcept;
  MixedTransform(const MixedTransform &) = delete;
  MixedTransform &operator=(const MixedTransform &) = delete;

  [[nodiscard]] Box3i inbox() const;
  [[nodiscard]] Box3i outbox() const;
  [[nodiscard]] std::size_t size_inbox() const;
  [[nodiscard]] std::size_t size_outbox() const;
  /// Owned arrays only: excludes opaque FFTW/heFFTe plan allocations.
  [[nodiscard]] std::size_t buffer_bytes() const;
  void forward(std::span<const double> in, std::span<std::complex<double>> out);
  void backward(std::span<const std::complex<double>> in, std::span<double> out);
  /// Positive eigenvalue of -Delta for a global outbox index.
  [[nodiscard]] double eigenvalue(std::array<int, 3> mode,
                                  LaplacianSymbol symbol) const;
  void laplacian(std::span<const double> in, std::span<double> out,
                 LaplacianSymbol symbol = LaplacianSymbol::Spectral);
  /// Solve -Delta u=f with homogeneous z boundaries. Reject incompatible
  /// Neumann mean collectively, and set the solution's weighted mean to zero.
  void poisson(std::span<const double> forcing, std::span<double> solution,
               LaplacianSymbol symbol = LaplacianSymbol::Spectral,
               double compatibility_tolerance = 1e-12);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace pfc::fft
