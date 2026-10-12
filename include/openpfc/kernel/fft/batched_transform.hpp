// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <complex>
#include <mpi.h>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/real_transform.hpp>

namespace pfc {
struct CUDASpace;
struct HIPSpace;
} // namespace pfc
namespace heffte {
enum class reshape_algorithm;
} // namespace heffte

namespace pfc::fft {
/// Ownership of mixed-transform spectral coefficients. Physical data always
/// uses y slabs with complete bounded-z lines. Exposed coefficients are
/// canonical x-fast regardless of the internal FFT memory order.
enum class SpectralSlab { Y, X };
/** Planned device-resident DCT-II/DST-II batches, with FFTW raw forward and
 * normalized inverse. Only CellCentered is supported. Sample (batch,j) is
 * at batch*distance+j*stride; positive nonoverlapping strides are required.
 * Buffer spans have extent size_buffer(), including untouched padding.
 * Both inputs are preserved; identical input/output spans are supported.
 * Plans and workspace are reused. Execution uses the legacy default stream;
 * synchronize that stream before using results elsewhere. Not reentrant.
 * Construct, execute and destroy on the same current device. MemorySpace is
 * CUDASpace or HIPSpace, with the respective spectral build enabled.
 */
template <class MemorySpace> class BatchedRealTransform {
public:
  BatchedRealTransform(BoundedAxis axis, int batches, int stride, int distance);
  ~BatchedRealTransform();
  BatchedRealTransform(BatchedRealTransform &&) noexcept;
  BatchedRealTransform &operator=(BatchedRealTransform &&) noexcept;
  BatchedRealTransform(const BatchedRealTransform &) = delete;
  BatchedRealTransform &operator=(const BatchedRealTransform &) = delete;
  [[nodiscard]] const BoundedAxis &axis() const;
  [[nodiscard]] std::size_t size_buffer() const;
  /// Owned device arrays including vendor FFT workspace, excluding plans.
  [[nodiscard]] std::size_t buffer_bytes() const;
  void forward(std::span<const double> in, std::span<double> out);
  void backward(std::span<const double> in, std::span<double> out);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

/** Periodic x/y, cell-centered DCT-II or DST-II in z, double precision.
 * Physical geometry and raw coefficient conventions match MixedTransform.
 * Physical x/z lines are complete; y slabs permit at most ny ranks. Spectral
 * ownership defaults to y slabs, with an explicit x-slab alternative below.
 * The communicator is borrowed:
 * all ranks construct/execute/destroy collectively before MPI_Finalize.
 * One rank uses a vendor half-volume 3D FFT; multiple ranks use existing
 * heFFTe reshapes, not a separate transpose implementation. Forward exposes
 * actual DCT/DST coefficients times unscaled Fourier coefficients; backward
 * preserves them and normalizes. Solves fuse paired-mode diagonalization.
 * All fields and workspaces stay on device. Neumann Poisson transfers only
 * three compatibility scalars to the host (then a collective reduction).
 * Same device/default-stream/non-reentrancy contract as BatchedRealTransform.
 * Buffer extents must be correct on every rank before execution collectives.
 * Values must be finite. Only nonnegative Helmholtz reaction is supported.
 */
template <class MemorySpace> class BatchedMixedTransform {
public:
  BatchedMixedTransform(int nx, int ny, double period_x, double period_y,
                        BoundedAxis z, MPI_Comm comm = MPI_COMM_WORLD);
  /// Select an existing heFFTe exchange algorithm without changing the layout.
  /// All ranks must select the same valid algorithm. The original overload
  /// retains alltoallv. On one rank the policy is validated but no MPI reshape
  /// is needed. GPU-aware versus host-staged execution remains a build policy.
  BatchedMixedTransform(int nx, int ny, double period_x, double period_y,
                        BoundedAxis z, heffte::reshape_algorithm algorithm,
                        MPI_Comm comm = MPI_COMM_WORLD);
  /// Opt into full-z spectral x slabs. Distributed operators retain heFFTe's
  /// y-fast memory order; forward/backward still expose canonical coefficients.
  /// All ranks must agree. X requires at least one R2C x coefficient per rank.
  /// One-rank execution keeps the original vendor plan and storage order.
  BatchedMixedTransform(int nx, int ny, double period_x, double period_y,
                        BoundedAxis z, heffte::reshape_algorithm algorithm,
                        SpectralSlab spectral_slab, MPI_Comm comm = MPI_COMM_WORLD);
  ~BatchedMixedTransform();
  BatchedMixedTransform(BatchedMixedTransform &&) noexcept;
  BatchedMixedTransform &operator=(BatchedMixedTransform &&) noexcept;
  BatchedMixedTransform(const BatchedMixedTransform &) = delete;
  BatchedMixedTransform &operator=(const BatchedMixedTransform &) = delete;
  [[nodiscard]] Box3i inbox() const;
  [[nodiscard]] Box3i outbox() const;
  [[nodiscard]] std::size_t size_inbox() const;
  [[nodiscard]] std::size_t size_outbox() const;
  [[nodiscard]] std::size_t buffer_bytes() const;
  [[nodiscard]] double eigenvalue(std::array<int, 3> mode,
                                  LaplacianSymbol symbol) const;
  void forward(std::span<const double> in, std::span<std::complex<double>> out);
  void backward(std::span<const std::complex<double>> in, std::span<double> out);
  void laplacian(std::span<const double> in, std::span<double> out,
                 LaplacianSymbol symbol = LaplacianSymbol::Spectral);
  void poisson(std::span<const double> in, std::span<double> out,
               LaplacianSymbol symbol = LaplacianSymbol::Spectral,
               double compatibility_tolerance = 1e-12);
  /// Solve (-Delta + reaction)u=f. Zero reaction has the Poisson contract.
  void helmholtz(std::span<const double> in, std::span<double> out, double reaction,
                 LaplacianSymbol symbol = LaplacianSymbol::Spectral,
                 double compatibility_tolerance = 1e-12);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};
} // namespace pfc::fft
