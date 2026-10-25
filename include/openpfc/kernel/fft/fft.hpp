// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft.hpp
 * @brief Fast Fourier Transform API for spectral methods (no HeFFTe headers)
 *
 * @details
 * Core layout and `IHostFFT` / `IDeviceFFT` / `IFFTQueries` live in
 * fft_interface.hpp / fft_layout.hpp / kspace.hpp. The default CPU backend `CPUFFT`
 * (HeFFTe + FFTW) is declared here and defined in fft_fftw.hpp — include that header
 * (or openpfc.hpp / openpfc_minimal.hpp) when you need the complete type,
 * `pfc::FFT`, or `heffte::plan_options` values.
 *
 * @see fft_fftw.hpp for CPUFFT, plan_options alias, and HeFFTe-backed factories
 * @see fft_interface.hpp for IHostFFT, IDeviceFFT, and buffer aliases
 */

#pragma once

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>
#include <openpfc/kernel/fft/kspace.hpp>

#include <memory>
#include <mpi.h>
#include <type_traits>

namespace heffte {
struct plan_options;
}

namespace pfc {
namespace fft {

using Decomposition = pfc::decomposition::Decomposition;
using layout::FFTLayout;

/** HeFFTe+FFTW backend; complete definition in fft_fftw.hpp */
class CPUFFT;

/**
 * @name Collective host FFT factories
 * Every rank in comm must create the same plan in the same call order.
 * The decomposition/layout must contain exactly MPI_Comm_size(comm) boxes;
 * rank_id is the caller's rank in comm (not a world rank or an arbitrary box).
 * r2c_direction is the reduced real-transform axis: 0 (x), 1 (y), or 2 (z).
 * Layouts own their geometry; the source decomposition need not outlive a plan.
 * The communicator must remain valid while the plan is used.
 *
 * @throws std::logic_error If communicator and decomposition sizes differ,
 *         or if the reduced axis is outside 0..2.
 * @throws std::invalid_argument If rank_id differs from MPI_Comm_rank(comm).
 * Input errors are checked locally: callers must agree inputs before invoking
 * a collective factory; a local exception does not perform an all-rank vote.
 * @note OPENPFC_FFT_LOG_LAYOUT enables rank-zero layout logging when set.
 * @{
 */

/** @brief Create the caller's real/complex-box plan using explicit options. */
[[nodiscard]] CPUFFT create(const FFTLayout &fft_layout, int rank_id,
                            const heffte::plan_options &options,
                            MPI_Comm comm = MPI_COMM_WORLD);

/** @brief Derive a layout using the explicit reduced axis and plan options. */
[[nodiscard]] CPUFFT create(const Decomposition &decomposition, int rank_id,
                            MPI_Comm comm, int r2c_direction,
                            const heffte::plan_options &options);

/** @brief Derive a layout using FFTW default plan options. */
[[nodiscard]] CPUFFT create(const Decomposition &decomposition, int rank_id,
                            MPI_Comm comm = MPI_COMM_WORLD, int r2c_direction = 0);

/** @brief Query the communicator rank and create an x-reduced default plan. */
[[nodiscard]] CPUFFT create(const Decomposition &decomposition,
                            MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Create a type-erased host FFT with Backend::FFTW.
 * @throws std::invalid_argument For Backend::CUDA or Backend::HIP: use the
 *         corresponding device factory (create_cuda/create_hip) instead.
 */
[[nodiscard]] std::unique_ptr<IHostFFT>
create_with_backend(const FFTLayout &fft_layout, int rank_id,
                    const heffte::plan_options &options, Backend backend,
                    MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Create a type-erased host FFT with Backend::FFTW.
 * @throws std::invalid_argument For Backend::CUDA or Backend::HIP: use the
 *         corresponding device factory (create_cuda/create_hip) instead.
 */
[[nodiscard]] std::unique_ptr<IHostFFT>
create_with_backend(const Decomposition &decomposition, int rank_id, Backend backend,
                    MPI_Comm comm = MPI_COMM_WORLD, int r2c_direction = 0);

/** @} */

} // namespace fft

using FFTLayout = fft::layout::FFTLayout;

} // namespace pfc
