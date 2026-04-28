// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft.hpp
 * @brief Fast Fourier Transform API for spectral methods (no HeFFTe headers)
 *
 * @details
 * Core layout and `IFFT` live in fft_interface.hpp / fft_layout.hpp / kspace.hpp.
 * The default CPU backend `CpuFft` (HeFFTe + FFTW) is declared here and defined in
 * fft_fftw.hpp — include that header (or openpfc.hpp / openpfc_minimal.hpp) when you
 * need the complete type, `pfc::FFT`, or `heffte::plan_options` values.
 *
 * @see fft_fftw.hpp for CpuFft, plan_options alias, and HeFFTe-backed factories
 * @see fft_interface.hpp for IFFT and buffer aliases
 */

#pragma once

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>
#include <openpfc/kernel/fft/kspace.hpp>

#include <memory>
#include <mpi.h>

namespace heffte {
struct plan_options;
}

namespace pfc {
namespace fft {

using Decomposition = pfc::decomposition::Decomposition;
using layout::FFTLayout;

/** HeFFTe+FFTW backend; complete definition in fft_fftw.hpp */
class CpuFft;

CpuFft create(const FFTLayout &fft_layout, int rank_id,
              const heffte::plan_options &options, MPI_Comm comm = MPI_COMM_WORLD);

CpuFft create(const Decomposition &decomposition, int rank_id,
              MPI_Comm comm = MPI_COMM_WORLD);

CpuFft create(const Decomposition &decomposition, MPI_Comm comm = MPI_COMM_WORLD);

std::unique_ptr<IFFT> create_with_backend(const FFTLayout &fft_layout, int rank_id,
                                          const heffte::plan_options &options,
                                          Backend backend,
                                          MPI_Comm comm = MPI_COMM_WORLD);

std::unique_ptr<IFFT> create_with_backend(const Decomposition &decomposition,
                                          int rank_id, Backend backend,
                                          MPI_Comm comm = MPI_COMM_WORLD);

} // namespace fft

using FFTLayout = fft::layout::FFTLayout;

} // namespace pfc
