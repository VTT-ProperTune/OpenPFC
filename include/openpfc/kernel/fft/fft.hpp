// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft.hpp
 * @brief Fast Fourier Transform API for spectral methods
 *
 * @details
 * Public entry point for distributed FFTs. Core interface types live in
 * fft_interface.hpp (no HeFFTe). The HeFFTe-backed implementation template
 * `FFT_Impl` is provided by detail/fft_heffte_backend.hpp, included from here so
 * existing `#include <openpfc/kernel/fft/fft.hpp>` code keeps working.
 *
 * @see fft_interface.hpp for IFFT and buffer aliases
 * @see detail/fft_heffte_backend.hpp for FFT_Impl and HeFFTe types
 */

#pragma once

#include <openpfc/kernel/fft/detail/fft_heffte_backend.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>
#include <openpfc/kernel/fft/kspace.hpp>

#include <memory>

namespace pfc {
namespace fft {

using Decomposition = pfc::decomposition::Decomposition;

using FFT = FFT_Impl<heffte::backend::fftw>;

using heffte::plan_options;
using layout::FFTLayout;

FFT create(const FFTLayout &fft_layout, int rank_id, plan_options options);

FFT create(const Decomposition &decomposition, int rank_id);

FFT create(const Decomposition &decomposition);

std::unique_ptr<IFFT> create_with_backend(const FFTLayout &fft_layout, int rank_id,
                                          plan_options options, Backend backend);

std::unique_ptr<IFFT> create_with_backend(const Decomposition &decomposition,
                                          int rank_id, Backend backend);

} // namespace fft

using FFT = fft::FFT;
using FFTLayout = fft::layout::FFTLayout;

} // namespace pfc
