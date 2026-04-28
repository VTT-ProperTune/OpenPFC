// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_fftw.hpp
 * @brief Default CPU FFT backend (HeFFTe + FFTW) and `pfc::FFT` alias
 *
 * @details
 * Pulls in HeFFTe. Prefer including only `<openpfc/kernel/fft/fft.hpp>` when you
 * only need `IFFT`, layout, or k-space helpers without HeFFTe on the include path.
 */

#pragma once

#include <heffte.h>
#include <openpfc/kernel/fft/detail/fft_heffte_backend.hpp>
#include <openpfc/kernel/fft/fft.hpp>

namespace pfc::fft {

class CpuFft : public FFT_Impl<heffte::backend::fftw> {
public:
  using FFT_Impl<heffte::backend::fftw>::FFT_Impl;
};

using plan_options = heffte::plan_options;

} // namespace pfc::fft

namespace pfc {
using FFT = fft::CpuFft;
}
