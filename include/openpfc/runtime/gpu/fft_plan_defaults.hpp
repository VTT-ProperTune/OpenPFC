// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fft_plan_defaults.hpp
 * @brief Default heFFTe plan options for a device FFT backend.
 *
 * @details
 * heFFTe's `default_options<cufft|rocfft>()` always set `use_gpu_aware`,
 * so its reshapes hand device pointers to MPI whatever OpenPFC decided
 * about the MPI library. Every default device plan in OpenPFC starts here
 * instead: `use_gpu_aware` follows `pfc::gpu::runtime_mpi_gpu_aware()`,
 * the same decision the device halo exchanges use. A caller or JSON
 * `use_gpu_aware` applied afterwards still wins.
 */

#include <heffte.h>

#include <openpfc/runtime/gpu/gpu_aware_mpi.hpp>

namespace pfc::fft {

template <class Backend>
[[nodiscard]] inline heffte::plan_options default_device_plan_options() {
  heffte::plan_options options = heffte::default_options<Backend>();
  options.use_gpu_aware = pfc::gpu::runtime_mpi_gpu_aware();
  return options;
}

} // namespace pfc::fft
