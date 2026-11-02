// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fft_selection.hpp
 * @brief Automatic single-rank FFT implementation (#508).
 *
 * The decision uses the communicator size and whether that rank owns the
 * whole domain. It does not look at how many devices are visible.
 *
 * Admitted layout is the existing double-precision real-to-complex
 * transform with x fastest. Float and any other layout stay on the
 * explicit path. More than one rank, or a rank that does not own the
 * whole domain, stays on HeFFTe. An explicit `heffte` or `local` choice
 * is applied by the caller and overrides this function.
 *
 * Direct rocFFT is admitted for every positive grid on one HIP rank.
 * Measured cubic, anisotropic, odd/prime and singleton grids were faster
 * there once plan setup was included, and they used less device memory.
 * FFTW and cuFFT stay on HeFFTe: on at least one of those classes the
 * direct plan lost on plan setup or on the round trip, and the memory
 * comparison did not favour it. Campaign figures stay outside this header.
 */

namespace pfc::fft {

/// Hardware family of the vendor plan. One result does not transfer.
enum class FftHardware { cpu, cuda, hip };

/// What automatic selection may return. `local` is the direct vendor plan.
enum class AutomaticFft { heffte, local };

/// Inputs for `automatic_single_rank_fft`. Extents are the global grid.
struct SingleRankFftQuery {
  FftHardware hardware;
  int communicator_size;
  bool owns_complete_domain;
  int nx;
  int ny;
  int nz;
};

/**
 * @brief Choose HeFFTe or the direct vendor plan for one omitted `fft` key.
 * @return `local` only for one HIP rank that owns a positive complete grid.
 */
[[nodiscard]] inline AutomaticFft
automatic_single_rank_fft(SingleRankFftQuery query) noexcept {
  if (query.communicator_size != 1 || !query.owns_complete_domain)
    return AutomaticFft::heffte;
  if (query.nx < 1 || query.ny < 1 || query.nz < 1) return AutomaticFft::heffte;
  if (query.hardware == FftHardware::hip) return AutomaticFft::local;
  return AutomaticFft::heffte;
}

} // namespace pfc::fft
