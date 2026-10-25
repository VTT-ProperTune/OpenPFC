// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_layout.hpp
 * @brief FFT box layout (split from fft.hpp for lighter includes)
 *
 * @details
 * `create(decomp, r2c, options)` evaluates the complex-outbox selector
 * with the same HeFFTe `plan_options` the FFT instance will use. The
 * two-argument overload uses FFTW defaults and is the CPU convenience
 * path; GPU factories must pass backend options explicitly.
 */

#pragma once

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/box3i.hpp>

#include <vector>

namespace heffte {
struct plan_options;
}

namespace pfc::fft::layout {

using Decomposition = pfc::decomposition::Decomposition;
using pfc::types::Int3;

struct FFTLayout {
  const Decomposition m_decomposition;
  const int m_r2c_direction = 0;
  const std::vector<Box3i> m_real_boxes;
  const std::vector<Box3i> m_complex_boxes;
  const Int3 m_real_proc_grid{};
  const Int3 m_complex_proc_grid{};
};

[[nodiscard]] FFTLayout create(const Decomposition &decomposition,
                               int r2c_direction);

[[nodiscard]] FFTLayout create(const Decomposition &decomposition, int r2c_direction,
                               const heffte::plan_options &options);

inline const Box3i &get_real_box(const FFTLayout &layout, int i) {
  return layout.m_real_boxes.at(i);
}

inline const Box3i &get_complex_box(const FFTLayout &layout, int i) {
  return layout.m_complex_boxes.at(i);
}

inline auto get_r2c_direction(const FFTLayout &layout) {
  return layout.m_r2c_direction;
}

inline const Int3 &get_real_proc_grid(const FFTLayout &layout) {
  return layout.m_real_proc_grid;
}

inline const Int3 &get_complex_proc_grid(const FFTLayout &layout) {
  return layout.m_complex_proc_grid;
}

} // namespace pfc::fft::layout
