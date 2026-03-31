// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_layout.hpp
 * @brief FFT box layout (split from fft.hpp for lighter includes)
 */

#pragma once

#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

#include <heffte.h>
#include <vector>

namespace pfc::fft::layout {

using box3di = heffte::box3d<int>;
using Decomposition = pfc::decomposition::Decomposition;
using pfc::types::Int3;

struct FFTLayout {
  const Decomposition m_decomposition;
  const int m_r2c_direction = 0;
  const std::vector<heffte::box3d<int>> m_real_boxes;
  const std::vector<heffte::box3d<int>> m_complex_boxes;
};

const FFTLayout create(const Decomposition &decomposition, int r2c_direction);

inline const auto &get_real_box(const FFTLayout &layout, int i) {
  return layout.m_real_boxes.at(i);
}

inline const auto &get_complex_box(const FFTLayout &layout, int i) {
  return layout.m_complex_boxes.at(i);
}

inline auto get_r2c_direction(const FFTLayout &layout) {
  return layout.m_r2c_direction;
}

} // namespace pfc::fft::layout
