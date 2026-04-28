// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file box3i.hpp
 * @brief Integer axis-aligned box for FFT inbox/outbox (HeFFTe-free public type)
 *
 * @details
 * Describes a 3D index box with inclusive low/high corners and per-axis span
 * sizes, matching the layout information historically carried by
 * `heffte::box3d<int>` in OpenPFC. This type intentionally omits HeFFTe-specific
 * fields (e.g. dimension ordering) so public headers do not depend on HeFFTe.
 */

#pragma once

#include <array>

namespace pfc::fft {

/**
 * @brief Inclusive integer index box with explicit per-axis sizes
 */
struct Box3i {
  std::array<int, 3> low{};
  std::array<int, 3> high{};
  std::array<int, 3> size{};
};

} // namespace pfc::fft
