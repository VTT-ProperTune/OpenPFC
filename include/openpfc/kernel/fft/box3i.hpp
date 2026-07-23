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

#include <openpfc/kernel/data/box3i.hpp>

namespace pfc::fft {

/// @brief Alias of the canonical `pfc::Box3i` (M1). Kept so existing
/// `pfc::fft::Box3i` / `fft::get_inbox`/`get_outbox` call sites are unchanged.
using Box3i = pfc::Box3i;

} // namespace pfc::fft
