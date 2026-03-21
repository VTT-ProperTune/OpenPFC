// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_policy.hpp
 * @brief Documentation-oriented enum: where ghost data lives vs FFT-safe core
 *
 * @details
 * Describes how a distributed real-space field relates to halo storage and
 * whether the primary buffer may be passed to HeFFTe unchanged. See
 * `docs/halo_exchange.md` (Halo policies) for the full table and invariants.
 *
 * This enum is for orchestration, documentation, and future APIs — not used in
 * inner stencil loops.
 */

#pragma once

namespace pfc {
namespace halo {

/**
 * @brief High-level halo layout / usage policy
 */
enum class HaloPolicy {
  /// Subdomain-sized buffer only; no halo MPI for this field.
  None,
  /// Ghosts in boundary slabs of the same nx×ny×nz array (`HaloExchanger`).
  /// Not FFT-safe on that buffer after exchange (multi-rank), in general.
  InPlace,
  /// Core nx×ny×nz for FFT; ghosts in separate face slabs
  /// (`SeparatedFaceHaloExchanger`).
  Separated,
  /// Explicit slow path: extra copies or side structures; same idea as
  /// Separated but caller accepts orchestration cost.
  MixedHybrid
};

} // namespace halo
} // namespace pfc
