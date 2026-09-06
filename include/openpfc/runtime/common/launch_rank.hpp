// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file launch_rank.hpp
 * @brief Node-local rank from the launcher environment (before MPI_Init).
 *
 * Cray MPICH's `MPICH_OFI_NIC_POLICY=GPU` snapshots the current HIP/CUDA
 * device at `MPI_Init`. Binding after Init with every GCD visible made every
 * rank look like device 0, so all ranks shared NIC 0 (16-GCD 768³ ~375 ms).
 * Pin the device from `SLURM_LOCALID` first; leave every GCD visible so
 * GPU-aware MPI can still use intra-node IPC.
 */

#include <cstdlib>

namespace pfc::runtime {

/// Node-local rank from Slurm / PMI / Open MPI, or -1 if unset / invalid.
[[nodiscard]] inline int local_rank_from_launch_env() noexcept {
  static constexpr const char *kKeys[] = {
      "SLURM_LOCALID",
      "OMPI_COMM_WORLD_LOCAL_RANK",
      "MPI_LOCALRANKID",
      "PALS_LOCAL_RANKID",
  };
  for (const char *key : kKeys) {
    const char *v = std::getenv(key);
    if (v == nullptr || v[0] == '\0') {
      continue;
    }
    char *end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && n >= 0 && n < 1024) {
      return static_cast<int>(n);
    }
  }
  return -1;
}

} // namespace pfc::runtime
