// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/runtime/common/mpi_main.hpp>
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#endif

namespace pfc::runtime::detail {
void bind_local_device_before_mpi() {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
  // Keep the existing binding and no-visible-device behavior unchanged.
  gpu::bind_local_device_before_mpi();
#endif
}
} // namespace pfc::runtime::detail
