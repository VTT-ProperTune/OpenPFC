// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kspace_iterator_gpu.hpp
 * @brief Device-callable k-space scalars for GPU kernels (M5).
 *
 * @details
 * Host `for_each_kpoint` stays in `kernel/fft/kspace_iterator.hpp` (it takes
 * a host callable). Device kernels should compute per-thread wavenumbers
 * with the `OPENPFC_HD` helpers in `kspace.hpp`:
 *
 * @code
 * const double kx = pfc::fft::kspace::k_component(i, Nx, fx);
 * const double kx_odd = pfc::fft::kspace::k_component_odd(i, Nx, fx);
 * @endcode
 *
 * This header pulls those helpers in for GPU TUs. There is no device
 * analogue of the host triple-loop iterator: a kernel already iterates
 * threads.
 */

#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
