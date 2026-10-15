// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file microelasticity_kernels.hip
 * @brief HIP translation unit of the microelasticity device kernels.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "microelasticity_kernels.hip requires OpenPFC_ENABLE_HIP"
#endif

#include "microelasticity_kernels_gpu.inc"
