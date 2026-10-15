// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file microelasticity_kernels.cu
 * @brief CUDA translation unit of the microelasticity device kernels.
 */

#if !defined(OpenPFC_ENABLE_CUDA)
#error "microelasticity_kernels.cu requires OpenPFC_ENABLE_CUDA"
#endif

#include "microelasticity_kernels_gpu.inc"
