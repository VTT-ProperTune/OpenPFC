# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_subdirectory(common)

# Header-only decide_gpu_aware_mpi() — no device required for assume/Cray env.
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/test_gpu_aware_mpi.cpp
)

if((OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE) OR
   (OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE))
    add_subdirectory(gpu)
endif()
