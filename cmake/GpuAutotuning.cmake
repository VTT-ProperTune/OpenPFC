# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# GPU kernel auto-tuning support

option(OpenPFC_ENABLE_GPU_AUTOTUNING "Enable GPU kernel auto-tuning infrastructure" OFF)

if(OpenPFC_ENABLE_GPU_AUTOTUNING)
    # GPU autotuning requires CUDA or HIP to be available
    if(NOT OpenPFC_ENABLE_CUDA AND NOT OpenPFC_ENABLE_HIP)
        message(WARNING "⚠️  OpenPFC_ENABLE_GPU_AUTOTUNING=ON but neither CUDA nor HIP is enabled. GPU autotuning disabled.")
        set(OpenPFC_ENABLE_GPU_AUTOTUNING OFF)
    else()
        add_compile_definitions(OpenPFC_ENABLE_GPU_AUTOTUNING)
        message(STATUS "✅ GPU kernel auto-tuning enabled")
    endif()
else()
    message(STATUS "GPU kernel auto-tuning disabled (use -DOpenPFC_ENABLE_GPU_AUTOTUNING=ON to enable)")
endif()
