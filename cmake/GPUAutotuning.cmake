# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# GPU kernel auto-tuning support

option(OpenPFC_ENABLE_GPU_AUTOTUNING "Enable GPU kernel auto-tuning infrastructure" OFF)

if(OpenPFC_ENABLE_GPU_AUTOTUNING)
    # Host callbacks exercise tuning/cache behavior without a device. GPU
    # builds additionally synchronize their backend around each timed batch.
    if(NOT OpenPFC_ENABLE_CUDA AND NOT OpenPFC_ENABLE_HIP)
        message(STATUS "GPU autotuning enabled for host callbacks (no device backend)")
    else()
        # OpenPFC_ENABLE_GPU_AUTOTUNING is a PUBLIC usage requirement on
        # openpfc (and kernel libs) in LibraryConfiguration.cmake.
        message(STATUS "✅ GPU kernel auto-tuning enabled")
    endif()
else()
    message(STATUS "GPU kernel auto-tuning disabled (use -DOpenPFC_ENABLE_GPU_AUTOTUNING=ON to enable)")
endif()
