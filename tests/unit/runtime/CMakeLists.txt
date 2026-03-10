# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    add_subdirectory(gpu)
endif()
