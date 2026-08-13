# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Installation rules for headers, libraries, and binaries

# Install public headers only. Device TUs live under src/openpfc/runtime/gpu/;
# kernel .inc files live next to those TUs under src/ and are not installed.
# Stray .md under include/ must not ship.
# FetchContent nlohmann_json is a build-time dependency only — do not dump its
# headers into the prefix. Consumers that include JSON-using public headers
# get nlohmann_json via find_dependency in OpenPFCConfig.cmake.
install(DIRECTORY include/openpfc DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

# Install library binary
install(TARGETS openpfc
    EXPORT OpenPFCTargets
    ARCHIVE DESTINATION lib   # .a files
    LIBRARY DESTINATION lib   # .so files
    RUNTIME DESTINATION bin   # executable files (not needed now but future proof)
)

# Install GPU kernel library if CUDA is enabled
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    install(TARGETS openpfc_gpu_kernels
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

# Install HIP kernel library if HIP is enabled (audit 11 / PM: this block was
# missing, so install(EXPORT OpenPFCTargets) failed or HIP installs shipped
# without libopenpfc_hip_kernels -- mirror the CUDA block above).
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    install(TARGETS openpfc_hip_kernels
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()
