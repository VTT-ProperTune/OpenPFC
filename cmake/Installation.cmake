# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Installation rules for headers, libraries, and binaries

# Install nlohmann_json headers, but only if nlohmann_json_SOURCE_DIR is
# defined, i.e. the package is built from source during the configure step.
# This is to avoid installing the headers if the package is installed from
# a system wide package manager.
if(DEFINED nlohmann_json_SOURCE_DIR)
  message(STATUS "Installing nlohmann_json headers")
  install(DIRECTORY ${nlohmann_json_SOURCE_DIR}/include/nlohmann
          DESTINATION include
  )
endif()

# Install headers
# Install public headers only (audit 11 / PM): the tree also contains .cu/.hip
# device sources and stray .md files that must not be shipped into the include
# prefix. (Relocating those sources under src/ is deferred to M3.)
install(DIRECTORY include/openpfc DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

# Install library binary
install(TARGETS openpfc
    EXPORT OpenPFCTargets
    ARCHIVE DESTINATION lib   # .a files
    LIBRARY DESTINATION lib   # .so files
    RUNTIME DESTINATION bin   # executable files (not needed now but future proof)
)

# FetchContent nlohmann_json is linked to openpfc; CMake requires it in the same
# export set when installing OpenPFCTargets. System nlohmann_json has no installable
# target (only nlohmann_json::nlohmann_json), so gate on FetchContent.
if(DEFINED nlohmann_json_SOURCE_DIR AND TARGET nlohmann_json)
  install(TARGETS nlohmann_json EXPORT OpenPFCTargets
    INCLUDES DESTINATION include)
endif()

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
