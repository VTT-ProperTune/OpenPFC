# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
install(DIRECTORY include/openpfc DESTINATION include)

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
