# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# CMake package configuration file generation

# Install CMake config file
install(EXPORT OpenPFCTargets
    FILE OpenPFCTargets.cmake
    NAMESPACE OpenPFC::
    DESTINATION lib/cmake/OpenPFC
)

# Generate config and write package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenPFCConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/OpenPFCConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/OpenPFC"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/OpenPFCConfigVersion.cmake"
  VERSION "${OpenPFC_VERSION_MAJOR}.${OpenPFC_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/OpenPFCConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/OpenPFCConfigVersion.cmake
  DESTINATION lib/cmake/OpenPFC
)

# Avoid export errors in local/dev builds where HeFFTe is fetched via
# FetchContent and not part of any install/export set.
if(DEFINED Heffte_DIR AND NOT Heffte_DIR STREQUAL "Heffte_DIR-NOTFOUND")
  export(EXPORT OpenPFCTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/OpenPFCTargets.cmake"
  )
else()
  message(STATUS "Skipping export of OpenPFCTargets (Heffte not installed via config)")
endif()
