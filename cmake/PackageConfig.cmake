# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# CMake package configuration file generation

# Install CMake config file
install(EXPORT OpenPFCTargets
    FILE OpenPFCTargets.cmake
    NAMESPACE OpenPFC::
    DESTINATION lib/cmake/OpenPFC
)

include("${CMAKE_CURRENT_LIST_DIR}/SourceProvenance.cmake")

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
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/OpenPFCConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/OpenPFCConfigVersion.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFTW3.cmake
  DESTINATION lib/cmake/OpenPFC
)

# Build-tree consumers use the same relocatable dependency finder.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFTW3.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/FindFFTW3.cmake COPYONLY)

# Avoid export errors in local/dev builds where HeFFTe is fetched via
# FetchContent and not part of any install/export set.
if(DEFINED Heffte_DIR AND NOT Heffte_DIR STREQUAL "Heffte_DIR-NOTFOUND")
  export(EXPORT OpenPFCTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/OpenPFCTargets.cmake"
    NAMESPACE OpenPFC::
  )
else()
  message(STATUS "Skipping export of OpenPFCTargets (Heffte not installed via config)")
endif()

if(OpenPFC_BUILD_TESTS AND OpenPFC_HAS_FFTW3)
  add_test(NAME LocalFFT_InstalledConsumer
    COMMAND ${CMAKE_COMMAND} "-Dbuild=${CMAKE_BINARY_DIR}" "-Dsource=${CMAKE_SOURCE_DIR}"
      "-Dcc=${CMAKE_C_COMPILER}" "-Dcxx=${CMAKE_CXX_COMPILER}"
      -P ${CMAKE_SOURCE_DIR}/tests/packaging/check_local_fft_consumer.cmake)
  set_tests_properties(LocalFFT_InstalledConsumer PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
endif()
