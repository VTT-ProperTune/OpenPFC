# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Project setup and version configuration
# Note: project() is called in root CMakeLists.txt to avoid CMake warnings
# This file only handles version suffix and development mode settings

# Define an option for development mode
option(OpenPFC_DEVELOPMENT "Enable development build settings" OFF)

if(OpenPFC_DEVELOPMENT)
  message(STATUS "Development version detected, enabling compile_commands.json export")
  set(OpenPFC_VERSION_SUFFIX "-dev")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
else()
  set(OpenPFC_VERSION_SUFFIX "")
endif()

# Configure a header file to pass the version number
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h
)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# Default to Debug build type if not set
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Debug.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# To preserve RPATH when installing
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Prefer "config mode", i.e. system wide installed packages
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
