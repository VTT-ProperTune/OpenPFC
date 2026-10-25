# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Try to find tomlplusplus via find_package first
set(OpenPFC_TOML_FETCHED FALSE)
find_package(tomlplusplus QUIET)

if(NOT tomlplusplus_FOUND)
  # If not found, download via FetchContent
  if(NOT tomlplusplus_FIND_VERSION)
    set(tomlplusplus_FIND_VERSION 3.4.0)
  endif()
  
  message(STATUS "Fetching tomlplusplus from GitHub")
  include(FetchContent)
  FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v${tomlplusplus_FIND_VERSION}
  )
  FetchContent_MakeAvailable(tomlplusplus)
  set(OpenPFC_TOML_FETCHED TRUE)
  
  # Store source directory for include path
  set(tomlplusplus_SOURCE_DIR ${tomlplusplus_SOURCE_DIR} CACHE INTERNAL "")
  
endif()

# Preserve all usage requirements if a package exposes only the plain target.
if(NOT TARGET tomlplusplus::tomlplusplus AND TARGET tomlplusplus_tomlplusplus)
  add_library(tomlplusplus::tomlplusplus ALIAS tomlplusplus_tomlplusplus)
endif()

if(tomlplusplus_FOUND OR TARGET tomlplusplus::tomlplusplus OR TARGET tomlplusplus_tomlplusplus)
  message(STATUS "✅ tomlplusplus found")
  # Set variable for consistency
  set(tomlplusplus_FOUND TRUE)
else()
  message(FATAL_ERROR "tomlplusplus not found")
endif()

