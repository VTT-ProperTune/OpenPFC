# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Try to find tomlplusplus via find_package first
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
  
  # Store source directory for include path
  set(tomlplusplus_SOURCE_DIR ${tomlplusplus_SOURCE_DIR} CACHE INTERNAL "")
  
  # tomlplusplus via FetchContent creates target "tomlplusplus_tomlplusplus"
  # Create alias for consistency
  if(NOT TARGET tomlplusplus::tomlplusplus AND TARGET tomlplusplus_tomlplusplus)
    add_library(tomlplusplus::tomlplusplus INTERFACE IMPORTED)
    get_target_property(TOML_INC_DIR tomlplusplus_tomlplusplus INTERFACE_INCLUDE_DIRECTORIES)
    if(TOML_INC_DIR)
      set_target_properties(tomlplusplus::tomlplusplus PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TOML_INC_DIR}")
    else()
      set_target_properties(tomlplusplus::tomlplusplus PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${tomlplusplus_SOURCE_DIR}/include")
    endif()
  endif()
endif()

if(tomlplusplus_FOUND OR TARGET tomlplusplus::tomlplusplus OR TARGET tomlplusplus_tomlplusplus)
  message(STATUS "✅ tomlplusplus found")
  # Set variable for consistency
  set(tomlplusplus_FOUND TRUE)
else()
  message(FATAL_ERROR "tomlplusplus not found")
endif()

