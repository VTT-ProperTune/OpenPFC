# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

if(NOT Catch2_FIND_VERSION)
  set(Catch2_FIND_VERSION 3.3.2)
endif()

message(STATUS "Fetching Catch2 version ${Catch2_FIND_VERSION} from GitHub https://github.com/catchorg/Catch2.git")

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        "v${Catch2_FIND_VERSION}"
)

FetchContent_MakeAvailable(Catch2)

# Set variables to indicate Catch2 was found (without PARENT_SCOPE in Find modules)
set(Catch2_FOUND TRUE)
set(Catch2_VERSION ${Catch2_FIND_VERSION})
set(Catch2_DIR "${catch2_BINARY_DIR}")
