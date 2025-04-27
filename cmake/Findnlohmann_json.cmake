# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

if(NOT nlohmann_json_FIND_VERSION)
  set(nlohmann_json_FIND_VERSION 3.11.2)
endif()
set(nlohmann_json_DOWNLOAD_URL https://github.com/nlohmann/json/releases/download/v${nlohmann_json_FIND_VERSION}/json.tar.xz)
message(STATUS "Fetching nlohmann-json from ${nlohmann_json_DOWNLOAD_URL}")
include(FetchContent)
FetchContent_Declare(json URL ${nlohmann_json_DOWNLOAD_URL})
FetchContent_MakeAvailable(json)
