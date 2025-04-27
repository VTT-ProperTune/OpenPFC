# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(tungsten tungsten.cpp)
target_link_libraries(tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json)

option(TUNGSTEN_REUSE_ARRAYS "Reuse some arrays" OFF)
if(TUNGSTEN_REUSE_ARRAYS)
  message(STATUS "Tungsten: reusing arrays")
  target_compile_definitions(tungsten PUBLIC MAHTI_HACK)
endif()

install(TARGETS tungsten DESTINATION bin)

add_subdirectory(aluminumNew)
