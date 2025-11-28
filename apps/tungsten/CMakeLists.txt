# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(tungsten tungsten.cpp)
find_package(Heffte REQUIRED)
target_link_libraries(tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)

option(TUNGSTEN_REUSE_ARRAYS "Reuse some arrays" OFF)

if(TUNGSTEN_REUSE_ARRAYS)
  message(STATUS "Tungsten: reusing arrays")
  target_compile_definitions(tungsten PUBLIC MAHTI_HACK)
endif()

install(TARGETS tungsten DESTINATION bin)

option(TUNGSTEN_ENABLE_TESTS "Enable building Tungsten tests" ON)

if(TUNGSTEN_ENABLE_TESTS)
    find_package(Catch2 REQUIRED)
    if(TARGET Catch2::Catch2WithMain)
        add_executable(tungstenTest tungstenTest.cpp)
        target_link_libraries(tungstenTest PRIVATE OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte Catch2::Catch2WithMain)
        include(CTest)
        include(Catch)
        catch_discover_tests(tungstenTest)
    else()
        message(WARNING "Tungsten: Catch2 not found, not building tests")
    endif()
endif()

