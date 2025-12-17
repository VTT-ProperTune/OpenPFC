# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.15)
project(Aluminum VERSION 0.1.0 LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    find_package(OpenPFC REQUIRED)
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
    find_package(nlohmann_json REQUIRED)
    set(CMAKE_CXX_STANDARD 17)
endif()

add_executable(aluminumNew aluminumNew.cpp)
target_link_libraries(aluminumNew OpenPFC nlohmann_json::nlohmann_json)

# Only build tests if tests are enabled for the parent project
option(ALUMINUM_ENABLE_TESTS "Enable building Aluminum tests" ON)

if(ALUMINUM_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    find_package(Catch2 QUIET)
    if(TARGET Catch2::Catch2WithMain)
        add_executable(aluminumTest aluminumTest.cpp)
        target_link_libraries(aluminumTest OpenPFC nlohmann_json::nlohmann_json Catch2::Catch2WithMain)
        include(CTest)
        # Run all aluminum tests in a single invocation to avoid per-test MPI overhead
        add_test(NAME aluminum-all-tests COMMAND aluminumTest)
    else()
        message(STATUS "Aluminum: Catch2 not found, not building tests")
    endif()
endif()

install(TARGETS aluminumNew DESTINATION bin)
