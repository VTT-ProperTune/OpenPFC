# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.15)

project(OpenPFC VERSION 0.1.1 DESCRIPTION "Phase Field Crystal simulation framework")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Debug build type if not set
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Debug.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Enable debug macros if build type is Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(NAN_CHECK_ENABLED)
endif()

# To preserve RPATH when installing
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(USE_CLANG_TIDY "Enable clang-tidy static code analysis" OFF)
# Enable clang-tidy if USE_CLANG_TIDY is ON
if(USE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy")
    if(CLANG_TIDY_EXECUTABLE)
        message(STATUS "Enabling clang-tidy")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}" "-extra-arg=-stdlib=libc++")
    else()
        message(WARNING "clang-tidy executable not found. Please install clang-tidy and ensure it is in your system PATH.")
        message(WARNING "Refer to the installation instructions at: https://clang.llvm.org/extra/clang-tidy/")
    endif()
endif()

option(OpenPFC_BUILD_DOCUMENTATION "Build documentation" ON)

if(OpenPFC_BUILD_DOCUMENTATION)
  message(STATUS "📚 Generating sources for documentation (build with target docs)")
  find_package(Doxygen)

  if(Doxygen_FOUND)
    message(STATUS "✅ Doxygen v${DOXYGEN_VERSION} found at ${DOXYGEN_EXECUTABLE}")
    add_subdirectory(docs)
  else()
    message(WARNING "⚠️  Doxygen not found, skipping documentation generation.")
  endif()
endif()

# Prefer "config mode", i.e. system wide installed packages
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

find_package(MPI REQUIRED)

# Try to find HeFFTe installed on the system (e.g., installed by Nix)
find_package(Heffte CONFIG QUIET)
# If not found, fallback to downloading via FetchContent
if (Heffte_FOUND)
  message(STATUS "✅ HeFFTe v${Heffte_VERSION} found at ${Heffte_DIR}")
else()
  message(WARNING "⚠️ Heffte not found via find_package(), falling back to FetchContent.")
  include(cmake/FindHeffte.cmake)
  if (NOT Heffte_FOUND)
    message(FATAL_ERROR "HeFFTe not found. Please install HeFFTe or set the Heffte_DIR variable to the location of HeffteConfig.cmake.")
  endif()
endif()

find_package(nlohmann_json REQUIRED)

option(BUILD_SHARED_LIBS "Build OpenPFC as a shared library" OFF)

# Create library
add_library(OpenPFC
    src/openpfc/core/world.cpp
    # src/openpfc/core/decomposition.cpp
    # src/openpfc/core/fft.cpp
    # Add more .cpp files as you go
)

# Public API (headers)
target_include_directories(OpenPFC
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Options
option(OpenPFC_ENABLE_MPI "Enable MPI support" ON)
option(OpenPFC_ENABLE_HEFFTE "Enable HeFFTe FFT support" ON)

# Conditionally find MPI
if(OpenPFC_ENABLE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(OpenPFC PUBLIC MPI::MPI_CXX)
endif()

# Conditionally find HeFFTe
if(OpenPFC_ENABLE_HEFFTE)
    find_package(Heffte REQUIRED)
    target_link_libraries(OpenPFC PUBLIC Heffte::Heffte)
endif()

# Require C++17
target_compile_features(OpenPFC PUBLIC cxx_std_17)

option(OpenPFC_BUILD_APPS "Build OpenPFC applications" ON)
option(OpenPFC_BUILD_EXAMPLES "Build OpenPFC examples" ON)
option(OpenPFC_BUILD_TESTS "Build OpenPFC tests" ON)
option(OpenPFC_ENABLE_CODE_COVERAGE "Enable coverage" ON)

if(OpenPFC_BUILD_TESTS)
  message(STATUS "🔍 Building tests")
  find_package(Catch2 REQUIRED)
  if(Catch2_FOUND)
    message(STATUS "✅ Catch2 v${Catch2_VERSION} found at ${Catch2_DIR}")
    add_subdirectory(tests)
    message(STATUS "Installing openpfc-tests binary")
    install(TARGETS openpfc-tests DESTINATION bin)
  else()
    message(WARNING "⚠️  Catch2 not found, skipping tests.")
    message(WARNING "⚠️  Please install Catch2 or set the CATCH2_DIR variable to the location of Catch2Config.cmake.")
  endif()
endif()

if(OpenPFC_BUILD_APPS)
  message(STATUS "📦 Building applications")
  add_subdirectory(apps)
endif()

if(OpenPFC_BUILD_EXAMPLES)
  message(STATUS "📚 Building examples")
  add_subdirectory(examples)
endif()

if(OpenPFC_ENABLE_CODE_COVERAGE)
  message(STATUS "📊 Enabling code coverage")
  target_compile_options(OpenPFC PUBLIC --coverage)
  target_link_options(OpenPFC PUBLIC --coverage)
endif()

# Install nlohmann_json headers, but only if nlohmann_json_SOURCE_DIR is
# defined, i.e. the package is built from source during the configure step.
# This is to avoid installing the headers if the package is installed from
# a system wide package manager.
if(DEFINED nlohmann_json_SOURCE_DIR)
  message(STATUS "Installing nlohmann_json headers")
  install(DIRECTORY ${nlohmann_json_SOURCE_DIR}/include/nlohmann
          DESTINATION include
  )
endif()

# generate cmake file containing code to import all targets

# Install headers
install(DIRECTORY include/openpfc DESTINATION include)

# Install library binary
install(TARGETS OpenPFC
    EXPORT OpenPFCTargets
    ARCHIVE DESTINATION lib   # .a files
    LIBRARY DESTINATION lib   # .so files
    RUNTIME DESTINATION bin   # executable files (not needed now but future proof)
)

# Install CMake config file
install(EXPORT OpenPFCTargets
    FILE OpenPFCTargets.cmake
    NAMESPACE OpenPFC::
    DESTINATION lib/cmake/OpenPFC
)

# generate config and write package config

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

export(EXPORT OpenPFCTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/OpenPFCTargets.cmake"
)

# -----------------------------------------------------------------------------------
# Final Build Summary
# -----------------------------------------------------------------------------------

message(STATUS "-------------------------------------------------------------")
message(STATUS " ✅ OpenPFC Build Configuration Summary")
message(STATUS "-------------------------------------------------------------")

message(STATUS " CMake Version          : ${CMAKE_VERSION}")
message(STATUS " C++ Compiler           : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " C++ Standard           : ${CMAKE_CXX_STANDARD}")
message(STATUS " Build Type             : ${CMAKE_BUILD_TYPE}")
message(STATUS " Install RPATH          : ${CMAKE_INSTALL_RPATH_USE_LINK_PATH}")

message(STATUS "-------------------------------------------------------------")
message(STATUS " 📦 Third-Party Packages:")
message(STATUS " MPI                    : ${MPI_CXX_COMPILER}")
message(STATUS " Heffte_DIR             : ${Heffte_DIR}")
message(STATUS " nlohmann_json          : ${nlohmann_json_DIR}")
if(Doxygen_FOUND)
  message(STATUS " Doxygen                : ${DOXYGEN_EXECUTABLE} (version ${DOXYGEN_VERSION})")
else()
  message(STATUS " Doxygen                : NOT FOUND")
endif()
if(Catch2_FOUND)
  message(STATUS " Catch2                 : ${Catch2_DIR} (version ${Catch2_VERSION})")
else()
  message(STATUS " Catch2                 : NOT FOUND")
endif()

message(STATUS "-------------------------------------------------------------")
message(STATUS " 🛠 Build Options:")
message(STATUS " OpenPFC_BUILD_APPS               = ${OpenPFC_BUILD_APPS}")
message(STATUS " OpenPFC_BUILD_EXAMPLES           = ${OpenPFC_BUILD_EXAMPLES}")
message(STATUS " OpenPFC_BUILD_TESTS              = ${OpenPFC_BUILD_TESTS}")
message(STATUS " OpenPFC_BUILD_DOCUMENTATION      = ${OpenPFC_BUILD_DOCUMENTATION}")
message(STATUS " OpenPFC_ENABLE_CODE_COVERAGE     = ${OpenPFC_ENABLE_CODE_COVERAGE}")
message(STATUS " USE_CLANG_TIDY                   = ${USE_CLANG_TIDY}")
message(STATUS "-------------------------------------------------------------")
message(STATUS " 📂 Install prefix       : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "-------------------------------------------------------------")
message(STATUS " 🎉 Ready to build OpenPFC!")
message(STATUS "-------------------------------------------------------------")
message(STATUS "To build OpenPFC, run:")
message(STATUS "  cmake --build build")
message(STATUS "-------------------------------------------------------------")
