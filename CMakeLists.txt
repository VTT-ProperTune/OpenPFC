# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.15)

project(OpenPFC VERSION 0.1.3 DESCRIPTION "Phase Field Crystal simulation framework")

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Debug build type if not set
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Debug.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Base compiler flags for all build types
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors -Werror=format-security")

# Additional warnings for better code quality
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wnull-dereference    # Warn about potential null pointer dereferences
    -Wduplicated-cond     # Warn about duplicated conditions
    -Wduplicated-branches # Warn about duplicated branches
    -Wlogical-op          # Warn about logical operator issues
  )
endif()

# Build-type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Enable debug macros and stricter warnings for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(NAN_CHECK_ENABLED)
    # Additional warning flags for Debug mode to catch more issues
    # Note: We enable useful warnings but avoid overly pedantic ones
    add_compile_options(
        -Wunused                # Warn on unused entities (variables, functions, etc.)
        -Wshadow                # Warn on variable shadowing
        # Disabled overly pedantic warnings:
        # -Wconversion          # Too many false positives with int/size_t conversions
        # -Wsign-conversion     # Too pedantic for array indexing with int
        # -Wmissing-braces      # C++ aggregate initialization is fine without double braces
        # -Wpedantic            # Enforces strict ISO C++, sometimes too restrictive
        # -Wfloat-equal         # Sometimes we do need exact floating point comparisons
        # Future consideration:
        # -Werror               # Treat all warnings as errors (good for CI/CD)
    )
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
add_library(openpfc
    src/openpfc/core/world.cpp
    src/openpfc/core/box3d.cpp
    src/openpfc/core/decomposition.cpp
    src/openpfc/factory/decomposition_factory.cpp
    src/openpfc/fft.cpp
    src/openpfc/ui_errors.cpp
    # Add more .cpp files as you go
)

add_library(OpenPFC ALIAS openpfc)

# Set the library version
set_target_properties(openpfc PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "openpfc"  # ensures the filename is lowercase
)

#set_target_properties(openpfc PROPERTIES
#    PUBLIC_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
#)

# Public API (headers)
target_include_directories(openpfc
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
    target_link_libraries(openpfc PUBLIC MPI::MPI_CXX)
endif()

# Conditionally find HeFFTe
if(OpenPFC_ENABLE_HEFFTE)
    find_package(Heffte REQUIRED)
    target_link_libraries(openpfc PUBLIC Heffte::Heffte)
endif()

# Require C++17
target_compile_features(openpfc PUBLIC cxx_std_17)

option(OpenPFC_BUILD_APPS "Build OpenPFC applications" ON)
option(OpenPFC_BUILD_EXAMPLES "Build OpenPFC examples" ON)
option(OpenPFC_BUILD_TESTS "Build OpenPFC tests" ON)
option(OpenPFC_BUILD_BENCHMARKS "Build performance benchmarks (slow tests)" OFF)
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
  target_compile_options(openpfc PUBLIC --coverage)
  target_link_options(openpfc PUBLIC --coverage)
  
  # Find coverage tools
  find_program(LCOV_EXECUTABLE lcov)
  find_program(GENHTML_EXECUTABLE genhtml)
  
  # Use GCOV from toolchain file if specified, otherwise find it
  if(NOT GCOV_EXECUTABLE)
    find_program(GCOV_EXECUTABLE gcov)
  endif()
  
  if(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE AND GCOV_EXECUTABLE)
    message(STATUS "✅ Coverage tools found: lcov, genhtml, and gcov")
    message(STATUS "   lcov:    ${LCOV_EXECUTABLE}")
    message(STATUS "   genhtml: ${GENHTML_EXECUTABLE}")
    message(STATUS "   gcov:    ${GCOV_EXECUTABLE}")
    
    # Add custom target for generating coverage report
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E echo "==> Cleaning old coverage data..."
      COMMAND ${CMAKE_COMMAND} -E remove -f coverage.info coverage_filtered.info
      COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Running tests..."
      COMMAND ${CMAKE_BINARY_DIR}/tests/openpfc-tests
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Capturing coverage data..."
      COMMAND ${LCOV_EXECUTABLE}
        --gcov-tool ${GCOV_EXECUTABLE}
        --capture
        --directory ${CMAKE_BINARY_DIR}
        --output-file coverage.info
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Filtering coverage data..."
      COMMAND ${LCOV_EXECUTABLE}
        --remove coverage.info
        '/usr/*'
        '*/build/_deps/*'
        '*/tests/*'
        '*/examples/*'
        '*/apps/*'
        --output-file coverage_filtered.info
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Generating HTML report..."
      COMMAND ${GENHTML_EXECUTABLE}
        coverage_filtered.info
        --output-directory coverage-html
        --title "OpenPFC Coverage Report"
        --legend
      
      COMMAND ${CMAKE_COMMAND} -E echo ""
      COMMAND ${CMAKE_COMMAND} -E echo "✅ Coverage report generated!"
      COMMAND ${CMAKE_COMMAND} -E echo "   Open: ${CMAKE_BINARY_DIR}/coverage-html/index.html"
      
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPENDS openpfc-tests
      COMMENT "Generating code coverage report"
    )
    
    # Add target to clean coverage data
    add_custom_target(coverage-clean
      COMMAND ${CMAKE_COMMAND} -E remove -f coverage.info coverage_filtered.info
      COMMAND ${CMAKE_COMMAND} -E remove_directory coverage-html
      COMMAND ${LCOV_EXECUTABLE}
        --gcov-tool ${GCOV_EXECUTABLE}
        --zerocounters
        --directory ${CMAKE_BINARY_DIR}
      COMMAND ${CMAKE_COMMAND} -E echo "✅ Coverage data cleaned"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Cleaning coverage data"
    )
    
    message(STATUS "   Run 'ninja coverage' to generate coverage report")
    message(STATUS "   Run 'ninja coverage-clean' to reset coverage data")
  else()
    if(NOT LCOV_EXECUTABLE)
      message(WARNING "⚠️  lcov not found - coverage report generation disabled")
      message(WARNING "   Install: sudo apt install lcov (Ubuntu/Debian)")
      message(WARNING "           brew install lcov (macOS)")
    endif()
    if(NOT GENHTML_EXECUTABLE)
      message(WARNING "⚠️  genhtml not found - coverage report generation disabled")
    endif()
    if(NOT GCOV_EXECUTABLE)
      message(WARNING "⚠️  gcov not found - coverage report generation disabled")
    endif()
  endif()
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
install(TARGETS openpfc
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

message(STATUS " CMake Version                : ${CMAKE_VERSION}")
message(STATUS " C++ Compiler                 : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " C++ Standard                 : ${CMAKE_CXX_STANDARD}")
message(STATUS " Build Type                   : ${CMAKE_BUILD_TYPE}")
message(STATUS " Install RPATH                : ${CMAKE_INSTALL_RPATH_USE_LINK_PATH}")

# development mode or not?
if(OpenPFC_DEVELOPMENT)
  message(STATUS " Development mode             : ON")
else()
  message(STATUS " Development mode             : OFF")
endif()

# shared or static library?
if(BUILD_SHARED_LIBS)
  message(STATUS " Build type                   : SHARED")
else()
  message(STATUS " Build type                   : STATIC")
endif()

# build system: ninja or make?
if(CMAKE_GENERATOR MATCHES "Ninja")
  message(STATUS " Build system                 : Ninja")
else()
  message(STATUS " Build system                 : Make")
endif()

# export compile_commands.json or not?
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  message(STATUS " Export compile_commands.json : YES")
else()
  message(STATUS " Export compile_commands.json : NO")
endif()

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
message(STATUS " OpenPFC_DEVELOPMENT              = ${OpenPFC_DEVELOPMENT}")
message(STATUS " OpenPFC_BUILD_APPS               = ${OpenPFC_BUILD_APPS}")
message(STATUS " OpenPFC_BUILD_EXAMPLES           = ${OpenPFC_BUILD_EXAMPLES}")
message(STATUS " OpenPFC_BUILD_TESTS              = ${OpenPFC_BUILD_TESTS}")
message(STATUS " OpenPFC_BUILD_BENCHMARKS         = ${OpenPFC_BUILD_BENCHMARKS}")
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
