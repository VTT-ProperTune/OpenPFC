cmake_minimum_required(VERSION 3.15)

project(OpenPFC VERSION 0.1.0 DESCRIPTION "Phase Field Crystal simulation framework")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
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

option(BUILD_SHARED_LIBS "Builds shared libraries using CMake conventions" OFF)

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
  message(STATUS "Generating sources for documentation (build with target docs)")
  find_package(Doxygen)

  if(Doxygen_FOUND)
    message(STATUS "Using Doxygen ${DOXYGEN_VERSION} found from ${DOXYGEN_EXECUTABLE}")
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building documentation")
  endif()
endif()

# Prefer "config mode", i.e. system wide installed packages
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

find_package(MPI REQUIRED)
find_package(Heffte REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(OpenPFC INTERFACE)
target_include_directories(OpenPFC
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:include>
                          )
target_link_libraries(OpenPFC INTERFACE Heffte::Heffte MPI::MPI_CXX)
target_compile_features(OpenPFC INTERFACE cxx_std_17)

option(OpenPFC_BUILD_APPS "Build OpenPFC applications" ON)
option(OpenPFC_BUILD_EXAMPLES "Build OpenPFC examples" ON)
option(OpenPFC_BUILD_TESTS "Build OpenPFC tests" ON)
option(OpenPFC_ENABLE_CODE_COVERAGE "Enable coverage" ON)

if(OpenPFC_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(OpenPFC_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(OpenPFC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(OpenPFC_ENABLE_CODE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

install(DIRECTORY include/openpfc DESTINATION include)

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

install(TARGETS OpenPFC EXPORT OpenPFCTargets DESTINATION include)
install(EXPORT OpenPFCTargets
  FILE OpenPFCTargets.cmake
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
