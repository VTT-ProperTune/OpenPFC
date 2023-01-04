cmake_minimum_required(VERSION 3.15)

project(OpenPFC VERSION 0.1.0 DESCRIPTION "Phase Field Crystal simulation framework")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(BUILD_SHARED_LIBS "Builds shared libraries using CMake conventions" OFF)

option(ENABLE_CLANG_TIDY "Enable clang-tidy static code analysis" OFF)

if(ENABLE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy)
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
target_include_directories(OpenPFC INTERFACE include/)
target_link_libraries(OpenPFC INTERFACE Heffte::Heffte MPI::MPI_CXX)

add_subdirectory(apps)
add_subdirectory(examples)

install(DIRECTORY include/openpfc DESTINATION include)
