cmake_minimum_required(VERSION 3.15)

project(OpenPFC VERSION 0.1.0 DESCRIPTION "Phase Field Crystal simulation framework")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

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

option(BUILD_DOCUMENTATION "Build documentation" ON)

if(BUILD_DOCUMENTATION)
  message(STATUS "Building documentation")
  find_package(Doxygen)

  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building documentation")
  endif()
endif()

find_package(MPI REQUIRED)
find_package(Heffte REQUIRED)

add_library(pfc INTERFACE)
target_include_directories(pfc INTERFACE include/)
target_link_libraries(pfc INTERFACE Heffte MPI::MPI_CXX)

add_subdirectory(apps)
add_subdirectory(examples)
