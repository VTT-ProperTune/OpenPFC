cmake_minimum_required(VERSION 3.15)

project(pfc-heFFTe)

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

find_package(MPI REQUIRED)
find_package(Heffte REQUIRED)

add_library(pfc INTERFACE)
target_include_directories(pfc INTERFACE include/)
target_link_libraries(pfc INTERFACE Heffte MPI::MPI_CXX)

add_subdirectory(apps)
add_subdirectory(examples)
