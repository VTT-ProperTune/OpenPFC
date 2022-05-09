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

option(Heffte_ENABLE_FFTW "Enable the FFTW backend" ON)
option(Heffte_ENABLE_CUDA "Enable the CUDA and cuFFT backend" OFF)
option(Heffte_ENABLE_ROCM "Enable the HIP and rocFFT backend" OFF)
option(Heffte_ENABLE_ONEAPI "Enable the oneAPI/DPC++ and oneMKL backend" OFF)
option(Heffte_ENABLE_MKL "Enable the Intel MKL backend" OFF)
option(Heffte_ENABLE_DOXYGEN "Build the Doxygen documentation" OFF)
option(Heffte_ENABLE_AVX "Enable the use of AVX registers in the stock backend, adds flags: -mfma -mavx" OFF)
option(Heffte_ENABLE_AVX512 "Enable the use of AVX512 registers in the stock backend, adds AVX flags plus: -mavx512f -mavx512dq" OFF)
option(Heffte_ENABLE_MAGMA "Enable some helper functions from UTK MAGMA for GPU backends" OFF)
option(Heffte_ENABLE_PYTHON "Configure the Python scripts" OFF)
option(Heffte_ENABLE_FORTRAN "Build the Fortran modules for the selected backends." OFF)
option(Heffte_ENABLE_SWIG "Rebuild the SWIG bindings." OFF)
option(Heffte_ENABLE_TRACING "Enable the tracing capabilities" OFF)

option(ENABLE_CLANG_TIDY "Enable clang-tidy static code analysis" OFF)

if (ENABLE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif()

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY "https://github.com/nlohmann/json.git"
  GIT_TAG "v3.10.4")

FetchContent_Declare(
  argparse
  GIT_REPOSITORY "https://github.com/p-ranav/argparse.git"
  GIT_TAG "v2.2")

FetchContent_Declare(
  heffte
  GIT_REPOSITORY "https://bitbucket.org/icl/heffte.git"
  GIT_TAG "v2.2.0")

FetchContent_Declare(
  heffte
  GIT_REPOSITORY "https://bitbucket.org/icl/heffte.git"
  GIT_TAG "v2.2.0")

# https://bitbucket.org/icl/heffte/issues/42/add-option-to-use-static-fftw-libraries
FetchContent_GetProperties(heffte)
if(NOT heffte_POPULATED)
  FetchContent_Populate(heffte)
  include(FindHeffteFFTWLibraries)
  if (BUILD_SHARED_LIBS)
    message(STATUS "HeFFTe: using shared FFTW libraries")
    if (NOT FFTW_REQUIRED_LIBRARIES)
      set(FFTW_REQUIRED_LIBRARIES "fftw3" "fftw3f")
    endif()
    if (NOT FFTW_OPTIONAL_LIBRARIES)
      set(FFTW_OPTIONAL_LIBRARIES "fftw3_threads" "fftw3f_threads" "fftw3_omp" "fftw3f_omp")
    endif()
  else()
  message(STATUS "HeFFTe: using static FFTW libraries")
    # find static libraries instead of shared ones
    if (NOT FFTW_REQUIRED_LIBRARIES)
      set(FFTW_REQUIRED_LIBRARIES "libfftw3.a" "libfftw3f.a")
    endif()
    if (NOT FFTW_OPTIONAL_LIBRARIES)
      set(FFTW_OPTIONAL_LIBRARIES "libfftw3_threads.a" "libfftw3f_threads.a" "libfftw3_omp.a" "libfftw3f_omp.a")
    endif()
  endif()
  heffte_find_fftw_libraries(
        PREFIX $ENV{FFTW_ROOT}
        VAR FFTW_LIBRARIES
        REQUIRED ${FFTW_REQUIRED_LIBRARIES}
        OPTIONAL ${FFTW_OPTIONAL_LIBRARIES})
  add_subdirectory(${heffte_SOURCE_DIR} ${heffte_BINARY_DIR})
endif()

FetchContent_MakeAvailable(argparse json heffte)

add_library(pfc INTERFACE)
target_include_directories(pfc INTERFACE include/)
target_link_libraries(pfc INTERFACE Heffte)

#add_executable(pfc-heffte pfc-heffte.cpp)
#target_link_libraries(pfc-heffte PRIVATE argparse nlohmann_json Heffte)

#add_executable(diffusion examples/diffusion.cpp)
#target_link_libraries(diffusion PRIVATE pfc argparse nlohmann_json Heffte)
#target_compile_options(diffusion PRIVATE -Wall -Wextra -Wpedantic)
#target_compile_options(diffusion PRIVATE -Wall -Wextra -Wpedantic -Werror)


add_executable(clean_tungsten examples/clean_tungsten.cpp)
target_link_libraries(clean_tungsten PRIVATE pfc)

#add_executable(clean_tungsten_orig examples/clean_tungsten_orig.cpp)
#target_link_libraries(clean_tungsten_orig PRIVATE argparse nlohmann_json Heffte)
