cmake_minimum_required(VERSION 3.15)

project(heFFTe)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
  heffte
  GIT_REPOSITORY "https://bitbucket.org/icl/heffte.git"
  GIT_TAG "v2.2.0")

FetchContent_Declare(
  json
  GIT_REPOSITORY "https://github.com/nlohmann/json.git"
  GIT_TAG "v3.10.4")

FetchContent_Declare(
  argparse
  GIT_REPOSITORY "https://github.com/p-ranav/argparse.git"
  GIT_TAG "v2.2")

FetchContent_MakeAvailable(heffte json argparse)

add_executable(pfc-heffte pfc-heffte)
target_link_libraries(pfc-heffte PRIVATE argparse nlohmann_json Heffte)
