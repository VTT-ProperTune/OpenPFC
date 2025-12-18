# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

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

include(FetchContent)

if(NOT Heffte_FIND_VERSION)
  set(Heffte_FIND_VERSION 2.4.1)
endif()

set(Heffte_DOWNLOAD_URL "https://bitbucket.org/icl/heffte.git")

message(STATUS "Fetching HeFFTe version ${Heffte_FIND_VERSION} from ${Heffte_DOWNLOAD_URL}")

FetchContent_Declare(
  heffte
  GIT_REPOSITORY ${Heffte_DOWNLOAD_URL}
  GIT_TAG "v${Heffte_FIND_VERSION}")

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

FetchContent_MakeAvailable(heffte)

# Set Heffte_FOUND to indicate success
set(Heffte_FOUND TRUE)
