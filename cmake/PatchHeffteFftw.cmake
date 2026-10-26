# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Apply only to the pinned HeFFTe 2.4.1 header. A repeated invocation is harmless;
# an unexpected source fails before any write. Never modify an installed prefix.

if(NOT DEFINED SOURCE_DIR OR NOT IS_DIRECTORY "${SOURCE_DIR}")
  message(FATAL_ERROR "Set SOURCE_DIR to the unpacked HeFFTe 2.4.1 source")
endif()
get_filename_component(SOURCE_DIR "${SOURCE_DIR}" REALPATH)
set(_header "${SOURCE_DIR}/include/heffte_backend_fftw.h")
file(SHA256 "${_header}" _actual)
set(_original "8114874b60565db7a4f779b2364b29cf4db3f2055b878f3a0296a9428d40ec11")
set(_patched "c7b0d134809a3fa8105bb52cb8ee1476a386ac30a6bc1951ee24e81f11ae6788")
if(_actual STREQUAL _patched)
  return()
endif()
if(NOT _actual STREQUAL _original)
  message(FATAL_ERROR "Unexpected HeFFTe FFTW header; refusing to overwrite local changes")
endif()

find_package(Git REQUIRED)
# Stop repository discovery at the dependency source, including when the build
# directory happens to be nested in an OpenPFC checkout.
set(_apply "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
  "GIT_CEILING_DIRECTORIES=${SOURCE_DIR}" "${GIT_EXECUTABLE}" apply)
set(_patch "${CMAKE_CURRENT_LIST_DIR}/patches/heffte-2.4.1-fftw-plans.patch")
execute_process(COMMAND ${_apply} --check "${_patch}"
  WORKING_DIRECTORY "${SOURCE_DIR}" RESULT_VARIABLE _check ERROR_VARIABLE _error)
if(NOT _check EQUAL 0)
  message(FATAL_ERROR "Cannot apply HeFFTe FFTW planning correction: ${_error}")
endif()
execute_process(COMMAND ${_apply} "${_patch}"
  WORKING_DIRECTORY "${SOURCE_DIR}" RESULT_VARIABLE _result ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "HeFFTe FFTW patch failed: ${_error}")
endif()
file(SHA256 "${_header}" _actual)
if(NOT _actual STREQUAL _patched)
  message(FATAL_ERROR "Unexpected patched HeFFTe FFTW header")
endif()
