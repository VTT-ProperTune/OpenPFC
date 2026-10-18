# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(rk_preparation_consumer LANGUAGES C CXX)
find_package(OpenPFC CONFIG REQUIRED)
set(backend cuda CACHE STRING "cuda or hip")
if(backend STREQUAL "cuda")
  enable_language(CUDA)
  set(native_source ../../integration/rk_coupled/resident.cu)
elseif(backend STREQUAL "hip")
  enable_language(HIP)
  set(native_source ../../integration/rk_coupled/resident.hip)
else()
  message(FATAL_ERROR "backend must be cuda or hip")
endif()
# Reuse the complete stage/halo/rollback test against installed headers only.
add_executable(rk_preparation_consumer ${native_source})
target_link_libraries(rk_preparation_consumer PRIVATE OpenPFC::openpfc)
set_target_properties(rk_preparation_consumer PROPERTIES CXX_STANDARD 20
  CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
if(backend STREQUAL "cuda")
  target_link_libraries(rk_preparation_consumer PRIVATE CUDA::cudart)
else()
  target_link_libraries(rk_preparation_consumer PRIVATE hip::host)
endif()
enable_testing()
foreach(ranks 1 2)
  add_test(NAME prepared_time_installed_${ranks}ranks
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
      $<TARGET_FILE:rk_preparation_consumer>)
  set_tests_properties(prepared_time_installed_${ranks}ranks
    PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
endforeach()
