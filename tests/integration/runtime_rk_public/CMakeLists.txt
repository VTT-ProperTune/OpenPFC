# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(runtime_rk_public_consumer LANGUAGES C CXX)
find_package(OpenPFC 0.4 CONFIG REQUIRED)
if(NOT OpenPFC_SOURCE_STATE STREQUAL "clean" OR
   NOT OpenPFC_SOURCE_REVISION STREQUAL EXPECTED_OPENPFC_REVISION)
  message(FATAL_ERROR "Consumer requires the exact clean installed revision")
endif()
add_executable(runtime_rk_consumer main.cpp)
target_link_libraries(runtime_rk_consumer PRIVATE OpenPFC::openpfc)
enable_testing()
foreach(ranks 1 2 4)
  add_test(NAME runtime_rk_installed_${ranks}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:runtime_rk_consumer> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(runtime_rk_installed_${ranks} PROPERTIES
    RUN_SERIAL TRUE TIMEOUT 120)
endforeach()
