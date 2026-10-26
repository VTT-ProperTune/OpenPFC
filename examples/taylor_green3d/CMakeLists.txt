# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Issue #229 verification driver. Not an apps/ executable.

set(_tg_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_tg_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_tg_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_tg_heffte_tgt heffte)
endif()

if(OpenPFC_BUILD_TESTS)
  # The driver calls MPI_Init itself. Do not wrap it with srun.
  add_test(NAME taylor-green3d-smoke
           COMMAND $<TARGET_FILE:taylor_green3d>
                   --case smoke
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/smoke)
  set_tests_properties(taylor-green3d-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "taylor_green3d status=ok"
                       TIMEOUT 120)
endif()

if(OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
  add_executable(test_taylor_green3d tests/test_taylor_green.cpp)
  target_include_directories(test_taylor_green3d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../flow)
  target_link_libraries(test_taylor_green3d PRIVATE OpenPFC Catch2::Catch2
                        ${_tg_heffte_tgt})
  add_test(NAME taylor-green3d-cpu COMMAND test_taylor_green3d)
  set_tests_properties(taylor-green3d-cpu PROPERTIES TIMEOUT 180)
endif()
