# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Research prototype for issues #21 (NS), #23 (MHD), and #113
# (island coalescence). Admission to apps/ is a separate decision.

set(_ns2d_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_ns2d_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_ns2d_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_ns2d_heffte_tgt heffte)
endif()

add_executable(mhd2d src/mhd.cpp)
target_include_directories(mhd2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../flow)
target_link_libraries(mhd2d PRIVATE OpenPFC ${_ns2d_heffte_tgt})

if(OpenPFC_BUILD_TESTS)
  # The driver calls MPI_Init itself. Do not wrap with srun/mpiexec here:
  # on LUMI login nodes srun needs a partition and fails the regex check.
  add_test(NAME ns2d-vorticity-cli-verify
           COMMAND $<TARGET_FILE:ns2d_vorticity>
                   --verify --N 16 --steps 8 --dt 0.05 --nu 0.1)
  set_tests_properties(ns2d-vorticity-cli-verify PROPERTIES
                       PASS_REGULAR_EXPRESSION "VERIFY PASS"
                       TIMEOUT 120)
  add_test(NAME ns2d-spatial-refine-smoke
           COMMAND $<TARGET_FILE:ns2d_spatial_refine>
                   --resolutions 16,32 --reference 32
                   --dt 0.01 --times 0.02
                   --nu 1e-3 --rho 10 --eps 0.05
                   --control-n 16 --control-dt-factor 0.5
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/spatial_refine_smoke)
  set_tests_properties(ns2d-spatial-refine-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "spatial_refine status=ok"
                       TIMEOUT 120)
  add_test(NAME ns2d-temporal-refine-smoke
           COMMAND $<TARGET_FILE:ns2d_temporal_refine>
                   --case two-mode --n 16
                   --dts 0.02,0.01 --times 0.02 --nu 0.05
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/temporal_refine_smoke)
  set_tests_properties(ns2d-temporal-refine-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "temporal_refine status=ok"
                       TIMEOUT 120)
  add_test(NAME ns2d-flux-balance-smoke
           COMMAND $<TARGET_FILE:ns2d_flux_balance>
                   --case two-mode --n 16
                   --dt 0.02 --times 0.02 --nu 0.05
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/flux_balance_smoke)
  set_tests_properties(ns2d-flux-balance-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "flux_balance status=ok"
                       TIMEOUT 120)
  add_test(NAME mhd2d-cli-verify
           COMMAND $<TARGET_FILE:mhd2d>
                   --verify --N 16 --steps 8 --dt 0.05 --nu 0.1 --eta 0.1)
  set_tests_properties(mhd2d-cli-verify PROPERTIES
                       PASS_REGULAR_EXPRESSION "VERIFY PASS"
                       TIMEOUT 120)
  add_test(NAME mhd2d-cli-coalescence
           COMMAND $<TARGET_FILE:mhd2d>
                   --case island_coalescence --N 16 --steps 4 --dt 0.05
                   --nu 0.01 --eta 0.01)
  set_tests_properties(mhd2d-cli-coalescence PROPERTIES
                       PASS_REGULAR_EXPRESSION "case=island_coalescence"
                       TIMEOUT 120)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME mhd-topology-python
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_mhd_topology.py)
    set_tests_properties(mhd-topology-python PROPERTIES TIMEOUT 120)
    add_test(NAME mhd-sheet-geometry-python
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_mhd_sheet_geometry.py)
    set_tests_properties(mhd-sheet-geometry-python PROPERTIES TIMEOUT 120)
    add_test(NAME mhd-conference-vis-python
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_mhd_conference_vis.py)
    set_tests_properties(mhd-conference-vis-python PROPERTIES TIMEOUT 120)
  endif()
endif()

if(OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
  add_executable(test_ns2d_vorticity tests/test_ns2d.cpp)
  target_include_directories(test_ns2d_vorticity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../flow)
  target_link_libraries(test_ns2d_vorticity PRIVATE OpenPFC Catch2::Catch2
                        ${_ns2d_heffte_tgt})
  add_test(NAME ns2d-vorticity-cpu COMMAND test_ns2d_vorticity)
  set_tests_properties(ns2d-vorticity-cpu PROPERTIES TIMEOUT 180)

  add_executable(test_mhd2d tests/test_mhd.cpp)
  target_include_directories(test_mhd2d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../flow)
  target_link_libraries(test_mhd2d PRIVATE OpenPFC Catch2::Catch2
                        ${_ns2d_heffte_tgt})
  add_test(NAME mhd2d-cpu COMMAND test_mhd2d)
  set_tests_properties(mhd2d-cpu PROPERTIES TIMEOUT 300)
endif()
