# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(_flow_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_flow_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_flow_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_flow_heffte_tgt heffte)
endif()

add_executable(ns2d_vorticity vorticity/model.cpp)
target_include_directories(ns2d_vorticity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ns2d_vorticity PRIVATE OpenPFC ${_flow_heffte_tgt})
set_target_properties(ns2d_vorticity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/ns2d_vorticity")

if(OpenPFC_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME example-vorticity-output
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/examples/render_flow_vorticity.py"
      --execute $<TARGET_FILE:ns2d_vorticity> --run "${PROJECT_SOURCE_DIR}/results/example-checks/vorticity-1" --check)
  set_tests_properties(example-vorticity-output PROPERTIES TIMEOUT 90 ENVIRONMENT "OMP_NUM_THREADS=1")
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-vorticity-output-mpi
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/examples/render_flow_vorticity.py"
        --execute $<TARGET_FILE:ns2d_vorticity> --run "${PROJECT_SOURCE_DIR}/results/example-checks/vorticity-2" --check
        --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-vorticity-output-mpi PROPERTIES TIMEOUT 120 PROCESSORS 2
      ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
endif()
