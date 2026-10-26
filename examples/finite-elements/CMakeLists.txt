# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(q1_multigrid q1-multigrid/model.cpp)
target_link_libraries(q1_multigrid PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(q1_multigrid PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

if(OpenPFC_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME example-q1-multigrid
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../render_q1_multigrid.py"
      --execute $<TARGET_FILE:q1_multigrid>
      --run "${PROJECT_SOURCE_DIR}/results/example-checks/q1-1")
  set_tests_properties(example-q1-multigrid PROPERTIES TIMEOUT 240 ENVIRONMENT "OMP_NUM_THREADS=1")
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME example-q1-multigrid-mpi
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../render_q1_multigrid.py"
        --execute $<TARGET_FILE:q1_multigrid>
        --run "${PROJECT_SOURCE_DIR}/results/example-checks/q1-2"
        --launcher "${MPIEXEC_EXECUTABLE}" "--rank-flag=${MPIEXEC_NUMPROC_FLAG}" --ranks 2)
    set_tests_properties(example-q1-multigrid-mpi PROPERTIES TIMEOUT 240 PROCESSORS 2
      ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
endif()
