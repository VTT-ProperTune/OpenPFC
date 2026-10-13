# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rk_stage_binding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_butcher_tableau.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_embedded_rk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_etd1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_imex_stage_composition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stage_workspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_integrator_method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stepper_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stepper_concept.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_method_composition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_step_attempt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_step_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rk_nfield.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_runtime_field_rk.cpp
)

# AC-facing CTest name: `ctest -R '^method_composition$'`.
add_test(NAME method_composition
         COMMAND $<TARGET_FILE:openpfc-tests> "[method_composition]")
set_tests_properties(method_composition PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# AC-facing CTest name: `ctest -R test_imex_stage_composition` (shared openpfc-tests binary).
add_test(NAME test_imex_stage_composition
         COMMAND $<TARGET_FILE:openpfc-tests> "[imex_stage_composition]")
set_tests_properties(test_imex_stage_composition PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# AC-facing CTest name: `ctest -R '^test_step_attempt$'` (shared openpfc-tests binary).
add_test(NAME test_step_attempt
         COMMAND $<TARGET_FILE:openpfc-tests> "[step_attempt]")
set_tests_properties(test_step_attempt PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

# AC-facing CTest name: `ctest -R '^test_step_protocol$'` (shared openpfc-tests binary).
add_test(NAME test_step_protocol
         COMMAND $<TARGET_FILE:openpfc-tests> "[step_protocol]")
set_tests_properties(test_step_protocol PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

add_test(NAME test_rk_stage_binding
         COMMAND $<TARGET_FILE:openpfc-tests> "[rk_stage_binding]")
# Isolate the existing complete global allocation provider from other suites.
add_executable(test_rk_stage_allocations test_rk_stage_allocations.cpp)
target_include_directories(test_rk_stage_allocations PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_rk_stage_allocations PRIVATE OpenPFC Catch2::Catch2WithMain)
add_test(NAME test_rk_stage_allocations COMMAND test_rk_stage_allocations)
add_test(NAME test_rk_nfield
         COMMAND $<TARGET_FILE:openpfc-tests> "[rk_nfield]")
set_tests_properties(test_rk_nfield PROPERTIES
  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

add_test(NAME test_runtime_field_rk
         COMMAND $<TARGET_FILE:openpfc-tests> "[runtime_field_rk]")
if(OpenPFC_RUN_MPI_SUITES)
  foreach(_runtime_ranks 1 2 4)
    if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
       _runtime_ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE)
      continue()
    endif()
    add_test(NAME runtime_field_rk_${_runtime_ranks}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_runtime_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:openpfc-tests> "[runtime_field_rk]"
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(runtime_field_rk_${_runtime_ranks}ranks PROPERTIES
      RUN_SERIAL TRUE TIMEOUT 120)
  endforeach()
endif()
