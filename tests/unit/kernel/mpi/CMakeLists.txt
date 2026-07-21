# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_worker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_environment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mpi_io_helpers.cpp
)

# Standalone MPI error handling test executable
add_executable(test_environment_error_handling test_environment_error_handling.cpp)
target_link_libraries(test_environment_error_handling PRIVATE openpfc Catch2::Catch2 MPI::MPI_CXX)
add_test(NAME environment_error_handling COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_environment_error_handling> ${MPIEXEC_POSTFLAGS})
add_test(NAME environment_error_handling_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_environment_error_handling> ${MPIEXEC_POSTFLAGS})
set_tests_properties(environment_error_handling environment_error_handling_parallel PROPERTIES RUN_SERIAL TRUE)
