# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# I/O component tests
add_executable(test_vtk_writer test_vtk_writer.cpp)
target_link_libraries(test_vtk_writer PRIVATE openpfc Catch2::Catch2 MPI::MPI_CXX)
add_test(NAME vtk_writer COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})

# Also test with multiple ranks
add_test(NAME vtk_writer_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})

# Prevent concurrent execution to avoid cross-test file cleanup races
set_tests_properties(vtk_writer vtk_writer_parallel PROPERTIES RUN_SERIAL TRUE)
