# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Frontend I/O tests (standalone executables for MPI/serial)
add_executable(test_vtk_writer test_vtk_writer.cpp)
target_link_libraries(test_vtk_writer PRIVATE openpfc Catch2::Catch2 MPI::MPI_CXX)
add_test(NAME vtk_writer COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})
add_test(NAME vtk_writer_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})
set_tests_properties(vtk_writer vtk_writer_parallel PROPERTIES RUN_SERIAL TRUE)

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging PRIVATE openpfc Catch2::Catch2WithMain)
add_test(NAME logging COMMAND $<TARGET_FILE:test_logging>)
set_tests_properties(logging PROPERTIES RUN_SERIAL TRUE)
