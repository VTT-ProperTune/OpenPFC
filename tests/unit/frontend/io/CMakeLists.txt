# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Frontend I/O tests (standalone executables for MPI/serial)
target_sources(openpfc-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_hdf5_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_snapshot_series.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_diagnostics_series.cpp
)
add_executable(test_vtk_writer test_vtk_writer.cpp)
target_link_libraries(test_vtk_writer PRIVATE openpfc Catch2::Catch2 MPI::MPI_CXX)
add_test(NAME vtk_writer COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})
add_test(NAME vtk_writer_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})
# CI sets OpenPFC_MPI_TEST_MAX_WORLD_SIZE=2. A 4-rank run is registered only
# when that cap allows it. The cache value is visible here even though the
# option() declaration lives later in tests/CMakeLists.txt.
if(NOT DEFINED OpenPFC_MPI_TEST_MAX_WORLD_SIZE OR
   OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
   OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 4)
  add_test(NAME vtk_writer_4ranks COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(vtk_writer vtk_writer_parallel vtk_writer_4ranks
                       PROPERTIES RUN_SERIAL TRUE)
else()
  set_tests_properties(vtk_writer vtk_writer_parallel PROPERTIES RUN_SERIAL TRUE)
endif()

add_executable(test_binary_writer test_binary_writer.cpp)
target_link_libraries(test_binary_writer PRIVATE openpfc Catch2::Catch2 MPI::MPI_CXX)
add_test(NAME binary_writer COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_binary_writer> ${MPIEXEC_POSTFLAGS})
add_test(NAME binary_writer_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_binary_writer> ${MPIEXEC_POSTFLAGS})
set_tests_properties(binary_writer binary_writer_parallel PROPERTIES RUN_SERIAL TRUE)

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging PRIVATE openpfc Catch2::Catch2WithMain)
add_test(NAME logging COMMAND $<TARGET_FILE:test_logging>)
set_tests_properties(logging PROPERTIES RUN_SERIAL TRUE)

add_executable(test_png_writer test_png_writer.cpp)
target_link_libraries(test_png_writer PRIVATE openpfc Catch2::Catch2 MPI::MPI_CXX)
add_test(NAME png_writer COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_png_writer> ${MPIEXEC_POSTFLAGS})
add_test(NAME png_writer_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_png_writer> ${MPIEXEC_POSTFLAGS})
set_tests_properties(png_writer png_writer_parallel PROPERTIES RUN_SERIAL TRUE)

