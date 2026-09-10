# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Frontend I/O tests (standalone executables for MPI/serial)
target_sources(openpfc-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_hdf5_writer.cpp
)
add_executable(test_vtk_writer test_vtk_writer.cpp)
target_link_libraries(test_vtk_writer PRIVATE openpfc Catch2::Catch2 MPI::MPI_CXX)
add_test(NAME vtk_writer COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})
add_test(NAME vtk_writer_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_vtk_writer> ${MPIEXEC_POSTFLAGS})
set_tests_properties(vtk_writer vtk_writer_parallel PROPERTIES RUN_SERIAL TRUE)

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

# `openpfc_apps/field_snapshots.hpp` lives in apps/common (it is shared by the
# standalone science drivers, not by the library), so this target needs that
# include path and nlohmann_json explicitly.
add_executable(test_field_snapshots test_field_snapshots.cpp)
target_include_directories(test_field_snapshots PRIVATE
    ${CMAKE_SOURCE_DIR}/apps/common/include)
target_link_libraries(test_field_snapshots PRIVATE openpfc Catch2::Catch2
    MPI::MPI_CXX nlohmann_json::nlohmann_json)
add_test(NAME field_snapshots COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_field_snapshots> ${MPIEXEC_POSTFLAGS})
set_tests_properties(field_snapshots PROPERTIES RUN_SERIAL TRUE)
