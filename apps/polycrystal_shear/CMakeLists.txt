# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(_shear_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_shear_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_shear_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_shear_heffte_tgt heffte)
endif()

add_executable(polycrystal_shear src/polycrystal_shear.cpp)
target_link_libraries(polycrystal_shear PRIVATE OpenPFC ${_shear_heffte_tgt})
install(TARGETS polycrystal_shear DESTINATION bin)

if(OpenPFC_BUILD_TESTS)
  # The driver calls MPI_Init itself. Do not wrap it with srun.
  add_test(NAME polycrystal-shear-smoke
           COMMAND $<TARGET_FILE:polycrystal_shear>
                   --size 4
                   --steps 2
                   --shear 0.002
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/smoke
                   --commit test)
  set_tests_properties(polycrystal-shear-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION
                       "polycrystal_shear status=ok grid=4x4x4 steps_completed=2"
                       TIMEOUT 180)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
     (NOT DEFINED OpenPFC_MPI_TEST_MAX_WORLD_SIZE OR
      OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
      OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
    add_test(NAME polycrystal-shear-smoke-mpi
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:polycrystal_shear>
                     ${MPIEXEC_POSTFLAGS}
                     --size 8
                     --steps 2
                     --shear 0.002
                     --grains 2
                     --outdir ${CMAKE_CURRENT_BINARY_DIR}/smoke-mpi
                     --commit test)
    set_tests_properties(polycrystal-shear-smoke-mpi PROPERTIES
                         PASS_REGULAR_EXPRESSION
                         "polycrystal_shear status=ok grid=8x8x8 steps_completed=2"
                         TIMEOUT 180 PROCESSORS 2)
    add_test(NAME polycrystal-shear-bad-outdir-mpi
             COMMAND ${CMAKE_COMMAND}
                     -DMODE=bad-outdir
                     -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                     -DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                     -DPREFLAGS=${MPIEXEC_PREFLAGS}
                     -DPOSTFLAGS=${MPIEXEC_POSTFLAGS}
                     -DEXE=$<TARGET_FILE:polycrystal_shear>
                     -DTIMEOUT=30
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_checks.cmake)
    set_tests_properties(polycrystal-shear-bad-outdir-mpi PROPERTIES
                         TIMEOUT 60 PROCESSORS 2)
    add_test(NAME polycrystal-shear-parity
             COMMAND ${CMAKE_COMMAND}
                     -DMODE=parity
                     -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                     -DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                     -DPREFLAGS=${MPIEXEC_PREFLAGS}
                     -DPOSTFLAGS=${MPIEXEC_POSTFLAGS}
                     -DEXE=$<TARGET_FILE:polycrystal_shear>
                     -DROOT=${CMAKE_CURRENT_BINARY_DIR}/parity
                     -DTIMEOUT=180
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_checks.cmake)
    set_tests_properties(polycrystal-shear-parity PROPERTIES
                         TIMEOUT 400 PROCESSORS 2)
  endif()

  add_test(NAME polycrystal-shear-render-missing
           COMMAND ${CMAKE_COMMAND}
                   -DMODE=render-missing
                   -DEXE=$<TARGET_FILE:polycrystal_shear>
                   -DROOT=${CMAKE_CURRENT_BINARY_DIR}/render-missing
                   -DTIMEOUT=180
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_checks.cmake)
  set_tests_properties(polycrystal-shear-render-missing PROPERTIES TIMEOUT 200)

  add_test(NAME polycrystal-shear-render-injection
           COMMAND ${CMAKE_COMMAND}
                   -DMODE=render-injection
                   -DEXE=$<TARGET_FILE:polycrystal_shear>
                   -DROOT=${CMAKE_CURRENT_BINARY_DIR}/render-injection
                   -DTIMEOUT=180
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_checks.cmake)
  set_tests_properties(polycrystal-shear-render-injection PROPERTIES TIMEOUT 200)

  find_program(FFMPEG_EXECUTABLE ffmpeg)
  if(FFMPEG_EXECUTABLE)
    add_test(NAME polycrystal-shear-render
             COMMAND ${CMAKE_COMMAND}
                     -DMODE=render
                     -DEXE=$<TARGET_FILE:polycrystal_shear>
                     -DFFMPEG=${FFMPEG_EXECUTABLE}
                     -DROOT=${CMAKE_CURRENT_BINARY_DIR}/render
                     -DTIMEOUT=180
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_checks.cmake)
    set_tests_properties(polycrystal-shear-render PROPERTIES TIMEOUT 200)
  endif()
endif()
