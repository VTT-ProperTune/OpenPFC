# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Issue #233. Periodic Fourier velocity. Not a copy of examples/ns2d_vorticity.

set(_flow_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_flow_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_flow_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_flow_heffte_tgt heffte)
endif()

add_executable(incompressible_flow
  src/main.cpp
  src/hit_series.cpp
  src/forced_series.cpp
  src/wall.cpp)
target_include_directories(incompressible_flow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(incompressible_flow PRIVATE OpenPFC ${_flow_heffte_tgt})
install(TARGETS incompressible_flow DESTINATION bin)

if(OpenPFC_BUILD_TESTS)
  # The driver calls MPI_Init itself. Do not wrap it with srun.
  add_test(NAME incompressible-flow-smoke
           COMMAND $<TARGET_FILE:incompressible_flow>
                   --case taylor-green
                   --n 16
                   --nu 0.05
                   --dt 0.015625
                   --time 0.03125
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/smoke)
  set_tests_properties(incompressible-flow-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
                       TIMEOUT 120)
  add_test(NAME incompressible-flow-hit-smoke
           COMMAND $<TARGET_FILE:incompressible_flow>
                   --case decaying-hit
                   --n 16
                   --nu 0.02
                   --dt 0.015625
                   --time 0.03125
                   --seed 1
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/hit-smoke)
  set_tests_properties(incompressible-flow-hit-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
                       TIMEOUT 120)
  add_test(NAME incompressible-flow-forced-smoke
           COMMAND $<TARGET_FILE:incompressible_flow>
                   --case forced-hit
                   --n 16
                   --nu 0.02
                   --dt 0.015625
                   --time 0.03125
                   --seed 1
                   --power 0.25
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/forced-smoke)
  set_tests_properties(incompressible-flow-forced-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
                       TIMEOUT 120)
endif()

if(OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
  add_executable(test_incompressible_flow
    tests/test_taylor_green.cpp
    tests/test_decaying_hit.cpp
    tests/test_forced_hit.cpp
    tests/test_channel.cpp
    src/wall.cpp)
  target_include_directories(test_incompressible_flow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_incompressible_flow PRIVATE OpenPFC Catch2::Catch2
                        ${_flow_heffte_tgt})
  add_test(NAME incompressible-flow-cpu COMMAND test_incompressible_flow)
  set_tests_properties(incompressible-flow-cpu PROPERTIES TIMEOUT 180)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
     (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
      OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
    add_test(NAME incompressible-flow-mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:test_incompressible_flow> ${MPIEXEC_POSTFLAGS} "[mpi]")
    set_tests_properties(incompressible-flow-mpi PROPERTIES
      TIMEOUT 180 PROCESSORS 2)
    add_test(NAME incompressible-flow-smoke-mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:incompressible_flow> ${MPIEXEC_POSTFLAGS}
        --case taylor-green --n 16 --nu 0.05 --dt 0.015625 --time 0.03125
        --outdir ${CMAKE_CURRENT_BINARY_DIR}/smoke-mpi)
    set_tests_properties(incompressible-flow-smoke-mpi PROPERTIES
      PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
      TIMEOUT 120 PROCESSORS 2)
  endif()
endif()

if(TARGET openpfc_incompressible_hip OR TARGET openpfc_incompressible_cuda)
  set(_flow_device_lib "")
  if(TARGET openpfc_incompressible_hip)
    set(_flow_device_lib openpfc_incompressible_hip)
    set(_flow_device_app incompressible_flow_hip)
    set(_flow_backend hip)
  elseif(TARGET openpfc_incompressible_cuda)
    set(_flow_device_lib openpfc_incompressible_cuda)
    set(_flow_device_app incompressible_flow_cuda)
    set(_flow_backend cuda)
  endif()
  # Same physics sources and one shared host-side session owner.
    add_executable(${_flow_device_app}
      src/main.cpp
      src/hit_series.cpp
      src/forced_series.cpp
      src/wall.cpp
      src/device_advance.cpp)
    target_include_directories(${_flow_device_app} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${_flow_device_app} PRIVATE
      OPENPFC_FLOW_DEVICE=1)
    target_link_libraries(${_flow_device_app} PRIVATE
      OpenPFC ${_flow_device_lib} ${_flow_heffte_tgt})
    install(TARGETS ${_flow_device_app} DESTINATION bin)
    if(OpenPFC_BUILD_TESTS)
      add_test(NAME incompressible-flow-${_flow_backend}-smoke
               COMMAND $<TARGET_FILE:${_flow_device_app}>
                       --case taylor-green
                       --n 16
                       --nu 0.05
                       --dt 0.015625
                       --time 0.03125
                       --outdir ${CMAKE_CURRENT_BINARY_DIR}/${_flow_backend}-smoke)
      set_tests_properties(incompressible-flow-${_flow_backend}-smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
        TIMEOUT 180)
      add_test(NAME incompressible-flow-${_flow_backend}-hit-smoke
        COMMAND $<TARGET_FILE:${_flow_device_app}>
          --case decaying-hit --n 16 --nu 0.02 --dt 0.015625 --time 0.03125
          --seed 1 --outdir ${CMAKE_CURRENT_BINARY_DIR}/${_flow_backend}-hit-smoke)
      set_tests_properties(incompressible-flow-${_flow_backend}-hit-smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "incompressible_flow status=ok" TIMEOUT 180)
      if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
         (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
          OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
        foreach(_flow_case taylor-green decaying-hit)
          add_test(NAME incompressible-flow-${_flow_backend}-${_flow_case}-smoke-mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:${_flow_device_app}> ${MPIEXEC_POSTFLAGS}
              --case ${_flow_case} --n 16 --nu 0.02 --dt 0.015625 --time 0.03125
              --outdir ${CMAKE_CURRENT_BINARY_DIR}/${_flow_backend}-${_flow_case}-smoke-mpi)
          set_tests_properties(incompressible-flow-${_flow_backend}-${_flow_case}-smoke-mpi
            PROPERTIES PASS_REGULAR_EXPRESSION "incompressible_flow status=ok"
            TIMEOUT 180 PROCESSORS 2)
        endforeach()
      endif()
    endif()
  add_executable(incompressible_device_parity
    tests/incompressible_device_parity.cpp src/device_advance.cpp)
  target_include_directories(incompressible_device_parity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(incompressible_device_parity PRIVATE
    OpenPFC ${_flow_device_lib} ${_flow_heffte_tgt})
  if(OpenPFC_BUILD_TESTS)
    add_test(NAME incompressible-flow-${_flow_backend}-local-smoke
      COMMAND $<TARGET_FILE:${_flow_device_app}>
        --case decaying-hit --n 16 --nu 0.02 --dt 0.015625 --time 0.03125
        --seed 1 --fft local --outdir ${CMAKE_CURRENT_BINARY_DIR}/${_flow_backend}-local-smoke)
    set_tests_properties(incompressible-flow-${_flow_backend}-local-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "incompressible_flow status=ok" TIMEOUT 180)
    add_test(NAME incompressible-flow-device-parity
             COMMAND $<TARGET_FILE:incompressible_device_parity> --n=16)
    set_tests_properties(incompressible-flow-device-parity PROPERTIES
      PASS_REGULAR_EXPRESSION "FLOW_DEVICE_PARITY_PASS"
      TIMEOUT 300)
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
       (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
        OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
      add_test(NAME incompressible-flow-device-parity-mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:incompressible_device_parity> ${MPIEXEC_POSTFLAGS} --n=16)
      set_tests_properties(incompressible-flow-device-parity-mpi PROPERTIES
        PASS_REGULAR_EXPRESSION "FLOW_DEVICE_PARITY_PASS"
        TIMEOUT 300 PROCESSORS 2)
      # The real CLI, not the parity helper. Rejection happens before any
      # collective and before a distributed plan is built.
      add_test(NAME incompressible-flow-${_flow_backend}-local-mpi-reject
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:${_flow_device_app}> ${MPIEXEC_POSTFLAGS}
          --case taylor-green --n 16 --nu 0.02 --dt 0.015625 --time 0.03125
          --fft local
          --outdir ${CMAKE_CURRENT_BINARY_DIR}/${_flow_backend}-local-mpi-reject)
      set_tests_properties(incompressible-flow-${_flow_backend}-local-mpi-reject
        PROPERTIES
          PASS_REGULAR_EXPRESSION "local transform requires one rank"
          FAIL_REGULAR_EXPRESSION "status=ok"
          TIMEOUT 180 PROCESSORS 2)
    endif()
  endif()
endif()
