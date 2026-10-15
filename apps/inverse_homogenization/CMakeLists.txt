# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Inverse homogenization (issue #161). Forward C_H plus Allen–Cahn
# descent on the tensor-mismatch objective. Not a general optimizer.

option(INVERSE_HOMOGENIZATION_ENABLE_TESTS
       "Enable building inverse-homogenization smoke tests" ON)

set(INVERSE_HOMOGENIZATION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_invhom_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_invhom_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_invhom_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_invhom_heffte_tgt heffte)
endif()

function(inverse_homogenization_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${INVERSE_HOMOGENIZATION_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC
                        ${_invhom_heffte_tgt})
endfunction()

inverse_homogenization_cpu_executable(openpfc_homogenize src/homogenize.cpp)
inverse_homogenization_cpu_executable(openpfc_inverse_homogenize
                                      src/inverse_homogenize.cpp)
add_executable(openpfc_finite_strain_forward src/finite_strain_forward.cpp)
target_include_directories(openpfc_finite_strain_forward PRIVATE
                           ${INVERSE_HOMOGENIZATION_INCLUDE_DIR})
set_target_properties(openpfc_finite_strain_forward PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON)
add_executable(openpfc_finite_strain_inverse src/finite_strain_inverse.cpp)
target_include_directories(openpfc_finite_strain_inverse PRIVATE
                           ${INVERSE_HOMOGENIZATION_INCLUDE_DIR})
set_target_properties(openpfc_finite_strain_inverse PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON)
install(TARGETS openpfc_homogenize openpfc_inverse_homogenize
                openpfc_finite_strain_forward openpfc_finite_strain_inverse
        DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  function(inverse_homogenization_hip_executable target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${INVERSE_HOMOGENIZATION_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE OpenPFC
                          openpfc_microelasticity_hip
                          ${_invhom_heffte_tgt} hip::host openpfc_hip_kernels)
    set_target_properties(${target} PROPERTIES
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON)
  endfunction()
  inverse_homogenization_hip_executable(openpfc_homogenize_hip
                                        src/hip/homogenize_hip.cpp)
  inverse_homogenization_hip_executable(openpfc_inverse_homogenize_hip
                                        src/hip/inverse_homogenize_hip.cpp)
  target_link_libraries(openpfc_inverse_homogenize_hip PRIVATE nlohmann_json::nlohmann_json)
  install(TARGETS openpfc_homogenize_hip openpfc_inverse_homogenize_hip
          DESTINATION bin)
  message(STATUS "openpfc_homogenize_hip / openpfc_inverse_homogenize_hip (HIP Green) enabled")
  if(INVERSE_HOMOGENIZATION_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    add_test(NAME inverse-homogenization-hip-homogeneous-smoke
             COMMAND $<TARGET_FILE:openpfc_homogenize_hip>
                     --shape=homogeneous --nx=8 --ny=8 --nz=8 --volume=1)
    set_tests_properties(inverse-homogenization-hip-homogeneous-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "HOMOGENIZATION_CHECKSUM"
      TIMEOUT 180)
    add_test(NAME inverse-homogenization-hip-ac-smoke
             COMMAND $<TARGET_FILE:openpfc_inverse_homogenize_hip>
                     --nx=8 --ny=8 --nz=1 --steps=1
                     --init=rotating-squares --lambda-reg=0 --dt=0.05)
    set_tests_properties(inverse-homogenization-hip-ac-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "INVERSE_CHECKSUM"
      TIMEOUT 300)
  endif()
endif()

if(INVERSE_HOMOGENIZATION_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  include(CTest)
  add_test(NAME inverse-homogenization-homogeneous-smoke
           COMMAND $<TARGET_FILE:openpfc_homogenize>
                   --shape=homogeneous --nx=8 --ny=8 --nz=8 --volume=1)
  set_tests_properties(inverse-homogenization-homogeneous-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "HOMOGENIZATION_CHECKSUM"
    TIMEOUT 180)
  add_test(NAME inverse-homogenization-rotating-cubes-smoke
           COMMAND $<TARGET_FILE:openpfc_homogenize>
                   --shape=rotating-cubes --nx=16 --ny=16 --nz=16
                   --E-solid=1 --nu-solid=0.3 --E-void=0.05 --nu-void=0.3)
  set_tests_properties(inverse-homogenization-rotating-cubes-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "HOMOGENIZATION_CHECKSUM"
    TIMEOUT 180)
  add_test(NAME inverse-homogenization-rotating-squares-extrusion-smoke
           COMMAND $<TARGET_FILE:openpfc_homogenize>
                   --shape=rotating-squares --nx=16 --ny=16 --nz=4
                   --E-solid=1 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3
                   --half=0.200 --angle=0.45)
  set_tests_properties(inverse-homogenization-rotating-squares-extrusion-smoke
                       PROPERTIES
    PASS_REGULAR_EXPRESSION "HOMOGENIZATION_CHECKSUM"
    TIMEOUT 180)
  add_test(NAME inverse-homogenization-reentrant-3d-smoke
           COMMAND $<TARGET_FILE:openpfc_homogenize>
                   --shape=reentrant-3d --nx=16 --ny=16 --nz=16
                   --E-solid=1 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3
                   --thickness=0.06 --inset=0.14)
  set_tests_properties(inverse-homogenization-reentrant-3d-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "HOMOGENIZATION_CHECKSUM"
    TIMEOUT 180)
  add_test(NAME inverse-homogenization-yang-a3-smoke
           COMMAND $<TARGET_FILE:openpfc_homogenize>
                   --shape=yang-a3 --nx=8 --ny=8 --nz=16
                   --E-solid=1 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3)
  set_tests_properties(inverse-homogenization-yang-a3-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "HOMOGENIZATION_CHECKSUM"
    TIMEOUT 180)
  add_test(NAME inverse-homogenization-ac-smoke
           COMMAND $<TARGET_FILE:openpfc_inverse_homogenize>
                   --target=isotropic --nx=8 --ny=8 --nz=8 --steps=2
                   --init=uniform --init-volume=0.8 --lambda-reg=0 --dt=0.1)
  set_tests_properties(inverse-homogenization-ac-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "INVERSE_CHECKSUM"
    TIMEOUT 180)
  add_test(NAME inverse-homogenization-finite-strain-smoke
           COMMAND $<TARGET_FILE:openpfc_finite_strain_forward>)
  set_tests_properties(inverse-homogenization-finite-strain-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "FINITE_STRAIN_CHECKSUM"
    TIMEOUT 60)
  add_test(NAME inverse-homogenization-finite-strain-inverse-smoke
           COMMAND $<TARGET_FILE:openpfc_finite_strain_inverse>)
  set_tests_properties(inverse-homogenization-finite-strain-inverse-smoke
                       PROPERTIES
    PASS_REGULAR_EXPRESSION "FINITE_STRAIN_INVERSE_CHECKSUM"
    TIMEOUT 120)

  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    inverse_homogenization_cpu_executable(test_inverse_homogenization
                                          tests/test_inverse_homogenization.cpp)
    target_link_libraries(test_inverse_homogenization PRIVATE Catch2::Catch2
                          nlohmann_json::nlohmann_json)
    add_test(NAME inverse-homogenization-ac-tests
             COMMAND test_inverse_homogenization)
    set_tests_properties(inverse-homogenization-ac-tests PROPERTIES TIMEOUT 600)
    inverse_homogenization_cpu_executable(test_homogenization
                                          tests/test_homogenization.cpp)
    target_link_libraries(test_homogenization PRIVATE Catch2::Catch2)
    add_test(NAME inverse-homogenization-forward COMMAND test_homogenization)
    set_tests_properties(inverse-homogenization-forward PROPERTIES TIMEOUT 300)
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
       (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
        OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
      add_test(NAME inverse-homogenization-forward-mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:test_homogenization> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(inverse-homogenization-forward-mpi PROPERTIES
        TIMEOUT 300 PROCESSORS 2)
    endif()
    add_executable(test_finite_strain_forward
                   tests/test_finite_strain_forward.cpp)
    target_include_directories(test_finite_strain_forward PRIVATE
                               ${INVERSE_HOMOGENIZATION_INCLUDE_DIR})
    target_link_libraries(test_finite_strain_forward PRIVATE Catch2::Catch2)
    set_target_properties(test_finite_strain_forward PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON)
    add_test(NAME inverse-homogenization-finite-strain-tests
             COMMAND test_finite_strain_forward)
    set_tests_properties(inverse-homogenization-finite-strain-tests
                         PROPERTIES TIMEOUT 60)
    add_executable(test_finite_strain_grid tests/test_finite_strain_grid.cpp)
    target_include_directories(test_finite_strain_grid PRIVATE
                               ${INVERSE_HOMOGENIZATION_INCLUDE_DIR})
    target_link_libraries(test_finite_strain_grid PRIVATE Catch2::Catch2)
    set_target_properties(test_finite_strain_grid PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON)
    add_test(NAME inverse-homogenization-finite-strain-grid-tests
             COMMAND test_finite_strain_grid)
    set_tests_properties(inverse-homogenization-finite-strain-grid-tests
                         PROPERTIES TIMEOUT 120)
    add_executable(test_inverse_convergence
                   tests/test_inverse_convergence.cpp)
    # Header-only: the app headers and the library SIMP law they use.
    target_include_directories(test_inverse_convergence PRIVATE
                               ${INVERSE_HOMOGENIZATION_INCLUDE_DIR}
                               ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(test_inverse_convergence PRIVATE Catch2::Catch2)
    set_target_properties(test_inverse_convergence PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON)
    add_test(NAME inverse-homogenization-convergence-tests
             COMMAND test_inverse_convergence)
    set_tests_properties(inverse-homogenization-convergence-tests
                         PROPERTIES TIMEOUT 30)
  endif()
endif()
