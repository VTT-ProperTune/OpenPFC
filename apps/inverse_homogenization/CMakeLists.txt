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
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        ${_invhom_heffte_tgt})
endfunction()

inverse_homogenization_cpu_executable(openpfc_homogenize src/homogenize.cpp)
inverse_homogenization_cpu_executable(openpfc_inverse_homogenize
                                      src/inverse_homogenize.cpp)
install(TARGETS openpfc_homogenize openpfc_inverse_homogenize DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  function(inverse_homogenization_hip_executable target)
    add_executable(${target} ${ARGN}
      src/hip/microelasticity_kernels.hip)
    target_include_directories(${target} PRIVATE ${INVERSE_HOMOGENIZATION_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                          ${_invhom_heffte_tgt} hip::host openpfc_hip_kernels)
    set_target_properties(${target} PROPERTIES
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON)
  endfunction()
  inverse_homogenization_hip_executable(openpfc_homogenize_hip
                                        src/hip/homogenize_hip.cpp)
  inverse_homogenization_hip_executable(openpfc_inverse_homogenize_hip
                                        src/hip/inverse_homogenize_hip.cpp)
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
  add_test(NAME inverse-homogenization-ac-smoke
           COMMAND $<TARGET_FILE:openpfc_inverse_homogenize>
                   --target=isotropic --nx=8 --ny=8 --nz=8 --steps=2
                   --init=uniform --init-volume=0.8 --lambda-reg=0 --dt=0.1)
  set_tests_properties(inverse-homogenization-ac-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "INVERSE_CHECKSUM"
    TIMEOUT 180)

  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    inverse_homogenization_cpu_executable(test_inverse_homogenization
                                          tests/test_inverse_homogenization.cpp)
    target_link_libraries(test_inverse_homogenization PRIVATE Catch2::Catch2)
    add_test(NAME inverse-homogenization-ac-tests
             COMMAND test_inverse_homogenization)
    set_tests_properties(inverse-homogenization-ac-tests PROPERTIES TIMEOUT 600)
  endif()
endif()
