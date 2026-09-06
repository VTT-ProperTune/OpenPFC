# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(HEAT3D_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_heat3d_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_heat3d_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_heat3d_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_heat3d_heffte_tgt heffte)
endif()

find_package(OpenMP COMPONENTS CXX)

# Per-method binaries (model hierarchy, primitive -> compact):
#   heat3d_fd_scratch          — version 0; bare triple loops, raw pointers,
#                                only HaloExchange from openpfc
#   heat3d_fd_manual           — laboratory-style explicit-stencil FD driver
#   heat3d_fd                  — compact FD (pfc::data::Field + PaddedHalo +
#                                FDGradient + for_each, decoupled in main)
#   heat3d_fd_hip              — 3D FD on HIP (FDGPUStack + device halo +
#                                for_each_interior_device); LUMI-G scaling
#   heat3d_spectral_pointwise  — point-wise spectral RHS + explicit Euler
#   heat3d_spectral            — implicit-Euler spectral
#   heat3d_spectral_hip        — implicit-Euler spectral on HIP (2 FFTs/step)
set(_heat3d_binaries heat3d_fd_scratch heat3d_fd_manual heat3d_fd)
if(OpenPFC_ENABLE_HEFFTE)
  list(APPEND _heat3d_binaries heat3d_spectral_pointwise heat3d_spectral)
endif()

# Catch2-based unit tests for HeatModel + field stack/Euler integration.
# Mirrors the gating used by apps/tungsten and apps/aluminumNew so the test
# target only builds when the project was configured with tests enabled and
# Catch2 is actually available. Declared before HIP smokes so the option is
# defined on the first configure.
option(HEAT3D_ENABLE_TESTS "Enable building heat3d tests" ON)
foreach(_target IN LISTS _heat3d_binaries)
  add_executable(${_target} src/cpu/${_target}.cpp)
  target_include_directories(${_target} PRIVATE ${HEAT3D_INCLUDE_DIR})
  target_link_libraries(${_target} PRIVATE OpenPFC openpfc_apps_common
                                           ${_heat3d_heffte_tgt})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${_target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  install(TARGETS ${_target} DESTINATION bin)
endforeach()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(heat3d_fd_hip
    src/hip/heat3d_fd_hip.cpp
    src/hip/heat3d_fd_hip_kernels.hip
  )
  target_include_directories(heat3d_fd_hip PRIVATE ${HEAT3D_INCLUDE_DIR})
  target_link_libraries(heat3d_fd_hip PRIVATE OpenPFC openpfc_apps_common
                        hip::host openpfc_hip_kernels ${_heat3d_heffte_tgt})
  set_target_properties(heat3d_fd_hip PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
  )
  install(TARGETS heat3d_fd_hip DESTINATION bin)
  message(STATUS "heat3d_fd_hip (MPI + HIP) enabled")

  if(HEAT3D_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME heat3d-fd-hip-smoke
             COMMAND $<TARGET_FILE:heat3d_fd_hip> 16 4 0.01 2)
    set_tests_properties(heat3d-fd-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "HEAT3D_HIP_CHECKSUM"
      TIMEOUT 180)
  endif()

  if(OpenPFC_ENABLE_HIP_SPECTRAL)
    add_executable(heat3d_spectral_hip src/hip/heat3d_spectral_hip.cpp)
    target_include_directories(heat3d_spectral_hip PRIVATE ${HEAT3D_INCLUDE_DIR})
    target_link_libraries(heat3d_spectral_hip PRIVATE OpenPFC openpfc_apps_common
                          hip::host openpfc_hip_kernels ${_heat3d_heffte_tgt})
    set_target_properties(heat3d_spectral_hip PROPERTIES
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON
    )
    install(TARGETS heat3d_spectral_hip DESTINATION bin)
    message(STATUS "heat3d_spectral_hip (MPI + HIP spectral) enabled")

    if(HEAT3D_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
      include(CTest)
      add_test(NAME heat3d-spectral-hip-smoke
               COMMAND $<TARGET_FILE:heat3d_spectral_hip> 16 4 0.01)
      set_tests_properties(heat3d-spectral-hip-smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "HEAT3D_SPECTRAL_HIP_CHECKSUM"
        TIMEOUT 180)
    endif()
  else()
    message(STATUS
      "heat3d_spectral_hip disabled: ROCm HeFFTe (rocFFT) missing (HIP spectral support OFF).")
  endif()
endif()

if(HEAT3D_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    add_executable(test_heat3d tests/test_heat3d.cpp)
    target_include_directories(test_heat3d PRIVATE ${HEAT3D_INCLUDE_DIR})
    target_link_libraries(test_heat3d PRIVATE
      OpenPFC
      openpfc_apps_common
      ${_heat3d_heffte_tgt}
      Catch2::Catch2
    )
    include(CTest)
    # Run the whole heat3d suite in a single invocation to avoid per-test MPI
    # init/finalize overhead (matches tungsten-all-tests / aluminum-all-tests).
    add_test(NAME heat3d-all-tests COMMAND test_heat3d)

    # Operator-evaluation contract tests
    add_executable(test_heat3d_operator_evaluation tests/test_heat3d_operator_evaluation.cpp)
    target_include_directories(test_heat3d_operator_evaluation PRIVATE ${HEAT3D_INCLUDE_DIR})
    target_link_libraries(test_heat3d_operator_evaluation PRIVATE
      OpenPFC
      openpfc_apps_common
      ${_heat3d_heffte_tgt}
      Catch2::Catch2
    )
    add_test(NAME heat3d-operator-evaluation COMMAND test_heat3d_operator_evaluation)

    # rhs() pattern structural tests (not a numerical vs-legacy baseline)
    add_executable(test_heat3d_rhs_pattern tests/test_heat3d_rhs_pattern.cpp)
    target_include_directories(test_heat3d_rhs_pattern PRIVATE ${HEAT3D_INCLUDE_DIR})
    target_link_libraries(test_heat3d_rhs_pattern PRIVATE
      OpenPFC
      openpfc_apps_common
      ${_heat3d_heffte_tgt}
      Catch2::Catch2
    )
    add_test(NAME heat3d-rhs-pattern COMMAND test_heat3d_rhs_pattern)
  else()
    message(STATUS "heat3d: Catch2 not found, not building tests")
  endif()
endif()
