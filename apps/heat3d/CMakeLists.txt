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
#                                only PaddedHaloExchanger from openpfc
#   heat3d_fd_manual           — laboratory-style explicit-stencil FD driver
#   heat3d_fd                  — compact FD driver (FdCpuStack + EulerStepper)
#   heat3d_spectral_pointwise  — point-wise spectral RHS + explicit Euler
#   heat3d_spectral            — implicit-Euler spectral
set(_heat3d_binaries heat3d_fd_scratch heat3d_fd_manual heat3d_fd
                     heat3d_spectral_pointwise heat3d_spectral)
foreach(_target IN LISTS _heat3d_binaries)
  add_executable(${_target} src/cpu/${_target}.cpp)
  target_include_directories(${_target} PRIVATE ${HEAT3D_INCLUDE_DIR})
  target_link_libraries(${_target} PRIVATE OpenPFC ${_heat3d_heffte_tgt})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${_target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  install(TARGETS ${_target} DESTINATION bin)
endforeach()

# Catch2-based unit tests for HeatModel + LocalField/Euler integration.
# Mirrors the gating used by apps/tungsten and apps/aluminumNew so the test
# target only builds when the project was configured with tests enabled and
# Catch2 is actually available.
option(HEAT3D_ENABLE_TESTS "Enable building heat3d tests" ON)

if(HEAT3D_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    add_executable(test_heat3d tests/test_heat3d.cpp)
    target_include_directories(test_heat3d PRIVATE ${HEAT3D_INCLUDE_DIR})
    target_link_libraries(test_heat3d PRIVATE
      OpenPFC
      ${_heat3d_heffte_tgt}
      Catch2::Catch2
    )
    include(CTest)
    # Run the whole heat3d suite in a single invocation to avoid per-test MPI
    # init/finalize overhead (matches tungsten-all-tests / aluminum-all-tests).
    add_test(NAME heat3d-all-tests COMMAND test_heat3d)
  else()
    message(STATUS "heat3d: Catch2 not found, not building tests")
  endif()
endif()
