# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Mullins surface diffusion: linear k^4 spectral ETD, CPU + HIP.

option(SURFACE_DIFFUSION_ENABLE_TESTS "Enable building surface-diffusion tests" ON)

set(SURFACE_DIFFUSION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_surface_diffusion_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_surface_diffusion_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_surface_diffusion_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_surface_diffusion_heffte_tgt heffte)
endif()

function(surface_diffusion_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${SURFACE_DIFFUSION_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_surface_diffusion_heffte_tgt})
endfunction()

function(surface_diffusion_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/surface_diffusion_pointwise.hip)
  target_include_directories(${target} PRIVATE ${SURFACE_DIFFUSION_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_surface_diffusion_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

surface_diffusion_cpu_executable(surface_diffusion src/surface_diffusion.cpp)
install(TARGETS surface_diffusion DESTINATION bin)

# Science driver for #115: orientation-dependent surface stiffness B(theta)
# through a self-contained spectral flux stepper. Host only, like thin_film's
# nonlinear driver -- the directional mobility needs a complex i*k_d multiply
# the device Ops layer does not expose.
surface_diffusion_cpu_executable(surface_diffusion_anisotropic
  src/surface_diffusion_anisotropic.cpp)
install(TARGETS surface_diffusion_anisotropic DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  surface_diffusion_hip_executable(surface_diffusion_hip
    src/hip/surface_diffusion.cpp)
  install(TARGETS surface_diffusion_hip DESTINATION bin)
  message(STATUS "surface_diffusion_hip (MPI + HIP spectral) enabled")
  if(SURFACE_DIFFUSION_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME surface-diffusion-hip-smoke
             COMMAND $<TARGET_FILE:surface_diffusion_hip>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/surface_diffusion_hip_smoke.json)
    set_tests_properties(surface-diffusion-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "SPECTRAL_CHECKSUM"
      TIMEOUT 180)
  endif()
endif()

if(SURFACE_DIFFUSION_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    surface_diffusion_cpu_executable(test_surface_diffusion
      tests/test_surface_diffusion.cpp)
    target_link_libraries(test_surface_diffusion PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME surface-diffusion-all-tests COMMAND test_surface_diffusion)

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      surface_diffusion_hip_executable(test_surface_diffusion_hip
        tests/test_surface_diffusion_gpu.cpp)
      target_include_directories(test_surface_diffusion_hip PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_surface_diffusion_hip PRIVATE Catch2::Catch2
                            MPI::MPI_CXX)
      add_test(NAME HIP_SurfaceDiffusionETD COMMAND test_surface_diffusion_hip)
    endif()
  else()
    message(STATUS "Surface-diffusion: Catch2 not found, not building tests")
  endif()
endif()
