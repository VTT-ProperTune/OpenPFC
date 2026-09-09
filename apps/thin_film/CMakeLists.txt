# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Thin-film lubrication: one physics header, JSON SpectralETDSession, CPU + HIP.

option(THIN_FILM_ENABLE_TESTS "Enable building thin-film tests" ON)

set(THIN_FILM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_thin_film_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_thin_film_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_thin_film_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_thin_film_heffte_tgt heffte)
endif()

function(thin_film_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${THIN_FILM_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_thin_film_heffte_tgt})
endfunction()

function(thin_film_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/thin_film_pointwise.hip)
  target_include_directories(${target} PRIVATE ${THIN_FILM_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_thin_film_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

thin_film_cpu_executable(thin_film src/thin_film.cpp)
install(TARGETS thin_film DESTINATION bin)

# Science driver for #114: full h^3 lubrication mobility through the shared
# conservative flux stepper (pfc::apps::SpectralFlux / FluxETD), templated on
# MemorySpace -- host build here, HIP twin below.
thin_film_cpu_executable(thin_film_nonlinear src/thin_film_nonlinear.cpp)
install(TARGETS thin_film_nonlinear DESTINATION bin)

# Two-method flagship (#124): the same h^3 lubrication equation, integrated by
# a conservative face-flux FD scheme (MPI Cartesian decomposition + halo
# exchange) instead of spectral ETD, so it can be pushed through rupture
# where thin_film_nonlinear structurally cannot. Same JSON case files.
thin_film_cpu_executable(thin_film_fd src/thin_film_fd.cpp)
install(TARGETS thin_film_fd DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  thin_film_hip_executable(thin_film_hip src/hip/thin_film.cpp)
  install(TARGETS thin_film_hip DESTINATION bin)
  message(STATUS "thin_film_hip (MPI + HIP spectral) enabled")

  thin_film_hip_executable(thin_film_nonlinear_hip src/hip/thin_film_nonlinear.cpp)
  install(TARGETS thin_film_nonlinear_hip DESTINATION bin)
  message(STATUS "thin_film_nonlinear_hip (MPI + HIP spectral flux) enabled")
  if(THIN_FILM_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME thin-film-hip-smoke
             COMMAND $<TARGET_FILE:thin_film_hip>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/thin_film_hip_smoke.json)
    set_tests_properties(thin-film-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "SPECTRAL_CHECKSUM"
      TIMEOUT 180)
  endif()
endif()

if(THIN_FILM_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    thin_film_cpu_executable(test_thin_film tests/test_thin_film.cpp)
    target_link_libraries(test_thin_film PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME thin-film-all-tests COMMAND test_thin_film)

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      thin_film_hip_executable(test_thin_film_hip tests/test_thin_film_gpu.cpp)
      target_include_directories(test_thin_film_hip PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_thin_film_hip PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      add_test(NAME HIP_ThinFilmETD COMMAND test_thin_film_hip)
    endif()
  else()
    message(STATUS "Thin-film: Catch2 not found, not building tests")
  endif()
endif()
