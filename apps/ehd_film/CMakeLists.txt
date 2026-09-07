# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# EHD film under a flexible plate: k^6 spectral ETD, CPU + HIP.

option(EHD_FILM_ENABLE_TESTS "Enable building ehd-film tests" ON)

set(EHD_FILM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_ehd_film_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_ehd_film_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_ehd_film_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_ehd_film_heffte_tgt heffte)
endif()

function(ehd_film_cpu_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${EHD_FILM_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_ehd_film_heffte_tgt})
endfunction()

function(ehd_film_hip_executable target)
  add_executable(${target} ${ARGN} src/gpu/ehd_film_pointwise.hip)
  target_include_directories(${target} PRIVATE ${EHD_FILM_INCLUDE_DIR})
  target_link_libraries(${target} PRIVATE OpenPFC openpfc_apps_common
                        nlohmann_json::nlohmann_json ${_ehd_film_heffte_tgt}
                        hip::host openpfc_hip_kernels)
  set_target_properties(${target} PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
endfunction()

ehd_film_cpu_executable(ehd_film src/ehd_film.cpp)
install(TARGETS ehd_film DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  ehd_film_hip_executable(ehd_film_hip src/hip/ehd_film.cpp)
  install(TARGETS ehd_film_hip DESTINATION bin)
  message(STATUS "ehd_film_hip (MPI + HIP spectral) enabled")
  if(EHD_FILM_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    include(CTest)
    add_test(NAME ehd-film-hip-smoke
             COMMAND $<TARGET_FILE:ehd_film_hip>
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/ehd_film_hip_smoke.json)
    set_tests_properties(ehd-film-hip-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "SPECTRAL_CHECKSUM"
      TIMEOUT 180)
  endif()
endif()

if(EHD_FILM_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    ehd_film_cpu_executable(test_ehd_film tests/test_ehd_film.cpp)
    target_link_libraries(test_ehd_film PRIVATE Catch2::Catch2)
    include(CTest)
    add_test(NAME ehd-film-all-tests COMMAND test_ehd_film)

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
      ehd_film_hip_executable(test_ehd_film_hip tests/test_ehd_film_gpu.cpp)
      target_include_directories(test_ehd_film_hip PRIVATE
          ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
      target_link_libraries(test_ehd_film_hip PRIVATE Catch2::Catch2 MPI::MPI_CXX)
      add_test(NAME HIP_EhdFilmETD COMMAND test_ehd_film_hip)
    endif()
  else()
    message(STATUS "EHD film: Catch2 not found, not building tests")
  endif()
endif()
