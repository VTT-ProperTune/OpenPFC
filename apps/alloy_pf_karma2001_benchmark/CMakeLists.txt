# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# OpenMP CPU binary for the Karma 2001 present-model isothermal dendrite
# (PRL 87, 115701). Does not share sources with other alloy PF apps.

set(ALLOY_PF_KARMA2001_BENCHMARK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_karma_bm_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_karma_bm_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_karma_bm_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_karma_bm_heffte_tgt heffte)
endif()

find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
  add_executable(alloy_pf_karma2001_benchmark_openmp
    src/cpu/alloy_pf_karma2001_benchmark_openmp.cpp
    src/cpu/alloy_pf_karma2001_benchmark_engine.cpp)
  target_include_directories(alloy_pf_karma2001_benchmark_openmp PRIVATE ${ALLOY_PF_KARMA2001_BENCHMARK_INCLUDE_DIR})
  target_link_libraries(alloy_pf_karma2001_benchmark_openmp PRIVATE OpenPFC ${_karma_bm_heffte_tgt} OpenMP::OpenMP_CXX)
  install(TARGETS alloy_pf_karma2001_benchmark_openmp DESTINATION bin)
  if(OpenPFC_BUILD_TESTS)
    add_test(NAME alloy-pf-karma2001-benchmark-smoke
             COMMAND alloy_pf_karma2001_benchmark_openmp smoke 1)
    set_tests_properties(alloy-pf-karma2001-benchmark-smoke PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      ENVIRONMENT "OPENPFC_KARMA_SKIP_PNG=1;OPENPFC_KARMA_QUIET=1;OPENPFC_KARMA_MAX_STEPS=2"
      TIMEOUT 120)
  endif()
else()
  message(STATUS "alloy_pf_karma2001_benchmark_openmp skipped: OpenMP CXX not found")
endif()
