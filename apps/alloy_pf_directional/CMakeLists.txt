# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(ALLOY_PF_DIRECTIONAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_dir_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_dir_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_dir_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_dir_heffte_tgt heffte)
endif()

find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
  add_executable(alloy_pf_directional_openmp
    src/cpu/alloy_pf_directional_openmp.cpp
    src/cpu/alloy_pf_directional_engine.cpp)
  target_include_directories(alloy_pf_directional_openmp PRIVATE
    ${ALLOY_PF_DIRECTIONAL_INCLUDE_DIR})
  target_link_libraries(alloy_pf_directional_openmp PRIVATE OpenPFC ${_dir_heffte_tgt} OpenMP::OpenMP_CXX)
  install(TARGETS alloy_pf_directional_openmp DESTINATION bin)
  if(OpenPFC_BUILD_TESTS)
    add_test(NAME alloy-pf-directional-repro
             COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_repro.sh"
                     "$<TARGET_FILE:alloy_pf_directional_openmp>"
                     "${CMAKE_CURRENT_BINARY_DIR}/repro"
                     1)
    set_tests_properties(alloy-pf-directional-repro PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      TIMEOUT 120)
  endif()
else()
  message(STATUS "alloy_pf_directional_openmp skipped: OpenMP CXX not found")
endif()

add_executable(alloy_pf_directional_mpi src/cpu/alloy_pf_directional_mpi.cpp)
target_include_directories(alloy_pf_directional_mpi PRIVATE
  ${ALLOY_PF_DIRECTIONAL_INCLUDE_DIR})
target_link_libraries(alloy_pf_directional_mpi PRIVATE OpenPFC ${_dir_heffte_tgt})
install(TARGETS alloy_pf_directional_mpi DESTINATION bin)

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(alloy_pf_directional_hip
    src/hip/alloy_pf_directional_hip.cpp
    src/hip/alloy_pf_directional_kernels.hip)
  target_include_directories(alloy_pf_directional_hip PRIVATE
    ${ALLOY_PF_DIRECTIONAL_INCLUDE_DIR})
  target_link_libraries(alloy_pf_directional_hip PRIVATE OpenPFC hip::host ${_dir_heffte_tgt})
  set_target_properties(alloy_pf_directional_hip PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON)
  target_compile_definitions(alloy_pf_directional_hip PRIVATE OpenPFC_ENABLE_HIP)
  install(TARGETS alloy_pf_directional_hip DESTINATION bin)
  message(STATUS "alloy_pf_directional_hip (MPI + HIP) enabled")
endif()
