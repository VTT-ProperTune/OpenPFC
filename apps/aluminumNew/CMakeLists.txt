# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.15)
project(Aluminum VERSION 0.1.0 LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    find_package(OpenPFC REQUIRED)
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
    find_package(nlohmann_json REQUIRED)
    set(CMAKE_CXX_STANDARD 20)
endif()

include_directories(${CMAKE_SOURCE_DIR}/apps/common/include)

add_executable(aluminumNew aluminumNew.cpp)
if(TARGET Heffte::Heffte)
    target_link_libraries(aluminumNew OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)
elseif(TARGET Heffte)
    target_link_libraries(aluminumNew OpenPFC nlohmann_json::nlohmann_json Heffte)
elseif(TARGET heffte)
    target_link_libraries(aluminumNew OpenPFC nlohmann_json::nlohmann_json heffte)
else()
    target_link_libraries(aluminumNew OpenPFC nlohmann_json::nlohmann_json)
endif()

# M9 A/B CPU driver: MovingFrameMeanFieldETDSystem (Gen-1 aluminumNew stays).
add_executable(aluminum_etd aluminum_etd.cpp)
target_include_directories(aluminum_etd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(TARGET Heffte::Heffte)
    target_link_libraries(aluminum_etd OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte)
elseif(TARGET Heffte)
    target_link_libraries(aluminum_etd OpenPFC nlohmann_json::nlohmann_json Heffte)
elseif(TARGET heffte)
    target_link_libraries(aluminum_etd OpenPFC nlohmann_json::nlohmann_json heffte)
else()
    target_link_libraries(aluminum_etd OpenPFC nlohmann_json::nlohmann_json)
endif()
install(TARGETS aluminum_etd DESTINATION bin)

if(OpenPFC_ENABLE_CUDA_SPECTRAL)
  add_executable(aluminum_etd_cuda aluminum_etd_cuda.cpp)
  target_include_directories(aluminum_etd_cuda PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(aluminum_etd_cuda PRIVATE
      OpenPFC
      nlohmann_json::nlohmann_json
      $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
      CUDA::cudart
      openpfc_gpu_kernels)
  install(TARGETS aluminum_etd_cuda DESTINATION bin)
  message(STATUS "aluminum_etd_cuda enabled")
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
  add_executable(aluminum_etd_hip aluminum_etd_hip.cpp)
  target_include_directories(aluminum_etd_hip PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(aluminum_etd_hip PRIVATE
      OpenPFC
      nlohmann_json::nlohmann_json
      $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
      hip::host
      openpfc_hip_kernels)
  set_target_properties(aluminum_etd_hip PROPERTIES
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON)
  install(TARGETS aluminum_etd_hip DESTINATION bin)
  message(STATUS "aluminum_etd_hip enabled")
endif()

# Only build tests if tests are enabled for the parent project
option(ALUMINUM_ENABLE_TESTS "Enable building Aluminum tests" ON)

if(ALUMINUM_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
    find_package(Catch2 QUIET)
    if(TARGET Catch2::Catch2WithMain)
        add_executable(aluminumTest aluminumTest.cpp tests/test_aluminum_physics.cpp)
        target_include_directories(aluminumTest PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
                if(TARGET Heffte::Heffte)
                    target_link_libraries(aluminumTest OpenPFC nlohmann_json::nlohmann_json Heffte::Heffte Catch2::Catch2WithMain)
                elseif(TARGET Heffte)
                    target_link_libraries(aluminumTest OpenPFC nlohmann_json::nlohmann_json Heffte Catch2::Catch2WithMain)
                elseif(TARGET heffte)
                    target_link_libraries(aluminumTest OpenPFC nlohmann_json::nlohmann_json heffte Catch2::Catch2WithMain)
                else()
                    target_link_libraries(aluminumTest OpenPFC nlohmann_json::nlohmann_json Catch2::Catch2WithMain)
                endif()
        include(CTest)
        # Run all aluminum tests in a single invocation to avoid per-test MPI overhead
        add_test(NAME aluminum-all-tests COMMAND aluminumTest)

        if(OpenPFC_ENABLE_CUDA_SPECTRAL)
            add_executable(test_aluminum_etd_cuda tests/test_aluminum_etd_gpu.cpp)
            target_include_directories(test_aluminum_etd_cuda PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
            target_link_libraries(test_aluminum_etd_cuda PRIVATE
                OpenPFC
                nlohmann_json::nlohmann_json
                $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
                Catch2::Catch2
                CUDA::cudart
                openpfc_gpu_kernels
                MPI::MPI_CXX)
            target_compile_definitions(test_aluminum_etd_cuda PRIVATE
                OPENPFC_TEST_ALUMINUM_ETD_CUDA)
            add_test(NAME CUDA_AluminumETD COMMAND test_aluminum_etd_cuda)
        endif()

        if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
            add_executable(test_aluminum_etd_hip tests/test_aluminum_etd_gpu.cpp)
            target_include_directories(test_aluminum_etd_hip PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/tests/unit/runtime/gpu)
            target_link_libraries(test_aluminum_etd_hip PRIVATE
                OpenPFC
                nlohmann_json::nlohmann_json
                $<IF:$<TARGET_EXISTS:Heffte::Heffte>,Heffte::Heffte,heffte>
                Catch2::Catch2
                hip::host
                openpfc_hip_kernels
                MPI::MPI_CXX)
            target_compile_definitions(test_aluminum_etd_hip PRIVATE
                OPENPFC_TEST_ALUMINUM_ETD_HIP)
            set_target_properties(test_aluminum_etd_hip PROPERTIES
                HIP_STANDARD 20
                HIP_STANDARD_REQUIRED ON)
            add_test(NAME HIP_AluminumETD COMMAND test_aluminum_etd_hip)
        endif()
    else()
        message(STATUS "Aluminum: Catch2 not found, not building tests")
    endif()
endif()

install(TARGETS aluminumNew DESTINATION bin)
