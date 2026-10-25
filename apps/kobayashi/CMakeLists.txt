# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(KOBAYASHI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kobayashi_fd_manual src/cpu/kobayashi_fd_manual.cpp)
target_include_directories(kobayashi_fd_manual PRIVATE ${KOBAYASHI_INCLUDE_DIR})
target_link_libraries(kobayashi_fd_manual PRIVATE OpenPFC::openpfc)
install(TARGETS kobayashi_fd_manual DESTINATION bin)

find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
  add_executable(kobayashi_fd_openmp
    src/cpu/kobayashi_fd_openmp.cpp
    src/cpu/kobayashi_fd_openmp_engine.cpp)
  target_include_directories(kobayashi_fd_openmp PRIVATE ${KOBAYASHI_INCLUDE_DIR})
  target_link_libraries(kobayashi_fd_openmp PRIVATE OpenPFC::openpfc

                                                          OpenMP::OpenMP_CXX)
  install(TARGETS kobayashi_fd_openmp DESTINATION bin)

  option(KOBAYASHI_ENABLE_TESTS "Build kobayashi unit tests" ON)
  if(KOBAYASHI_ENABLE_TESTS AND OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2WithMain)
    add_executable(test_kobayashi_fd_openmp tests/test_kobayashi_fd_openmp.cpp
                                            src/cpu/kobayashi_fd_openmp_engine.cpp)
    target_include_directories(test_kobayashi_fd_openmp PRIVATE ${KOBAYASHI_INCLUDE_DIR})
    target_link_libraries(test_kobayashi_fd_openmp PRIVATE OpenPFC::openpfc

                                                                Catch2::Catch2WithMain
                                                                OpenMP::OpenMP_CXX)
    include(CTest)
    add_test(NAME kobayashi-openmp-tests COMMAND test_kobayashi_fd_openmp)
  endif()
else()
  message(STATUS "kobayashi_fd_openmp skipped: OpenMP CXX not found")
endif()

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  add_executable(kobayashi_fd_cuda
    src/cuda/kobayashi_fd_cuda.cpp
    src/cuda/kobayashi_fd_cuda_kernels.cu
  )
  target_include_directories(kobayashi_fd_cuda PRIVATE ${KOBAYASHI_INCLUDE_DIR})
  target_link_libraries(kobayashi_fd_cuda PRIVATE OpenPFC::openpfc CUDA::cudart
                        openpfc_gpu_kernels)
  set_target_properties(kobayashi_fd_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  openpfc_app_link_cxx_runtime(kobayashi_fd_cuda)
  install(TARGETS kobayashi_fd_cuda DESTINATION bin)
  message(STATUS "kobayashi_fd_cuda (MPI + CUDA) enabled")

  if(OpenPFC_BUILD_TESTS)
    include(CTest)
    # Pinned on Tohtori g0005 (H100, gcc 15.2, CUDA 13.1, CUDA-aware Open MPI).
    # Matches CPU OpenMP/MPI 32²/4-step HEX on sum_phi/sumsq_phi/sumsq_T;
    # sum_T is 1 ULP below the CPU pin (see tests/baselines/BASELINES.md).
    set(_kob_cuda_hex
        "KOBAYASHI_VERIFY_HEX sum_phi=0x1\\.b96bf451009d9p\\+3 sumsq_phi=0x1\\.4e770b1504ae4p\\+3 sum_T=0x1\\.6e128af4d5ac5p\\+0 sumsq_T=0x1\\.6546ee0a021fp-2")
    add_test(NAME kobayashi-cuda-hex-smoke
             COMMAND $<TARGET_FILE:kobayashi_fd_cuda> 32 32 4 1e-4 0.03
                     ${CMAKE_CURRENT_BINARY_DIR}/kobayashi_cuda_hex_smoke)
    set_tests_properties(kobayashi-cuda-hex-smoke PROPERTIES
      ENVIRONMENT
        "OPENPFC_KOBAYASHI_SKIP_PNG=1;OPENPFC_KOBAYASHI_QUIET=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
      PASS_REGULAR_EXPRESSION "${_kob_cuda_hex}"
      TIMEOUT 120)
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
      add_test(NAME kobayashi-cuda-hex-2rank
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                       $<TARGET_FILE:kobayashi_fd_cuda> 32 32 4 1e-4 0.03
                       ${CMAKE_CURRENT_BINARY_DIR}/kobayashi_cuda_hex_2rank)
      set_tests_properties(kobayashi-cuda-hex-2rank PROPERTIES
        ENVIRONMENT
          "OPENPFC_KOBAYASHI_SKIP_PNG=1;OPENPFC_KOBAYASHI_QUIET=1;OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
        PASS_REGULAR_EXPRESSION "${_kob_cuda_hex}"
        TIMEOUT 180
        RUN_SERIAL TRUE)
    endif()
  endif()
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(kobayashi_fd_hip
    src/hip/kobayashi_fd_hip.cpp
    src/hip/kobayashi_fd_hip_kernels.hip
  )
  target_include_directories(kobayashi_fd_hip PRIVATE ${KOBAYASHI_INCLUDE_DIR})
  target_link_libraries(kobayashi_fd_hip PRIVATE OpenPFC::openpfc hip::host
                        openpfc_hip_kernels)
  set_target_properties(kobayashi_fd_hip PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
  )
  openpfc_app_link_cxx_runtime(kobayashi_fd_hip)
  install(TARGETS kobayashi_fd_hip DESTINATION bin)
  message(STATUS "kobayashi_fd_hip (MPI + HIP) enabled")

  if(OpenPFC_BUILD_TESTS)
    include(CTest)
    # Pinned on LUMI-G MI250X (job 21682844, 1-rank and 2-rank identical).
    # phi matches CPU/CUDA bitwise; T checksums differ by a few ULP.
    set(_kob_hip_hex
        "KOBAYASHI_VERIFY_HEX sum_phi=0x1\\.b96bf451009d9p\\+3 sumsq_phi=0x1\\.4e770b1504ae4p\\+3 sum_T=0x1\\.6e128af4d5ac8p\\+0 sumsq_T=0x1\\.6546ee0a021efp-2")
    add_test(NAME kobayashi-hip-hex-smoke
             COMMAND $<TARGET_FILE:kobayashi_fd_hip> 32 32 4 1e-4 0.03
                     ${CMAKE_CURRENT_BINARY_DIR}/kobayashi_hip_hex_smoke)
    set_tests_properties(kobayashi-hip-hex-smoke PROPERTIES
      PASS_REGULAR_EXPRESSION "${_kob_hip_hex}"
      TIMEOUT 120)
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
      add_test(NAME kobayashi-hip-hex-2rank
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                       $<TARGET_FILE:kobayashi_fd_hip> 32 32 4 1e-4 0.03
                       ${CMAKE_CURRENT_BINARY_DIR}/kobayashi_hip_hex_2rank)
      set_tests_properties(kobayashi-hip-hex-2rank PROPERTIES
        PASS_REGULAR_EXPRESSION "${_kob_hip_hex}"
        TIMEOUT 180
        RUN_SERIAL TRUE)
    endif()
  endif()
endif()
