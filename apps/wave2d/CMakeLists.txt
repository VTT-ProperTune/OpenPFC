# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(WAVE2D_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_wave2d_libstdcxx_so "")
foreach(_p IN ITEMS
    "$ENV{OPENPFC_LIBSTDCXX_SO}"
    "/share/apps/gcc/11.2.0/lib64/libstdc++.so"
    "/export/apps/gcc/11.2.0/lib64/libstdc++.so")
  if(EXISTS "${_p}")
    set(_wave2d_libstdcxx_so "${_p}")
    break()
  endif()
endforeach()

function(wave2d_link_libstdcxx target_name)
  if(_wave2d_libstdcxx_so)
    target_link_libraries(${target_name} PRIVATE "${_wave2d_libstdcxx_so}")
    get_filename_component(_rpath "${_wave2d_libstdcxx_so}" DIRECTORY)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${_rpath}"
      INSTALL_RPATH "${_rpath}")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(${target_name} PRIVATE -lstdc++)
  endif()
endfunction()

set(_wave2d_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_wave2d_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_wave2d_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_wave2d_heffte_tgt heffte)
endif()

find_package(OpenMP COMPONENTS CXX)

set(_wave2d_binaries wave2d_fd_manual wave2d_fd)
foreach(_target IN LISTS _wave2d_binaries)
  add_executable(${_target} src/cpu/${_target}.cpp)
  target_include_directories(${_target} PRIVATE ${WAVE2D_INCLUDE_DIR})
  target_link_libraries(${_target} PRIVATE OpenPFC ${_wave2d_heffte_tgt})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${_target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  install(TARGETS ${_target} DESTINATION bin)
endforeach()

option(WAVE2D_ENABLE_TESTS "Enable building wave2d tests" ON)

if(WAVE2D_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2)
    add_executable(test_wave2d tests/test_wave2d.cpp)
    target_include_directories(test_wave2d PRIVATE ${WAVE2D_INCLUDE_DIR})
    target_link_libraries(test_wave2d PRIVATE OpenPFC ${_wave2d_heffte_tgt} Catch2::Catch2)
    include(CTest)
    add_test(NAME wave2d-all-tests COMMAND test_wave2d)
  else()
    message(STATUS "wave2d: Catch2 not found, not building tests")
  endif()
endif()

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  add_executable(wave2d_cuda src/cuda/wave2d_cuda.cu src/cuda/wave2d_kernels.cu)
  target_include_directories(wave2d_cuda PRIVATE ${WAVE2D_INCLUDE_DIR})
  target_link_libraries(wave2d_cuda PRIVATE OpenPFC CUDA::cudart ${_wave2d_heffte_tgt})
  wave2d_link_libstdcxx(wave2d_cuda)
  set_target_properties(wave2d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  install(TARGETS wave2d_cuda DESTINATION bin)

  if(WAVE2D_ENABLE_TESTS AND OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
    add_executable(test_wave2d_cpu_vs_cuda tests/test_wave2d_cpu_vs_cuda.cpp
                                            src/cuda/wave2d_kernels.cu)
    target_include_directories(test_wave2d_cpu_vs_cuda PRIVATE ${WAVE2D_INCLUDE_DIR})
    target_link_libraries(test_wave2d_cpu_vs_cuda PRIVATE OpenPFC Catch2::Catch2
                                                            CUDA::cudart ${_wave2d_heffte_tgt})
    wave2d_link_libstdcxx(test_wave2d_cpu_vs_cuda)
    set_target_properties(test_wave2d_cpu_vs_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME wave2d-cpu-vs-cuda COMMAND test_wave2d_cpu_vs_cuda)
  endif()
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(wave2d_hip src/hip/wave2d_hip.cpp src/hip/wave2d_kernels.hip)
  target_include_directories(wave2d_hip PRIVATE ${WAVE2D_INCLUDE_DIR})
  target_link_libraries(wave2d_hip PRIVATE OpenPFC hip::host ${_wave2d_heffte_tgt})
  wave2d_link_libstdcxx(wave2d_hip)
  set_target_properties(wave2d_hip PROPERTIES LINKER_LANGUAGE CXX)
  install(TARGETS wave2d_hip DESTINATION bin)

  if(WAVE2D_ENABLE_TESTS AND OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
    add_executable(test_wave2d_cpu_vs_hip tests/test_wave2d_cpu_vs_hip.cpp
                                            src/hip/wave2d_kernels.hip)
    target_include_directories(test_wave2d_cpu_vs_hip PRIVATE ${WAVE2D_INCLUDE_DIR})
    target_link_libraries(test_wave2d_cpu_vs_hip PRIVATE OpenPFC Catch2::Catch2 hip::host
                                                         ${_wave2d_heffte_tgt})
    wave2d_link_libstdcxx(test_wave2d_cpu_vs_hip)
    set_target_properties(test_wave2d_cpu_vs_hip PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME wave2d-cpu-vs-hip COMMAND test_wave2d_cpu_vs_hip)
  endif()
endif()
