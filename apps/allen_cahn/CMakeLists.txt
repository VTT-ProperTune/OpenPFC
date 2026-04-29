# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(ALLEN_CAHN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libstdc++ by absolute path when the module-loaded GCC tree is visible at
# configure time (mpicxx + -lstdc++ alone may pick an older devtoolset DSO).
set(_allen_libstdcxx_so "")
foreach(_p IN ITEMS
    "$ENV{OPENPFC_LIBSTDCXX_SO}"
    "/share/apps/gcc/11.2.0/lib64/libstdc++.so"
    "/export/apps/gcc/11.2.0/lib64/libstdc++.so")
  if(EXISTS "${_p}")
    set(_allen_libstdcxx_so "${_p}")
    break()
  endif()
endforeach()

set(_allen_cahn_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_allen_cahn_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_allen_cahn_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_allen_cahn_heffte_tgt heffte)
endif()

function(allen_cahn_link_libstdcxx target_name)
  if(_allen_libstdcxx_so)
    target_link_libraries(${target_name} PRIVATE "${_allen_libstdcxx_so}")
    get_filename_component(_rpath "${_allen_libstdcxx_so}" DIRECTORY)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${_rpath}"
      INSTALL_RPATH "${_rpath}")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(${target_name} PRIVATE -lstdc++)
  endif()
endfunction()

add_executable(allen_cahn src/cpu/allen_cahn.cpp)
target_include_directories(allen_cahn PRIVATE ${ALLEN_CAHN_INCLUDE_DIR})
target_link_libraries(allen_cahn PRIVATE OpenPFC ${_allen_cahn_heffte_tgt})
allen_cahn_link_libstdcxx(allen_cahn)
install(TARGETS allen_cahn DESTINATION bin)

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  add_executable(allen_cahn_cuda
    src/cuda/allen_cahn.cpp
    src/cuda/allen_cahn_kernels.cu
  )
  target_include_directories(allen_cahn_cuda PRIVATE ${ALLEN_CAHN_INCLUDE_DIR})
  target_link_libraries(allen_cahn_cuda PRIVATE OpenPFC CUDA::cudart ${_allen_cahn_heffte_tgt})
  set_target_properties(allen_cahn_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  allen_cahn_link_libstdcxx(allen_cahn_cuda)
  install(TARGETS allen_cahn_cuda DESTINATION bin)
  message(STATUS "Allen–Cahn: CUDA executable enabled")
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(allen_cahn_hip
    src/hip/allen_cahn.cpp
    src/hip/allen_cahn_kernels.hip
  )
  target_include_directories(allen_cahn_hip PRIVATE ${ALLEN_CAHN_INCLUDE_DIR})
  target_link_libraries(allen_cahn_hip PRIVATE OpenPFC hip::host ${_allen_cahn_heffte_tgt})
  set_target_properties(allen_cahn_hip PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
  )
  allen_cahn_link_libstdcxx(allen_cahn_hip)
  install(TARGETS allen_cahn_hip DESTINATION bin)
  message(STATUS "Allen–Cahn: HIP executable enabled")
endif()

option(ALLEN_CAHN_ENABLE_TESTS "Build Allen–Cahn CPU vs GPU comparison tests" ON)

if(ALLEN_CAHN_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 REQUIRED)
  if(TARGET Catch2::Catch2)
    if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
      add_executable(test_allen_cahn_cpu_vs_cuda
        tests/test_allen_cahn_cpu_vs_cuda.cpp
        src/cuda/allen_cahn_kernels.cu
      )
      target_include_directories(test_allen_cahn_cpu_vs_cuda PRIVATE ${ALLEN_CAHN_INCLUDE_DIR})
      target_link_libraries(test_allen_cahn_cpu_vs_cuda PRIVATE
        OpenPFC Catch2::Catch2 CUDA::cudart ${_allen_cahn_heffte_tgt})
      set_target_properties(test_allen_cahn_cpu_vs_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
      )
      allen_cahn_link_libstdcxx(test_allen_cahn_cpu_vs_cuda)
      include(CTest)
      add_test(NAME allen-cahn-cpu-vs-cuda COMMAND test_allen_cahn_cpu_vs_cuda)
    endif()

    if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
      add_executable(test_allen_cahn_cpu_vs_hip
        tests/test_allen_cahn_cpu_vs_hip.cpp
        src/hip/allen_cahn_kernels.hip
      )
      target_include_directories(test_allen_cahn_cpu_vs_hip PRIVATE ${ALLEN_CAHN_INCLUDE_DIR})
      target_link_libraries(test_allen_cahn_cpu_vs_hip PRIVATE
        OpenPFC Catch2::Catch2 hip::host ${_allen_cahn_heffte_tgt})
      set_target_properties(test_allen_cahn_cpu_vs_hip PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
      )
      allen_cahn_link_libstdcxx(test_allen_cahn_cpu_vs_hip)
      include(CTest)
      add_test(NAME allen-cahn-cpu-vs-hip COMMAND test_allen_cahn_cpu_vs_hip)
    endif()
  endif()
endif()
