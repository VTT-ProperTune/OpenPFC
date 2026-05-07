# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(KOBAYASHI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libstdc++.so by absolute path when mixing CUDA/OpenMPI wrappers with a
# different GCC than HeFFTe's OpenMP deps — must match *the compiler used for C++*
# (e.g. OpenMPI loads gcc/15 while the list below used to pick gcc/11 first →
# undefined _M_replace_cold / __cxa_call_terminate at link time).
set(_kobayashi_libstdcxx_so "")
if(DEFINED ENV{OPENPFC_LIBSTDCXX_SO} AND EXISTS "$ENV{OPENPFC_LIBSTDCXX_SO}")
  set(_kobayashi_libstdcxx_so "$ENV{OPENPFC_LIBSTDCXX_SO}")
else()
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so
    OUTPUT_VARIABLE _kob_libstdcxx_from_compiler
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_kob_libstdcxx_from_compiler MATCHES "^/")
    if(EXISTS "${_kob_libstdcxx_from_compiler}")
      set(_kobayashi_libstdcxx_so "${_kob_libstdcxx_from_compiler}")
    endif()
  elseif(_kob_libstdcxx_from_compiler)
    get_filename_component(_kob_cxx_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_kob_cxx_root "${_kob_cxx_bindir}" DIRECTORY)
    foreach(_suffix IN ITEMS "lib64/${_kob_libstdcxx_from_compiler}"
                             "lib/${_kob_libstdcxx_from_compiler}")
      set(_kob_try "${_kob_cxx_root}/${_suffix}")
      if(EXISTS "${_kob_try}")
        set(_kobayashi_libstdcxx_so "${_kob_try}")
        break()
      endif()
    endforeach()
  endif()
endif()
if(NOT _kobayashi_libstdcxx_so)
  foreach(_p IN ITEMS
      "/export/apps/gcc/15.2.0/lib64/libstdc++.so"
      "/share/apps/gcc/15.2.0/lib64/libstdc++.so"
      "/share/apps/gcc/11.2.0/lib64/libstdc++.so"
      "/export/apps/gcc/11.2.0/lib64/libstdc++.so")
    if(EXISTS "${_p}")
      set(_kobayashi_libstdcxx_so "${_p}")
      break()
    endif()
  endforeach()
endif()

if(_kobayashi_libstdcxx_so)
  message(STATUS "kobayashi: linking libstdc++.so from ${_kobayashi_libstdcxx_so}")
endif()

set(_kobayashi_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_kobayashi_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_kobayashi_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_kobayashi_heffte_tgt heffte)
endif()

function(kobayashi_link_libstdcxx target_name)
  if(_kobayashi_libstdcxx_so)
    target_link_libraries(${target_name} PRIVATE "${_kobayashi_libstdcxx_so}")
    get_filename_component(_rpath "${_kobayashi_libstdcxx_so}" DIRECTORY)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${_rpath}"
      INSTALL_RPATH "${_rpath}")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(${target_name} PRIVATE -lstdc++)
  endif()
endfunction()

add_executable(kobayashi_fd_manual src/cpu/kobayashi_fd_manual.cpp)
target_include_directories(kobayashi_fd_manual PRIVATE ${KOBAYASHI_INCLUDE_DIR})
target_link_libraries(kobayashi_fd_manual PRIVATE OpenPFC ${_kobayashi_heffte_tgt})
install(TARGETS kobayashi_fd_manual DESTINATION bin)

find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
  add_executable(kobayashi_fd_openmp
    src/cpu/kobayashi_fd_openmp.cpp
    src/cpu/kobayashi_fd_openmp_engine.cpp)
  target_include_directories(kobayashi_fd_openmp PRIVATE ${KOBAYASHI_INCLUDE_DIR})
  target_link_libraries(kobayashi_fd_openmp PRIVATE OpenPFC ${_kobayashi_heffte_tgt}
                                                          OpenMP::OpenMP_CXX)
  install(TARGETS kobayashi_fd_openmp DESTINATION bin)

  option(KOBAYASHI_ENABLE_TESTS "Build kobayashi unit tests" ON)
  if(KOBAYASHI_ENABLE_TESTS AND OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2WithMain)
    add_executable(test_kobayashi_fd_openmp tests/test_kobayashi_fd_openmp.cpp
                                            src/cpu/kobayashi_fd_openmp_engine.cpp)
    target_include_directories(test_kobayashi_fd_openmp PRIVATE ${KOBAYASHI_INCLUDE_DIR})
    target_link_libraries(test_kobayashi_fd_openmp PRIVATE OpenPFC ${_kobayashi_heffte_tgt}
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
    ${CMAKE_SOURCE_DIR}/src/openpfc/runtime/cuda/padded_halo_faces.cu
  )
  target_include_directories(kobayashi_fd_cuda PRIVATE ${KOBAYASHI_INCLUDE_DIR})
  target_link_libraries(kobayashi_fd_cuda PRIVATE OpenPFC CUDA::cudart ${_kobayashi_heffte_tgt})
  set_target_properties(kobayashi_fd_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  kobayashi_link_libstdcxx(kobayashi_fd_cuda)
  install(TARGETS kobayashi_fd_cuda DESTINATION bin)
  message(STATUS "kobayashi_fd_cuda (MPI + CUDA) enabled")
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(kobayashi_fd_hip
    src/hip/kobayashi_fd_hip.cpp
    src/hip/kobayashi_fd_hip_kernels.hip
  )
  target_include_directories(kobayashi_fd_hip PRIVATE ${KOBAYASHI_INCLUDE_DIR})
  target_link_libraries(kobayashi_fd_hip PRIVATE OpenPFC hip::host ${_kobayashi_heffte_tgt})
  set_target_properties(kobayashi_fd_hip PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
  )
  kobayashi_link_libstdcxx(kobayashi_fd_hip)
  install(TARGETS kobayashi_fd_hip DESTINATION bin)
  message(STATUS "kobayashi_fd_hip (MPI + HIP) enabled")
endif()
