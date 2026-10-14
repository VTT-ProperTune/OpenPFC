# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

function(check_command)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE status
                  OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Local FFT consumer failed (${status}): ${output}\n${error}")
  endif()
endfunction()
# Each test invocation owns a fresh prefix and consumer build.
# link_coverage is set by the parent when libopenpfc.a was instrumented.
# --coverage belongs on this link only; it is not an exported package flag.
if(NOT DEFINED link_coverage)
  set(link_coverage OFF)
endif()
set(consumer_link_args)
if(link_coverage)
  list(APPEND consumer_link_args "-DCMAKE_EXE_LINKER_FLAGS=--coverage")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef label)
set(root "${build}/local-fft-consumer-${label}")
check_command("${CMAKE_COMMAND}" --install "${build}" --prefix "${root}/install")
file(RENAME "${root}/install" "${root}/relocated")
check_command("${CMAKE_COMMAND}" -S "${source}/tests/packaging/local_fft_consumer"
              -B "${root}/build" -G Ninja
              ${consumer_link_args}
              "-DCMAKE_PREFIX_PATH=${root}/relocated"
              "-DCMAKE_C_COMPILER=${cc}" "-DCMAKE_CXX_COMPILER=${cxx}")
check_command("${CMAKE_COMMAND}" --build "${root}/build" --parallel 2)
check_command("${root}/build/local_fft_consumer")
if(EXISTS "${root}/build/host_stack_consumer")
  check_command("${root}/build/host_stack_consumer")
endif()
