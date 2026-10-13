# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
function(check_command)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE status
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Device FFT consumer failed (${status}): ${output}\n${error}")
  endif()
endfunction()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef label)
set(root "${build}/local-fft-${backend}-consumer-${label}")
check_command("${CMAKE_COMMAND}" --install "${build}" --prefix "${root}/install")
file(RENAME "${root}/install" "${root}/relocated")
check_command("${CMAKE_COMMAND}" -S "${source}/tests/packaging/local_fft_device_consumer"
  -B "${root}/build" -G Ninja "-Dbackend=${backend}"
  "-DCMAKE_PREFIX_PATH=${root}/relocated"
  "-DCMAKE_C_COMPILER=${cc}" "-DCMAKE_CXX_COMPILER=${cxx}")
check_command("${CMAKE_COMMAND}" --build "${root}/build" --parallel 2)
execute_process(COMMAND "${root}/build/local_fft_device_consumer"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(status EQUAL 77)
  message("SKIP: no usable device")
elseif(NOT status EQUAL 0)
  message(FATAL_ERROR "Device FFT consumer failed (${status}): ${output}\n${error}")
endif()
