# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

function(check_command)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE status
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "SDK consumer failed (${status}): ${output}\n${error}")
  endif()
endfunction()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef label)
set(root "${build}/sdk-consumer-${label}")
check_command("${CMAKE_COMMAND}" --install "${build}" --config "${configuration}"
  --prefix "${root}/install")
file(RENAME "${root}/install" "${root}/relocated")
# The independent consumer has no OpenPFC source include or build directory.
file(COPY "${source}/tests/packaging/sdk_consumer" DESTINATION "${root}")
set(common -S "${root}/sdk_consumer" -G Ninja
  "-DCMAKE_PREFIX_PATH=${root}/relocated" "-DCMAKE_C_COMPILER=${cc}"
  "-DCMAKE_CXX_COMPILER=${cxx}" "-DCMAKE_BUILD_TYPE=${configuration}")
if(link_coverage)
  list(APPEND common "-DCMAKE_EXE_LINKER_FLAGS=--coverage")
endif()
check_command("${CMAKE_COMMAND}" ${common} -B "${root}/compatible" -Drequire_compatible=ON)
check_command("${CMAKE_COMMAND}" --build "${root}/compatible" --parallel 2)
check_command("${root}/compatible/sdk_consumer")
file(READ "${root}/relocated/share/openpfc/build-manifest.json" manifest)
foreach(case mpi abi library unknown)
  set(changed "${manifest}")
  set(expected incompatible)
  if(case STREQUAL mpi)
    string(JSON changed SET "${changed}" host_probe mpi_vendor "\"deliberately incompatible MPI vendor\"")
    set(reason mpi_vendor)
  elseif(case STREQUAL abi)
    string(JSON changed SET "${changed}" host_probe stdlib_abi "\"deliberately incompatible ABI\"")
    set(reason stdlib_abi)
  elseif(case STREQUAL library)
    string(JSON changed SET "${changed}" mpi libraries "[]")
    set(expected unverified)
    set(reason "MPI library files")
  else()
    string(JSON changed SET "${changed}" probe_status "\"unknown: cross-compiling\"")
    set(expected unverified)
    set(reason cross-compiling)
  endif()
  file(WRITE "${root}/${case}.json" "${changed}")
  check_command("${CMAKE_COMMAND}" ${common} -B "${root}/${case}"
    "-Dcheck_manifest=${root}/${case}.json" "-Dexpected=${expected}" "-Dreason=${reason}")
endforeach()
# Exercise a real consumer ABI switch, independently of edited evidence.
string(JSON stdlib GET "${manifest}" host_probe stdlib)
string(JSON debug GET "${manifest}" host_probe stdlib_debug)
if(stdlib STREQUAL "libstdc++" AND debug STREQUAL "OFF")
  check_command("${CMAKE_COMMAND}" ${common} -B "${root}/debug-abi"
    "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_DEBUG" -Dexpected=incompatible -Dreason=stdlib_debug)
endif()
# REQUIRED must actually fail with the concrete mismatch and provisioning advice.
execute_process(COMMAND "${CMAKE_COMMAND}" ${common} -B "${root}/required-failure"
  "-Dcheck_manifest=${root}/mpi.json" -Drequire_compatible=ON
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "mpi_vendor" OR
    NOT "${output}${error}" MATCHES "preinstalled OpenPFC SDK")
  message(FATAL_ERROR "REQUIRED did not reject the mismatch actionably: ${output}${error}")
endif()
foreach(backend CUDA HIP)
  string(TOLOWER "${backend}" key)
  string(JSON enabled GET "${manifest}" capabilities "${key}")
  if(NOT enabled)
    check_command("${CMAKE_COMMAND}" ${common} -B "${root}/absent-${backend}"
      "-Dbackend=${backend}" -Dexpected=incompatible "-Dreason=backend is absent")
    # Disabled backends must not publish stale cached flags/architectures.
    string(JSON enabled GET "${manifest}" toolchains "${backend}" enabled)
    string(JSON architectures GET "${manifest}" toolchains "${backend}" architectures)
    if(enabled OR NOT architectures STREQUAL none)
      message(FATAL_ERROR "Disabled ${backend} publishes stale toolchain metadata")
    endif()
  else()
    check_command("${CMAKE_COMMAND}" ${common} -B "${root}/unverified-${backend}"
      "-Dbackend=${backend}" -Dexpected=unverified "-Dreason=device compile/link/run probe")
  endif()
endforeach()
