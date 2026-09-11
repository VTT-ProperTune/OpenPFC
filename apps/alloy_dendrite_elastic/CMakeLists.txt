# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(ALLOY_DENDRITE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Equations (1)-(4) are pure finite difference and need no FFT. Equations
# (5)-(7) -- the eigenstrain microelasticity of `elasticity.hpp` -- are a
# spectral solve and do need one. Rather than move the whole application
# behind `OpenPFC_ENABLE_HEFFTE` (which would delete the Stage-1 planar
# verification from every non-HeFFTe configuration, including one CI runs),
# the elastic path is compiled in only when HeFFTe is available and the
# drivers report a clear error when it is asked for and absent.
set(_alloy_dendrite_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_alloy_dendrite_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_alloy_dendrite_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_alloy_dendrite_heffte_tgt heffte)
endif()

set(_alloy_dendrite_elastic_def ALLOY_DENDRITE_HAVE_ELASTICITY=0)
if(OpenPFC_ENABLE_HEFFTE)
  set(_alloy_dendrite_elastic_def ALLOY_DENDRITE_HAVE_ELASTICITY=1)
endif()

function(alloy_dendrite_add_driver name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${ALLOY_DENDRITE_INCLUDE_DIR})
  target_compile_definitions(${name} PRIVATE ${_alloy_dendrite_elastic_def})
  target_link_libraries(${name} PRIVATE OpenPFC openpfc_apps_common
                        ${_alloy_dendrite_heffte_tgt})
  install(TARGETS ${name} DESTINATION bin)
endfunction()

alloy_dendrite_add_driver(alloy_dendrite_planar src/cpu/alloy_dendrite_planar.cpp)
alloy_dendrite_add_driver(alloy_dendrite_growth src/cpu/alloy_dendrite_growth.cpp)

# --------------------------------------------------------------------------
# HIP twin of equations (1)-(4), and the coupled-cost driver.
#
# Two targets, because they answer two different questions and only one of
# them needs HeFFTe. `alloy_dendrite_hip_parity` is the port's acceptance
# gate: pure FD, no transforms, runs anywhere a GCD exists.
# `alloy_dendrite_coupled_cost` additionally drives `elasticity.hpp`, which
# is built on `pfc::fft::IHostFFT`, so it is only configured when HeFFTe is
# present -- a cost measurement of a solve that cannot be built is not a
# useful thing to ship.
#
# The HIP twin is the FD step. Coupled elasticity stays on the host; that
# is the whole point of the cost driver. Both targets compile with the same
# `ALLOY_DENDRITE_HAVE_ELASTICITY` define the CPU drivers use.
# --------------------------------------------------------------------------
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(alloy_dendrite_hip_parity
    src/hip/alloy_dendrite_hip_parity.cpp
    src/hip/alloy_dendrite_hip_kernels.hip)
  target_include_directories(alloy_dendrite_hip_parity PRIVATE
                             ${ALLOY_DENDRITE_INCLUDE_DIR})
  target_compile_definitions(alloy_dendrite_hip_parity PRIVATE
                             ${_alloy_dendrite_elastic_def})
  target_link_libraries(alloy_dendrite_hip_parity PRIVATE OpenPFC
                        openpfc_apps_common hip::host openpfc_hip_kernels
                        ${_alloy_dendrite_heffte_tgt})
  set_target_properties(alloy_dendrite_hip_parity PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON)
  install(TARGETS alloy_dendrite_hip_parity DESTINATION bin)
  message(STATUS "alloy_dendrite_hip_parity (MPI + HIP) enabled")

  if(_alloy_dendrite_heffte_tgt)
    add_executable(alloy_dendrite_coupled_cost
      src/hip/alloy_dendrite_coupled_cost.cpp
      src/hip/alloy_dendrite_hip_kernels.hip)
    target_include_directories(alloy_dendrite_coupled_cost PRIVATE
                               ${ALLOY_DENDRITE_INCLUDE_DIR})
    target_compile_definitions(alloy_dendrite_coupled_cost PRIVATE
                               ${_alloy_dendrite_elastic_def})
    target_link_libraries(alloy_dendrite_coupled_cost PRIVATE OpenPFC
                          openpfc_apps_common hip::host openpfc_hip_kernels
                          ${_alloy_dendrite_heffte_tgt})
    set_target_properties(alloy_dendrite_coupled_cost PROPERTIES
      HIP_STANDARD 17
      HIP_STANDARD_REQUIRED ON)
    install(TARGETS alloy_dendrite_coupled_cost DESTINATION bin)
    message(STATUS "alloy_dendrite_coupled_cost (MPI + HIP + HeFFTe) enabled")
  else()
    message(STATUS "alloy_dendrite_coupled_cost skipped: HeFFTe not found")
  endif()
endif()

option(ALLOY_DENDRITE_ENABLE_TESTS "Build alloy_dendrite_elastic tests" ON)

if(ALLOY_DENDRITE_ENABLE_TESTS AND OpenPFC_BUILD_TESTS
   AND TARGET Catch2::Catch2)
  add_executable(test_alloy_dendrite tests/test_alloy_dendrite.cpp)
  target_include_directories(test_alloy_dendrite PRIVATE
                             ${ALLOY_DENDRITE_INCLUDE_DIR})
  target_compile_definitions(test_alloy_dendrite PRIVATE
                             ${_alloy_dendrite_elastic_def})
  target_link_libraries(test_alloy_dendrite PRIVATE OpenPFC openpfc_apps_common
                        Catch2::Catch2 ${_alloy_dendrite_heffte_tgt})
  include(CTest)
  # The Catch binary is the Stage-1 planar run *and* the coupled elastic
  # controls. Debug FFT on a 2-core runner is the slow path; RUN_SERIAL so
  # ctest -j2 does not steal the cores, and 900 s covers gcc-13 Debug.
  add_test(NAME alloy-dendrite-planar COMMAND test_alloy_dendrite)
  set_tests_properties(alloy-dendrite-planar PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 900
    RUN_SERIAL TRUE)

  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    # Same physics on two ranks: the halo exchange of three field groups and
    # the transverse-mean reduction are the parts that a single rank cannot
    # exercise, and they are exactly the parts a conservation claim rests on.
    add_test(NAME alloy-dendrite-planar-2rank
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     $<TARGET_FILE:test_alloy_dendrite>
                     "[planar]")
    set_tests_properties(alloy-dendrite-planar-2rank PROPERTIES
      ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
      TIMEOUT 900
      RUN_SERIAL TRUE)
  endif()
endif()

# --------------------------------------------------------------------------
# CPU/GPU parity as a ctest.
#
# The binary decides pass or fail itself (`--tol`), rather than a pinned
# checksum in this file. Both forms are defensible; this one was chosen
# because the quantity that matters is "how far apart are the two paths",
# and a pinned hex checksum answers a different question -- it pins the GPU
# against its own past, which drifts the moment the ROCm version changes,
# and says nothing about the host. The tolerance below is the measured drift
# of a real run rounded up (see the app README, "CPU against GPU"), not a
# number picked to make the test pass.
# --------------------------------------------------------------------------
if(TARGET alloy_dendrite_hip_parity AND OpenPFC_BUILD_TESTS)
  include(CTest)
  set(_alloy_parity_tol "2e-11")
  set(_alloy_parity_args --nx=32 --ny=32 --nz=32 --dx=0.8 --fd-order=4
                         --steps=50 --tol=${_alloy_parity_tol})
  add_test(NAME alloy-dendrite-hip-parity
           COMMAND $<TARGET_FILE:alloy_dendrite_hip_parity>
                   ${_alloy_parity_args} --run-id=ctest-1rank)
  set_tests_properties(alloy-dendrite-hip-parity PROPERTIES
    PASS_REGULAR_EXPRESSION "ALLOY_PARITY_PASS"
    TIMEOUT 600)

  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    # Two ranks: this is where an unexchanged halo group shows up. The
    # CPU/GPU comparison is per-rank and therefore still exact in meaning;
    # the 1-vs-N-rank comparison needs two runs and lives in
    # slurm/alloy_dendrite_hip_parity.sbatch, which has a GPU node to do it on.
    add_test(NAME alloy-dendrite-hip-parity-2rank
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     $<TARGET_FILE:alloy_dendrite_hip_parity>
                     ${_alloy_parity_args} --run-id=ctest-2rank)
    set_tests_properties(alloy-dendrite-hip-parity-2rank PROPERTIES
      PASS_REGULAR_EXPRESSION "ALLOY_PARITY_PASS"
      TIMEOUT 900
      RUN_SERIAL TRUE)
  endif()
endif()
