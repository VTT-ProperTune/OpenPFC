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
  # The planar case at a small size: a few seconds, but it is a real
  # solidification run, not a unit test of a helper. Serial, because it
  # already reports through MPI_COMM_WORLD.
  add_test(NAME alloy-dendrite-planar COMMAND test_alloy_dendrite)
  set_tests_properties(alloy-dendrite-planar PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 600)

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
