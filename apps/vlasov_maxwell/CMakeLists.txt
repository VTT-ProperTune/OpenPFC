# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(VLASOV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# The x-advection is a 1-D FFT along a rank-local axis and the transverse
# Maxwell pair is advanced exactly in Fourier space, so this application
# needs an FFT even though nothing about it is a global spectral solve.
set(_vlasov_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_vlasov_heffte_tgt Heffte::Heffte)
elseif(TARGET HEFFTE::HEFFTE)
  set(_vlasov_heffte_tgt HEFFTE::HEFFTE)
elseif(TARGET heffte)
  set(_vlasov_heffte_tgt heffte)
endif()

function(vlasov_add_driver name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${VLASOV_INCLUDE_DIR})
  target_link_libraries(${name} PRIVATE OpenPFC openpfc_apps_common
                        ${_vlasov_heffte_tgt})
  install(TARGETS ${name} DESTINATION bin)
endfunction()

file(GLOB _vlasov_drivers ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp)
foreach(_src ${_vlasov_drivers})
  get_filename_component(_name ${_src} NAME_WE)
  vlasov_add_driver(${_name} ${_src})
endforeach()

option(VLASOV_ENABLE_TESTS "Build vlasov_maxwell unit tests" ON)
if(VLASOV_ENABLE_TESTS AND OpenPFC_BUILD_TESTS AND TARGET Catch2::Catch2)
  file(GLOB _vlasov_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
  foreach(_src ${_vlasov_tests})
    get_filename_component(_name ${_src} NAME_WE)
    add_executable(${_name} ${_src})
    target_include_directories(${_name} PRIVATE ${VLASOV_INCLUDE_DIR})
    target_link_libraries(${_name} PRIVATE OpenPFC openpfc_apps_common
                          Catch2::Catch2 ${_vlasov_heffte_tgt})
    include(CTest)
    add_test(NAME ${_name} COMMAND ${_name})
    set_tests_properties(${_name} PROPERTIES
      ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
      TIMEOUT 900)
  endforeach()
endif()
