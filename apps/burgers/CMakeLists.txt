# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Periodic Burgers on one CPU rank. The line DFT is ComplexLineFFT, not HeFFTe.

set(BURGERS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(burgers src/burgers.cpp)
target_include_directories(burgers PRIVATE ${BURGERS_INCLUDE_DIR})
target_link_libraries(burgers PRIVATE OpenPFC::openpfc)
if(OpenPFC_HAS_FFTW3)
  target_compile_definitions(burgers PRIVATE BURGERS_HAS_FFTW3)
endif()
install(TARGETS burgers DESTINATION bin)

option(BURGERS_ENABLE_TESTS "Enable building burgers tests" ON)
if(BURGERS_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  find_package(Catch2 QUIET)
  if(TARGET Catch2::Catch2WithMain)
    # Only Fourier tests need the optional compiled line FFT implementation.
    set(_burgers_tests tests/test_burgers_fd_fv.cpp)
    if(OpenPFC_HAS_FFTW3)
      list(APPEND _burgers_tests tests/test_burgers.cpp)
    endif()
    add_executable(test_burgers ${_burgers_tests})
    if(OpenPFC_HAS_FFTW3)
      target_compile_definitions(test_burgers PRIVATE BURGERS_HAS_FFTW3)
    endif()
    target_include_directories(test_burgers PRIVATE ${BURGERS_INCLUDE_DIR})
    target_link_libraries(test_burgers PRIVATE OpenPFC::openpfc Catch2::Catch2WithMain)
    include(CTest)
    add_test(NAME burgers-all-tests COMMAND test_burgers)
    set_tests_properties(burgers-all-tests PROPERTIES TIMEOUT 180)
    # These CTests execute the actual CLI entrypoint, not just helper code.
    add_test(NAME burgers-cli-fdm2
      COMMAND $<TARGET_FILE:burgers> --method fdm2 --N 64 --nu 0
        --dt 0.001 --time 0.5)
    add_test(NAME burgers-cli-fvm
      COMMAND $<TARGET_FILE:burgers> --method fvm --N 64 --nu 0
        --dt 0.001 --time 0.5)
    set_tests_properties(burgers-cli-fdm2 burgers-cli-fvm PROPERTIES
      PASS_REGULAR_EXPRESSION "status=ok" TIMEOUT 120)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(TARGET Python3::Interpreter)
      # This checks actual u.dat rows/coordinates and failure status, not just
      # the executable's success banner. Spectral is optional without FFTW3.
      add_test(NAME burgers-cli-file-and-failure
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli.py
          $<TARGET_FILE:burgers> $<BOOL:${OpenPFC_HAS_FFTW3}>)
      set_tests_properties(burgers-cli-file-and-failure PROPERTIES
        PASS_REGULAR_EXPRESSION "BURGERS_CLI_OUTPUT_OK" TIMEOUT 120)
    endif()
    if(OpenPFC_HAS_FFTW3)
      add_test(NAME burgers-cli-spectral
        COMMAND $<TARGET_FILE:burgers> --method spectral --N 64 --nu 0
          --dt 0.001 --time 0.5)
      set_tests_properties(burgers-cli-spectral PROPERTIES
        PASS_REGULAR_EXPRESSION "status=ok" TIMEOUT 120)
    endif()
  else()
    message(STATUS "burgers: Catch2 not found, not building tests")
  endif()
endif()
