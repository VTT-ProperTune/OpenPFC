# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Code coverage configuration and targets

option(OpenPFC_ENABLE_CODE_COVERAGE "Enable coverage" ON)

if(OpenPFC_ENABLE_CODE_COVERAGE)
  message(STATUS "📊 Enabling code coverage")
  target_compile_options(openpfc PUBLIC --coverage)
  target_link_options(openpfc PUBLIC --coverage)
  
  # Find coverage tools
  find_program(LCOV_EXECUTABLE lcov)
  find_program(GENHTML_EXECUTABLE genhtml)
  
  # Use GCOV from toolchain file if specified, otherwise find it
  if(NOT GCOV_EXECUTABLE)
    find_program(GCOV_EXECUTABLE gcov)
  endif()
  
  if(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE AND GCOV_EXECUTABLE)
    message(STATUS "✅ Coverage tools found: lcov, genhtml, and gcov")
    message(STATUS "   lcov:    ${LCOV_EXECUTABLE}")
    message(STATUS "   genhtml: ${GENHTML_EXECUTABLE}")
    message(STATUS "   gcov:    ${GCOV_EXECUTABLE}")
    
    # Add custom target for generating coverage report
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E echo "==> Cleaning old coverage data..."
      COMMAND ${CMAKE_COMMAND} -E remove -f coverage.info coverage_filtered.info
      COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Running tests..."
      COMMAND ${CMAKE_BINARY_DIR}/tests/openpfc-tests
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Capturing coverage data..."
      COMMAND ${LCOV_EXECUTABLE}
        --gcov-tool ${GCOV_EXECUTABLE}
        --capture
        --directory ${CMAKE_BINARY_DIR}
        --output-file coverage.info
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Filtering coverage data..."
      COMMAND ${LCOV_EXECUTABLE}
        --remove coverage.info
        '/usr/*'
        '*/build/_deps/*'
        '*/tests/*'
        '*/examples/*'
        '*/apps/*'
        --output-file coverage_filtered.info
      
      COMMAND ${CMAKE_COMMAND} -E echo "==> Generating HTML report..."
      COMMAND ${GENHTML_EXECUTABLE}
        coverage_filtered.info
        --output-directory coverage-html
        --title "OpenPFC Coverage Report"
        --legend
      
      COMMAND ${CMAKE_COMMAND} -E echo ""
      COMMAND ${CMAKE_COMMAND} -E echo "✅ Coverage report generated!"
      COMMAND ${CMAKE_COMMAND} -E echo "   Open: ${CMAKE_BINARY_DIR}/coverage-html/index.html"
      
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPENDS openpfc-tests
      COMMENT "Generating code coverage report"
    )
    
    # Add target to clean coverage data
    add_custom_target(coverage-clean
      COMMAND ${CMAKE_COMMAND} -E remove -f coverage.info coverage_filtered.info
      COMMAND ${CMAKE_COMMAND} -E remove_directory coverage-html
      COMMAND ${LCOV_EXECUTABLE}
        --gcov-tool ${GCOV_EXECUTABLE}
        --zerocounters
        --directory ${CMAKE_BINARY_DIR}
      COMMAND ${CMAKE_COMMAND} -E echo "✅ Coverage data cleaned"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Cleaning coverage data"
    )
    
    message(STATUS "   Run 'ninja coverage' to generate coverage report")
    message(STATUS "   Run 'ninja coverage-clean' to reset coverage data")
  else()
    if(NOT LCOV_EXECUTABLE)
      message(WARNING "⚠️  lcov not found - coverage report generation disabled")
      message(WARNING "   Install: sudo apt install lcov (Ubuntu/Debian)")
      message(WARNING "           brew install lcov (macOS)")
    endif()
    if(NOT GENHTML_EXECUTABLE)
      message(WARNING "⚠️  genhtml not found - coverage report generation disabled")
    endif()
    if(NOT GCOV_EXECUTABLE)
      message(WARNING "⚠️  gcov not found - coverage report generation disabled")
    endif()
  endif()
endif()
