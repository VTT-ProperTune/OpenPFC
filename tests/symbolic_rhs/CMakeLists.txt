# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
add_executable(test_symbolic_rhs test_rhs.cpp)
target_include_directories(test_symbolic_rhs PRIVATE ${PROJECT_SOURCE_DIR}/examples/coupling/symbolic-rhs)
target_link_libraries(test_symbolic_rhs PRIVATE OpenPFC)
add_test(NAME SymbolicRhsScatter COMMAND test_symbolic_rhs)
find_package(Python3 3.8 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME SymbolicRhsHighPrecision
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_runtime.py
      $<TARGET_FILE:test_symbolic_rhs>)
endif()
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(test_symbolic_rhs_hip test_rhs.hip)
  target_include_directories(test_symbolic_rhs_hip PRIVATE ${PROJECT_SOURCE_DIR}/examples/coupling/symbolic-rhs)
  target_link_libraries(test_symbolic_rhs_hip PRIVATE OpenPFC openpfc_hip_kernels hip::host)
  set_target_properties(test_symbolic_rhs_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  add_test(NAME HIP_SymbolicRhs COMMAND test_symbolic_rhs_hip)
endif()
