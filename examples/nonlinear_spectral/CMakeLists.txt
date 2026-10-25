# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
if(OpenPFC_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME nonlinear-spectral-course
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify.py
      --self-test $<TARGET_FILE:nonlinear_spectral_course>)
  set_tests_properties(nonlinear-spectral-course PROPERTIES TIMEOUT 60)
  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE)
    add_test(NAME nonlinear-spectral-course-2rank
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify.py
        --self-test $<TARGET_FILE:nonlinear_spectral_course>
        --ranks 2 --mpiexec ${MPIEXEC_EXECUTABLE})
    set_tests_properties(nonlinear-spectral-course-2rank PROPERTIES
      TIMEOUT 60 PROCESSORS 2
      ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
  endif()
endif()
