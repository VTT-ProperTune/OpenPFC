# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

if(OpenPFC_ENABLE_HEFFTE AND TARGET Catch2::Catch2)
  set(_me_heffte "")
  if(TARGET Heffte::Heffte)
    set(_me_heffte Heffte::Heffte)
  elseif(TARGET Heffte)
    set(_me_heffte Heffte)
  elseif(TARGET heffte)
    set(_me_heffte heffte)
  endif()

  add_executable(test_microelasticity test_microelasticity.cpp)
  target_link_libraries(test_microelasticity PRIVATE OpenPFC Catch2::Catch2
                        ${_me_heffte})
  add_test(NAME solvers-microelasticity COMMAND test_microelasticity)
  set_tests_properties(solvers-microelasticity PROPERTIES TIMEOUT 300)

  add_executable(test_load_case_compliance test_load_case_compliance.cpp)
  target_link_libraries(test_load_case_compliance PRIVATE OpenPFC Catch2::Catch2
                        ${_me_heffte})
  add_test(NAME solvers-load-case-compliance COMMAND test_load_case_compliance)
  set_tests_properties(solvers-load-case-compliance PROPERTIES TIMEOUT 300)

  if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
     (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
      OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
    add_test(NAME solvers-load-case-compliance-mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:test_load_case_compliance> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(solvers-load-case-compliance-mpi PROPERTIES
      TIMEOUT 300 PROCESSORS 2)
    add_test(NAME solvers-microelasticity-mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:test_microelasticity> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(solvers-microelasticity-mpi PROPERTIES
      TIMEOUT 300 PROCESSORS 2)
  endif()
endif()

set(_me_dev_heffte "")
if(TARGET Heffte::Heffte)
  set(_me_dev_heffte Heffte::Heffte)
elseif(TARGET Heffte)
  set(_me_dev_heffte Heffte)
elseif(TARGET heffte)
  set(_me_dev_heffte heffte)
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL
   AND TARGET openpfc_microelasticity_hip)
  add_executable(openpfc_microelasticity_hip_parity microelasticity_device_parity.cpp)
  target_link_libraries(openpfc_microelasticity_hip_parity PRIVATE
                        OpenPFC openpfc_microelasticity_hip hip::host
                        ${_me_dev_heffte})
  set_target_properties(openpfc_microelasticity_hip_parity PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON)
  if(OpenPFC_BUILD_TESTS)
    add_test(NAME solvers-microelasticity-hip-parity
             COMMAND $<TARGET_FILE:openpfc_microelasticity_hip_parity> --n=16)
    set_tests_properties(solvers-microelasticity-hip-parity PROPERTIES
      PASS_REGULAR_EXPRESSION "ELASTIC_HIP_PARITY_PASS"
      TIMEOUT 300)
  endif()
endif()

if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE AND OpenPFC_ENABLE_CUDA_SPECTRAL
   AND TARGET openpfc_microelasticity_cuda)
  add_executable(openpfc_microelasticity_cuda_parity
                 microelasticity_device_parity.cpp)
  target_link_libraries(openpfc_microelasticity_cuda_parity PRIVATE
                        OpenPFC openpfc_microelasticity_cuda CUDA::cudart
                        ${_me_dev_heffte})
  if(OpenPFC_BUILD_TESTS)
    add_test(NAME solvers-microelasticity-cuda-parity
             COMMAND $<TARGET_FILE:openpfc_microelasticity_cuda_parity> --n=16)
    set_tests_properties(solvers-microelasticity-cuda-parity PROPERTIES
      PASS_REGULAR_EXPRESSION "ELASTIC_CUDA_PARITY_PASS"
      TIMEOUT 300)
    if(OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
       (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR
        OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
      add_test(NAME solvers-microelasticity-cuda-parity-mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:openpfc_microelasticity_cuda_parity> --n=16
          ${MPIEXEC_POSTFLAGS})
      set_tests_properties(solvers-microelasticity-cuda-parity-mpi PROPERTIES
        PASS_REGULAR_EXPRESSION "ELASTIC_CUDA_PARITY_PASS"
        TIMEOUT 300 PROCESSORS 2)
    endif()
  endif()
endif()
