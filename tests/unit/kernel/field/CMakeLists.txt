# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel field operations unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_composite_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_central_derivatives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_apply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sbp_diffusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reaction_flux.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_brick_laplacian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_gradient_mixed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fd_symbols.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_periodic_spectra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stencil_apply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_padded_brick.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_brick_iteration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_state_access.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_state_concepts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/field_from_subdomain_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dirichlet_ghosts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reflection_ghosts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_face_flux.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_face_current.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fourier_and_noise.cpp
)
if(OpenPFC_ENABLE_HEFFTE)
  target_sources(openpfc-tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test_field_operations.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_field_comprehensive.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_gradient.cpp
  )
endif()

# Exact mixed-derivative decomposition matrix, including periodic corners.
if(OpenPFC_RUN_MPI_SUITES)
  foreach(_mixed_ranks 1 2 4 8)
    if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
       _mixed_ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE)
      continue()
    endif()
    add_test(NAME fd_mixed_${_mixed_ranks}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_mixed_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:openpfc-tests> "[MPI][fd_mixed]"
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(fd_mixed_${_mixed_ranks}ranks PROPERTIES
      TIMEOUT 120 RUN_SERIAL TRUE)
  endforeach()
  foreach(_reflection_ranks 1 2 4)
    if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
       _reflection_ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE)
      continue()
    endif()
    add_test(NAME fd_reflection_${_reflection_ranks}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_reflection_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:openpfc-tests> "[MPI][reflection]"
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(fd_reflection_${_reflection_ranks}ranks PROPERTIES
      TIMEOUT 180 RUN_SERIAL TRUE)
  endforeach()
  foreach(_face_ranks 1 2 4 8)
    if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
       _face_ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE)
      continue()
    endif()
    add_test(NAME fd_face_current_${_face_ranks}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_face_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:openpfc-tests> "[MPI][face_current]"
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(fd_face_current_${_face_ranks}ranks PROPERTIES
      TIMEOUT 120 RUN_SERIAL TRUE)
  endforeach()
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
  add_executable(test_fd_gradient_mixed_hip test_fd_gradient_mixed.hip)
  target_link_libraries(test_fd_gradient_mixed_hip PRIVATE OpenPFC Catch2::Catch2
    openpfc_hip_kernels hip::host)
  set_target_properties(test_fd_gradient_mixed_hip PROPERTIES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  foreach(_mixed_ranks 1 2 4 8)
    if(NOT OpenPFC_RUN_MPI_SUITES AND _mixed_ranks GREATER 1)
      continue()
    endif()
    if(OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER 0 AND
       _mixed_ranks GREATER OpenPFC_MPI_TEST_MAX_WORLD_SIZE)
      continue()
    endif()
    add_test(NAME HIP_FDMixed_${_mixed_ranks}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_mixed_ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_fd_gradient_mixed_hip>
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(HIP_FDMixed_${_mixed_ranks}ranks PROPERTIES
      TIMEOUT 120 RUN_SERIAL TRUE)
  endforeach()
endif()
