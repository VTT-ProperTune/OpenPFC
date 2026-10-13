# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel FFT unit tests. k-space folding does not need HeFFTe.
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_kspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_kspace_iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_power_spectrum.cpp
)
if(OpenPFC_HAS_FFTW3)
  target_sources(openpfc-tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test_local_fft.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_chebyshev.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_real_transform.cpp)
endif()
if(OpenPFC_ENABLE_HEFFTE)
  if(OpenPFC_HAS_FFTW3)
    target_sources(openpfc-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_mixed_transform.cpp)
    if(OpenPFC_ENABLE_CUDA_SPECTRAL OR OpenPFC_ENABLE_HIP_SPECTRAL)
      target_sources(openpfc-tests PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/test_batched_transform.cpp)
      add_test(NAME batched-boundary-mixed-self COMMAND openpfc-tests
          "[batched_boundary]")
      if(OpenPFC_RUN_MPI_SUITES)
        add_test(NAME batched-boundary-2ranks COMMAND ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:openpfc-tests>
            "[batched_boundary]"
            ${MPIEXEC_POSTFLAGS})
        set_tests_properties(batched-boundary-2ranks PROPERTIES PROCESSORS 2)
      endif()
    endif()
  endif()
  target_sources(openpfc-tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test_fft.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_complex_outbox.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_backend_selection.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_dealias.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_incompressible.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_comprehensive.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_heap_concept.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_buffer_sizes.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_input_preserved.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_fft_interface_overriders.cpp
  )
endif()

if(OpenPFC_HAS_FFTW3)
  add_test(NAME LocalFFT COMMAND openpfc-tests "[local_fft]~[MPI]")
endif()
