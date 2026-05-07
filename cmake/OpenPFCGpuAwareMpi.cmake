# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Optional configure-time checks for GPU-aware MPI when using CUDA with
# Open MPI (MPIX_Query_cuda_support). Other stacks are documented in INSTALL.md.

set(OPENPFC_OMPI_CUDA_MPI_AWARE_OK FALSE)
set(OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED "")
set(OPENPFC_MPIX_QUERY_CUDA_SUPPORT_COMPILES FALSE)

if(NOT TARGET MPI::MPI_C)
  return()
endif()

if(OpenPFC_CUDA_AVAILABLE AND OpenPFC_MPI_CUDA_AWARE)
  if(CMAKE_CROSSCOMPILING)
    set(OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED "cross-compiling")
    message(STATUS "OpenPFC: CUDA GPU-aware MPI probe skipped (cross-compiling).")
  else()
    # `Check(C)SourceCompiles` is fragile with imported MPI targets on some sites;
    # **`try_run`** builds `cmake/openpfc_mpix_cuda_probe.c` as a tiny real binary.
    try_run(
      OPENPFC_OMPI_CUDA_MPI_AWARE_RUN_EXIT
      OPENPFC_MPIX_PROBE_BUILT
      "${CMAKE_BINARY_DIR}/openpfc_mpix_probe_try"
      "${CMAKE_CURRENT_LIST_DIR}/openpfc_mpix_cuda_probe.c"
      LINK_LIBRARIES MPI::MPI_C)
    if(NOT OPENPFC_MPIX_PROBE_BUILT)
      set(OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED "MPIX_Query_cuda_support unavailable")
      message(
        STATUS
          "OpenPFC: CUDA GPU-aware MPI probe skipped (MPIX probe did not compile/link).")
    else()
      set(OPENPFC_MPIX_QUERY_CUDA_SUPPORT_COMPILES TRUE)
      if(OPENPFC_OMPI_CUDA_MPI_AWARE_RUN_EXIT EQUAL 0)
        set(OPENPFC_OMPI_CUDA_MPI_AWARE_OK TRUE)
        message(
          STATUS
            "OpenPFC: Open MPI CUDA-aware MPI probe succeeded (MPIX_Query_cuda_support)")
      else()
        message(
          WARNING
            "OpenPFC: Open MPI does not report CUDA-aware MPI (MPIX_Query_cuda_support != 1) "
            "while OpenPFC_MPI_CUDA_AWARE is ON (probe exit=${OPENPFC_OMPI_CUDA_MPI_AWARE_RUN_EXIT}). "
            "Use a CUDA-aware Open MPI, or set -DOpenPFC_MPI_CUDA_AWARE=OFF and reconfigure.")
      endif()
    endif()
  endif()
endif()

if(OpenPFC_HIP_AVAILABLE AND OpenPFC_MPI_HIP_AWARE)
  message(
    STATUS
      "OpenPFC: HIP GPU-aware MPI is not probed at configure; on Cray MPICH set "
      "MPICH_GPU_SUPPORT_ENABLED=1 and use verify_gpu_aware_mpi (see docs/INSTALL.LUMI.md).")
endif()
