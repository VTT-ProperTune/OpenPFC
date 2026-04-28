# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Optional configure-time checks for GPU-aware MPI when using CUDA with
# Open MPI (MPIX_Query_cuda_support). Other stacks are documented in INSTALL.md.

set(OPENPFC_MPI_IS_OPENMPI FALSE)
set(OPENPFC_OMPI_CUDA_MPI_AWARE_OK FALSE)
set(OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED "")

if(NOT TARGET MPI::MPI_CXX)
  return()
endif()

include(CMakePushCheckState)
cmake_push_check_state()
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "
#include <mpi.h>
#ifndef OPEN_MPI
#error not_open_mpi
#endif
int main(void) { return 0; }
"
  OPENPFC_MPI_IS_OPENMPI)

if(OpenPFC_CUDA_AVAILABLE AND OpenPFC_MPI_CUDA_AWARE)
  if(NOT OPENPFC_MPI_IS_OPENMPI)
    set(OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED "non-Open MPI")
    message(
      STATUS
        "OpenPFC: CUDA GPU-aware MPI not probed at configure (non-Open MPI); "
        "use verify_gpu_aware_mpi / site MPI docs at runtime.")
  elseif(CMAKE_CROSSCOMPILING)
    set(OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED "cross-compiling")
    message(STATUS "OpenPFC: CUDA GPU-aware MPI probe skipped (cross-compiling).")
  else()
    check_cxx_source_compiles(
      "
#include <mpi.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const int q = MPIX_Query_cuda_support();
  (void)q;
  MPI_Finalize();
  return 0;
}
"
      OPENPFC_MPIX_QUERY_CUDA_SUPPORT_COMPILES)
    if(NOT OPENPFC_MPIX_QUERY_CUDA_SUPPORT_COMPILES)
      set(OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED "MPIX_Query_cuda_support unavailable")
      message(
        STATUS
          "OpenPFC: CUDA GPU-aware MPI probe skipped (MPIX_Query_cuda_support missing or "
          "not linkable with this Open MPI).")
    else()
      include(CheckCXXSourceRuns)
      check_cxx_source_runs(
        "
#include <mpi.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const int q = MPIX_Query_cuda_support();
  MPI_Finalize();
  return (q == 1) ? 0 : 2;
}
"
        OPENPFC_OMPI_CUDA_MPI_AWARE_OK)
      if(NOT OPENPFC_OMPI_CUDA_MPI_AWARE_OK)
        message(
          WARNING
            "OpenPFC: Open MPI does not report CUDA-aware MPI (MPIX_Query_cuda_support != 1) "
            "while OpenPFC_MPI_CUDA_AWARE is ON. Use a CUDA-aware Open MPI, or set "
            "-DOpenPFC_MPI_CUDA_AWARE=OFF and reconfigure.")
      else()
        message(
          STATUS
            "OpenPFC: Open MPI CUDA-aware MPI probe succeeded (MPIX_Query_cuda_support)")
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

cmake_pop_check_state()
