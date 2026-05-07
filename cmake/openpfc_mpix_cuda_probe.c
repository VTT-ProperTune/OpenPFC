/* SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd */
/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Configure + runtime probe: Open MPI MPIX_Query_cuda_support (CUDA-aware MPI). */
#include <mpi.h>
#include <mpi-ext.h>
int main(int argc, char **argv) {
  int q = 0;
  MPI_Init(&argc, &argv);
  q = MPIX_Query_cuda_support();
  MPI_Finalize();
  return (q == 1) ? 0 : 2;
}
