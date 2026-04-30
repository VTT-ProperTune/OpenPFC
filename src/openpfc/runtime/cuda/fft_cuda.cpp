// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/common/heffte_gpu_r2c_layout.hpp>
#include <openpfc/runtime/cuda/fft_cuda.hpp>

#include <heffte.h>
#include <mpi.h>

namespace pfc {
namespace fft {

using Decomposition = pfc::decomposition::Decomposition;
using pfc::fft::FFT_Impl;

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition, int rank_id,
                                   MPI_Comm comm) {
  auto options = heffte::default_options<heffte::backend::cufft>();
  auto boxes =
      pfc::runtime::heffte_gpu::make_default_r2c_boxes(decomposition, rank_id);

  using fft_r2c_cuda_type = heffte::fft3d_r2c<heffte::backend::cufft>;
  fft_r2c_cuda_type fft_cuda(boxes.real_inbox, boxes.complex_outbox,
                             boxes.r2c_direction, comm, options);

  return FFT_CUDA(std::move(fft_cuda));
}

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition,
                                   MPI_Comm comm) {
  pfc::runtime::heffte_gpu::throw_if_mpi_decomposition_mismatch(
      comm, decomposition, "fft::create_cuda(decomposition, rank_id, comm)");
  const int rank_id = pfc::runtime::heffte_gpu::mpi_comm_rank(comm);
  return create_cuda(decomposition, rank_id, comm);
}

} // namespace fft
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
