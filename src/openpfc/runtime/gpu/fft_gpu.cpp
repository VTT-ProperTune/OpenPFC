// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/common/heffte_gpu_r2c_layout.hpp>
#include <openpfc/runtime/gpu/fft_gpu.hpp>

#include <heffte.h>
#include <mpi.h>

namespace pfc::fft {

using Decomposition = pfc::decomposition::Decomposition;

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition, int rank_id,
                                   MPI_Comm comm, int r2c_direction,
                                   const heffte::plan_options &options) {
  auto boxes = pfc::runtime::heffte_gpu::make_default_r2c_boxes(
      decomposition, rank_id, r2c_direction);

  using fft_r2c_cuda_type = heffte::fft3d_r2c<heffte::backend::cufft>;
  fft_r2c_cuda_type fft_cuda(boxes.real_inbox, boxes.complex_outbox,
                             boxes.r2c_direction, comm, options);

  return FFT_CUDA(std::move(fft_cuda));
}

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition, int rank_id,
                                   MPI_Comm comm, int r2c_direction) {
  return create_cuda(decomposition, rank_id, comm, r2c_direction,
                     heffte::default_options<heffte::backend::cufft>());
}

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition,
                                   MPI_Comm comm) {
  pfc::runtime::heffte_gpu::throw_if_mpi_decomposition_mismatch(
      comm, decomposition, "fft::create_cuda(decomposition, rank_id, comm)");
  const int rank_id = pfc::runtime::heffte_gpu::mpi_comm_rank(comm);
  return create_cuda(decomposition, rank_id, comm);
}

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)

[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition, int rank_id,
                                 MPI_Comm comm, int r2c_direction,
                                 const heffte::plan_options &options) {
  auto boxes = pfc::runtime::heffte_gpu::make_default_r2c_boxes(
      decomposition, rank_id, r2c_direction);

  using fft_r2c_hip_type = heffte::fft3d_r2c<heffte::backend::rocfft>;
  fft_r2c_hip_type fft_hip(boxes.real_inbox, boxes.complex_outbox,
                           boxes.r2c_direction, comm, options);

  return FFT_HIP(std::move(fft_hip));
}

[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition, int rank_id,
                                 MPI_Comm comm, int r2c_direction) {
  return create_hip(decomposition, rank_id, comm, r2c_direction,
                    heffte::default_options<heffte::backend::rocfft>());
}

[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition, MPI_Comm comm) {
  pfc::runtime::heffte_gpu::throw_if_mpi_decomposition_mismatch(
      comm, decomposition, "fft::create_hip(decomposition, rank_id, comm)");
  const int rank_id = pfc::runtime::heffte_gpu::mpi_comm_rank(comm);
  return create_hip(decomposition, rank_id, comm);
}

#endif // OpenPFC_ENABLE_HIP_SPECTRAL

} // namespace pfc::fft

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL || OpenPFC_ENABLE_HIP_SPECTRAL
