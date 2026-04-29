// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/common/heffte_gpu_r2c_layout.hpp>
#include <openpfc/runtime/hip/fft_hip.hpp>

#include <heffte.h>
#include <mpi.h>

namespace pfc {
namespace fft {

using Decomposition = pfc::decomposition::Decomposition;
using pfc::fft::FFT_Impl;

FFT_HIP create_hip(const Decomposition &decomposition, int rank_id, MPI_Comm comm) {
  auto options = heffte::default_options<heffte::backend::rocfft>();
  auto boxes =
      pfc::runtime::heffte_gpu::make_default_r2c_boxes(decomposition, rank_id);

  using fft_r2c_hip_type = heffte::fft3d_r2c<heffte::backend::rocfft>;
  fft_r2c_hip_type fft_hip(boxes.real_inbox, boxes.complex_outbox,
                           boxes.r2c_direction, comm, options);

  return FFT_HIP(std::move(fft_hip));
}

FFT_HIP create_hip(const Decomposition &decomposition, MPI_Comm comm) {
  pfc::runtime::heffte_gpu::throw_if_mpi_decomposition_mismatch(
      comm, decomposition, "fft::create_hip(decomposition, rank_id, comm)");
  const int rank_id = pfc::runtime::heffte_gpu::mpi_comm_rank(comm);
  return create_hip(decomposition, rank_id, comm);
}

} // namespace fft
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
