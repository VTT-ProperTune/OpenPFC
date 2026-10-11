// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/runtime/gpu/fcc_batch.hpp>
#include <openpfc/runtime/gpu/vector_operations.hpp>
namespace pfc::finite_strain::gpu {
/// Persistent point data. Newton iterate acceptance is separate from committing
/// a physical load step. Rejection never writes the committed buffer.
template <class Tag> class FccPoints {
  using Buffer = pfc::core::DataBuffer<Tag, double>;
  solvers::gpu::VectorOperations<Tag> ops_;
  bool valid_ = false;

public:
  FccSlipParameters parameters;
  Buffer orientation, committed, trial, piola, tangent, tau, rate;
  pfc::core::DataBuffer<Tag, FccStatus> status;
  pfc::core::DataBuffer<Tag, unsigned long long> workspace;
  std::size_t count, integrations = 0, scalar_transfer_bytes = 0;
  FccBatchFailure failure;
  explicit FccPoints(std::size_t n, FccSlipParameters p)
      : parameters(p), orientation(9 * n), committed(42 * n), trial(42 * n),
        piola(9 * n), tangent(81 * n), tau(12 * n), rate(12 * n), status(n),
        workspace(1), count(n) {
    crystal::require_parameters(p);
    auto initial = fcc_identity_state(p);
    std::vector<double> history(42 * n);
    for (std::size_t i = 0; i < n; ++i) {
      for (int c = 0; c < 9; ++c) history[c * n + i] = initial.plastic(c / 3, c % 3);
      for (int c = 0; c < 12; ++c) history[(18 + c) * n + i] = initial.resistance[c];
    }
    committed.copy_from_host(history);
  }
  void set_orientations(const std::vector<Tensor2> &q) {
    if (q.size() != count) throw std::invalid_argument("orientation count mismatch");
    std::vector<double> packed(9 * count);
    for (std::size_t p = 0; p < count; ++p) {
      if (!crystal::proper_rotation(q[p]))
        throw std::invalid_argument("invalid orientation");
      for (int c = 0; c < 9; ++c) packed[c * count + p] = q[p](c / 3, c % 3);
    }
    orientation.copy_from_host(packed);
  }
  FccBatchFailure evaluate(const Buffer &f) {
    if (f.size() != 9 * count)
      throw std::invalid_argument("deformation count mismatch");
    FccBatchView view{
        count,        orientation.data(), f.data(),       committed.data(),
        trial.data(), piola.data(),       tangent.data(), status.data(),
        tau.data(),   rate.data()};
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>) {
      evaluate_fcc_batch_cuda(parameters, view);
      failure = check_fcc_batch_cuda(status.data(), count, workspace.data());
    }
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>) {
      evaluate_fcc_batch_hip(parameters, view);
      failure = check_fcc_batch_hip(status.data(), count, workspace.data());
    }
#endif
    integrations += count;
    scalar_transfer_bytes += sizeof(unsigned long long);
    valid_ = failure.status == FccStatus::Success;
    return failure;
  }
  void accept_increment() {
    if (!valid_) throw std::logic_error("no successful FCC trial to commit");
    ops_.copy(trial, committed);
    valid_ = false;
  }
  void reject_increment() { valid_ = false; }
  std::size_t storage_bytes() const {
    return count * ((9 + 42 + 42 + 9 + 81 + 12 + 12) * sizeof(double) +
                    sizeof(FccStatus)) +
           8;
  }
};
} // namespace pfc::finite_strain::gpu
