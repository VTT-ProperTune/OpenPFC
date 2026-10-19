// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <array>
#include <memory>
#include <type_traits>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/runtime/gpu/fcc_batch.hpp>
#include <openpfc/runtime/gpu/vector_operations.hpp>
namespace pfc::finite_strain::gpu {
/// Persistent point data. Newton iterate acceptance is separate from committing
/// a physical load step. Rejection never writes the committed buffer.
/// Orientation may be an owned legacy 9-component matrix or a borrowed
/// 4-component device quaternion field (without allocating the matrix).
template <class Tag> class FccPoints {
  using Buffer = pfc::core::DataBuffer<Tag, double>;
  solvers::gpu::VectorOperations<Tag> ops_;
  bool valid_ = false;
  bool orientation_ready_ = false;
  bool use_quaternions_ = false;
  FccQuaternionFieldView quaternion_view_{};
  std::shared_ptr<const void> quaternion_owner_;

public:
  FccSlipParameters parameters;
  Buffer orientation, committed, trial, piola, tangent, tau, rate;
  pfc::core::DataBuffer<Tag, FccStatus> status;
  pfc::core::DataBuffer<Tag, unsigned long long> workspace;
  std::size_t count, integrations = 0, scalar_transfer_bytes = 0;
  FccBatchFailure failure;
  explicit FccPoints(std::size_t n, FccSlipParameters p)
      : parameters(p), committed(42 * n), trial(42 * n),
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

  /// Existing matrix API. The matrix buffer is allocated only for this path.
  void set_orientations(const std::vector<Tensor2> &q) {
    if (q.size() != count) throw std::invalid_argument("orientation count mismatch");
    std::vector<double> packed(9 * count);
    for (std::size_t p = 0; p < count; ++p) {
      if (!crystal::proper_rotation(q[p]))
        throw std::invalid_argument("invalid orientation");
      for (int c = 0; c < 9; ++c) packed[c * count + p] = q[p](c / 3, c % 3);
    }
    if (orientation.size() != 9 * count) orientation.resize(9 * count);
    orientation.copy_from_host(packed);
    quaternion_owner_.reset();
    quaternion_view_ = {};
    use_quaternions_ = false;
    orientation_ready_ = true;
    valid_ = false;
  }

  /// Borrow four resident quaternion Fields from an independently owned model.
  /// All fields must have identical Domain, owned Box3i, storage halo and
  /// memory backend. The owned real-space Box3i must match the FCC FFT inbox
  /// exactly; otherwise the caller must explicitly redistribute beforehand.
  /// The non-null strong owner token keeps the producing fields alive, but
  /// their values must not be mutated during an FCC solve. No host transfer.
  template<class MemorySpace>
  void borrow_quaternion_fields(
      const std::array<const pfc::data::Field<double, MemorySpace>*, 4> &fields,
      const pfc::Domain &fcc_domain, const pfc::Box3i &fcc_inbox,
      std::shared_ptr<const void> producer_owner) {
    using Field = pfc::data::Field<double, MemorySpace>;
    static_assert(std::is_same_v<Tag, typename Field::backend_tag>,
                  "FCC quaternion source and consumer must use the same backend");
    if (!producer_owner)
      throw std::invalid_argument("FCC quaternion source needs a strong lifetime token");
    if (!fcc_inbox.is_consistent() ||
        static_cast<std::size_t>(fcc_inbox.count()) != count)
      throw std::invalid_argument("FCC real-inbox box does not match point count");
    if (!fields[0]) throw std::invalid_argument("null quaternion field");
    const auto &base = *fields[0];
    if (base.domain() != fcc_domain || base.box() != fcc_inbox)
      throw std::invalid_argument("KWC/FCC Domain or real-inbox box mismatch");
    FccQuaternionFieldView view{};
    view.storage_halo = static_cast<std::size_t>(base.storage_halo());
    for (int d = 0; d < 3; ++d) {
      view.owned[d] = static_cast<std::size_t>(fcc_inbox.size[d]);
      view.padded[d] = static_cast<std::size_t>(base.padded_extent(d));
    }
    for (int c = 0; c < 4; ++c) {
      if (!fields[c] || fields[c]->domain() != fcc_domain ||
          fields[c]->box() != fcc_inbox ||
          fields[c]->storage_halo() != base.storage_halo())
        throw std::invalid_argument("KWC quaternion components differ in layout");
      for (int d = 0; d < 3; ++d)
        if (fields[c]->padded_extent(d) != base.padded_extent(d))
          throw std::invalid_argument("KWC quaternion components differ in stride");
      view.q[c] = fields[c]->data();
    }
    if (!view.valid_for(count))
      throw std::invalid_argument("invalid padded quaternion view");
    orientation.resize(0);  // free the unneeded 9-double-per-voxel matrix
    quaternion_view_ = view;
    quaternion_owner_ = std::move(producer_owner);
    use_quaternions_ = true;
    orientation_ready_ = true;
    valid_ = false;
  }

  FccBatchFailure evaluate(const Buffer &f) {
    if (f.size() != 9 * count)
      throw std::invalid_argument("deformation count mismatch");
    if (!orientation_ready_)
      throw std::logic_error("FCC orientations have not been set");
    FccBatchView view{
        count,        use_quaternions_ ? nullptr : orientation.data(),
        f.data(),    committed.data(), trial.data(), piola.data(),
        tangent.data(), status.data(), tau.data(), rate.data(),
        quaternion_view_, use_quaternions_};
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
    return (orientation.size() + committed.size() + trial.size() + piola.size() +
            tangent.size() + tau.size() + rate.size()) * sizeof(double) +
           status.size() * sizeof(FccStatus) +
           workspace.size() * sizeof(unsigned long long);
  }
};
} // namespace pfc::finite_strain::gpu
