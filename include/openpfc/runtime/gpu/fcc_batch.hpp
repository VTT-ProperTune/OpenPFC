// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>
#include <openpfc/runtime/gpu/quaternion_view.hpp>

namespace pfc::finite_strain {
/// Component-major arrays: entry(component, point) = component * count + point.
/// History components are Fp[9], Lp[9], resistance[12], shear[12].
/// The caller owns persistent storage and accepts by copying/swapping trial into
/// committed after checking every status. Evaluation never modifies committed.
struct FccBatchView {
  std::size_t count;
  const double *orientation; // legacy 9 components, row-major passive tensors
  const double *deformation; // 9 components
  const double *committed;   // 42 components
  double *trial;             // 42 components
  double *piola;             // 9 components
  double *tangent;           // 81 components, (ij)*9 + (kl)
  FccStatus *status;
  double *resolved_shear = nullptr; // optional 12 components
  double *shear_rate = nullptr;     // optional 12 components
  // Optional, read-only native KWC-style crystal->sample q=(w,x,y,z).
  // Only the caller's real-space owned brick is addressed. A non-owning
  // lifetime/decomposition check is the responsibility of FccPoints' adapter.
  FccQuaternionFieldView quaternions{};
  bool use_quaternion_fields = false;
};
struct FccBatchFailure {
  FccStatus status = FccStatus::Success;
  std::size_t point = 0;
};
/// Reduces all point statuses on the device; copies one encoded scalar only.
/// workspace points to one persistent device unsigned long long.
#if defined(OpenPFC_ENABLE_CUDA)
void evaluate_fcc_batch_cuda(FccSlipParameters parameters, FccBatchView view);
FccBatchFailure check_fcc_batch_cuda(const FccStatus *status, std::size_t count,
                                     unsigned long long *workspace);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void evaluate_fcc_batch_hip(FccSlipParameters parameters, FccBatchView view);
FccBatchFailure check_fcc_batch_hip(const FccStatus *status, std::size_t count,
                                    unsigned long long *workspace);
#endif
} // namespace pfc::finite_strain
