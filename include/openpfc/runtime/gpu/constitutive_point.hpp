// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file constitutive_point.hpp
 * @brief One-point CUDA/HIP launchers for the shared J2 and FCC arithmetic.
 *
 * Each launcher evaluates a single material point on the device and copies
 * the result back. It is the device oracle for the nonthrowing cores, not a
 * field solver and not a history transaction.
 */

#include <type_traits>

#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>
#include <openpfc/mechanics/constitutive/simo_j2.hpp>

namespace pfc::finite_strain {

static_assert(std::is_trivially_copyable_v<J2Moduli>);
static_assert(std::is_trivially_copyable_v<J2State>);
static_assert(std::is_trivially_copyable_v<J2EvaluationResult>);
static_assert(std::is_trivially_copyable_v<FccSlipParameters>);
static_assert(std::is_trivially_copyable_v<FccSlipState>);
static_assert(std::is_trivially_copyable_v<FccEvaluationResult>);
static_assert(std::is_trivially_copyable_v<Tensor2>);

#if defined(OpenPFC_ENABLE_CUDA)
void evaluate_j2_point_cuda(J2Moduli moduli, const J2State &state,
                            const Tensor2 &deformation, J2EvaluationResult *out);
void evaluate_fcc_point_cuda(const FccSlipParameters &parameters,
                             const Tensor2 &orientation, const FccSlipState &state,
                             const Tensor2 &deformation, FccEvaluationResult *out);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void evaluate_j2_point_hip(J2Moduli moduli, const J2State &state,
                           const Tensor2 &deformation, J2EvaluationResult *out);
void evaluate_fcc_point_hip(const FccSlipParameters &parameters,
                            const Tensor2 &orientation, const FccSlipState &state,
                            const Tensor2 &deformation, FccEvaluationResult *out);
#endif

} // namespace pfc::finite_strain
