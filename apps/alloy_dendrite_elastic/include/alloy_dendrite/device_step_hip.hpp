// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_step_hip.hpp
 * @brief HIP launch surface for the four stages of `step.hpp`.
 *
 * @details
 * ## Why this header is only PODs and four function declarations
 *
 * The CPU stepper (`step.hpp`) is a class that owns sixteen `Field`s, three
 * `HaloExchange` groups and nine `FDGradient` evaluators. None of that can
 * cross into a `__global__` function: `Field` is not trivially copyable, and
 * `FDGradient` holds a `std::function`. So the device side is expressed as
 * four free launchers taking three trivially copyable descriptors --
 * @ref DeviceGeom (where the cells are), @ref DeviceStencil (what the
 * finite-difference weights are) and @ref DeviceParams (what the physics is)
 * -- plus a flat pack of raw pointers, @ref DeviceFields. Everything that
 * decides *what* is computed lives in those four structs, which means the
 * kernels contain the arithmetic and nothing else, and the arithmetic can be
 * compared line by line against `step.hpp`.
 *
 * `apps/kobayashi/include/kobayashi/device_step_hip.hpp` is the model. It
 * gets away with a bare argument list because it has two kernels over two
 * fields at fixed second order; here the order is a run-time choice in
 * `[2, 14]`, so the stencil has to travel as data.
 *
 * ## Why the weights travel unscaled
 *
 * `pfc::gpu::FDGradientDevice` -- the library's device FD evaluator -- is the
 * obvious thing to reach for, and it would work. It is deliberately *not*
 * used here, for one reason: it pre-multiplies each stencil weight by
 * `1 / (h * denom)` before the kernel, whereas the CPU evaluator accumulates
 * the integer weights and scales once at the end
 * (`fd_apply.hpp::apply_d1_along` followed by `m_sx1 *`). Those two produce
 * different last bits. Since the whole point of this file is a CPU/GPU parity
 * measurement, the device stencil carries the *integer* coefficients and the
 * scale separately, so the two paths perform the same operations in the same
 * order and the residual difference measures what it is supposed to measure
 * -- reassociation by the compiler and fused multiply-add -- rather than a
 * gratuitous difference in the evaluator. See
 * `src/hip/alloy_dendrite_hip_parity.cpp` for what is left over in practice.
 *
 * ## Stage boundaries are halo-exchange boundaries
 *
 * The four launchers are separate kernels, not one fused kernel, for exactly
 * the reason `step.hpp` has four stages: an MPI halo exchange sits between A
 * and B and between C and D, and a device kernel cannot contain one. The
 * cost of that is three grid-wide synchronisations per step, which is the
 * price of a collocated high-order flux form and is paid identically on the
 * host.
 *
 * @see step.hpp -- the host twin; every kernel here mirrors one stage of it
 * @see device_stepper_hip.hpp -- the host-side owner of the device fields
 */

#include <cstddef>

namespace alloy_dendrite::hip {

/// Maximum D1 half-width carried in @ref DeviceStencil (order 14).
inline constexpr int kMaxHalfWidth1 = 7;
/// Maximum D2 half-width carried in @ref DeviceStencil (order 20; the app
/// caps the order at 14 because it needs both tables, so only 7 is reachable
/// -- the array is sized to the table's range rather than to the app's).
inline constexpr int kMaxHalfWidth2 = 10;

/// Owned extents and padded strides of one rank's brick.
struct DeviceGeom {
  int nx{0};             ///< Owned cells along x.
  int ny{0};             ///< Owned cells along y.
  int nz{0};             ///< Owned cells along z.
  int hw{0};             ///< Storage halo on every side (`fd_order / 2`).
  long long sy{0};       ///< Linear stride along y in the padded buffer.
  long long sz{0};       ///< Linear stride along z in the padded buffer.
};

/**
 * @brief Central-difference weights, integer coefficients plus a final scale.
 *
 * `c1[k]` is the tabulated D1 weight at offset `+k` (the `-k` weight is
 * `-c1[k]`); `c2[0]` is the D2 centre weight and `c2[k]` the symmetric weight
 * at `±k`. `s1x = 1 / (dx * denom_1)` and `s2x = 1 / (dx^2 * denom_2)` are
 * applied once, after the sum, exactly as `pfc::gradient::FDGradient` does.
 */
struct DeviceStencil {
  int hw1{0};
  int hw2{0};
  double c1[kMaxHalfWidth1 + 1]{};
  double c2[kMaxHalfWidth2 + 1]{};
  double s1x{0.0}, s1y{0.0}, s1z{0.0};
  double s2x{0.0}, s2y{0.0}, s2z{0.0};
};

/**
 * @brief Physical parameters of equations (1)-(4), flattened for the device.
 *
 * `at` is `at_scale * kAntiTrapCoeff * W0` precomputed on the host, because
 * `kAntiTrapCoeff = 1/(2 sqrt(2))` is a run-time `std::sqrt` on the host side
 * and evaluating it again on the device would be one more avoidable source of
 * difference in a parity test.
 *
 * Anisotropy is the normalised Karma-Rappel form of `step.hpp`:
 * `a_s = (1 - 3 eps4) + 4 eps4 sum n_i^4` and flux prefactor `16 eps4 W0^2
 * a_s / G^4`. The device evaluates those expressions from `eps4` the same
 * way the host does, rather than receiving pre-derived amplitude/slope
 * constants -- those existed when the host still had an `AnisotropyForm`
 * switch, and they would now be a second derivation of a formula the host
 * no longer derives that way.
 */
struct DeviceParams {
  double W0{1.0};
  double tau0{1.0};
  double lambda{1.0};
  double k{0.15};
  double D_l{2.0};
  double D_th{0.0};
  double M_c{0.0};
  double eps4{0.0};
  double at{0.0};          ///< `at_scale * 1/(2 sqrt 2) * W0`.
  double lambda_el{0.0};
  double grad_floor2{1.0e-24}; ///< `kGradNormFloor2`.
  int spec_source{0};
  int evolve_theta{1};
  int thermal{0}; ///< `evolve_theta && D_th > 0`; gates the thermal Laplacian.
};

/// Flat pack of the device buffers the four stages read and write. Pointers
/// address element `(-hw, -hw, -hw)` of the padded brick, i.e. `Field::data()`.
struct DeviceFields {
  double *phi{nullptr};
  double *U{nullptr};
  double *theta{nullptr};
  double *psi{nullptr};
  double *dphidt{nullptr};
  double *tau{nullptr};
  double *lap_theta{nullptr};
  double *gx{nullptr};
  double *gy{nullptr};
  double *gz{nullptr};
  double *Fx{nullptr};
  double *Fy{nullptr};
  double *Fz{nullptr};
  double *Jx{nullptr};
  double *Jy{nullptr};
  double *Jz{nullptr};
  /// `dF_el/dphi`, or `nullptr` to leave the elastic term out. Read-only, and
  /// only on owned cells, so the caller need not exchange its halo. The
  /// coupled driver fills this from the host solve; the parity driver leaves
  /// it null.
  const double *dfel{nullptr};
};

/// `psi = P(phi) U` over the owned cells; the device twin of
/// `Stepper::seed_conserved_solute`.
void alloy_seed_psi_hip(const DeviceFields &f, const DeviceGeom &g, double k,
                        int dim);

/// Stage A: `grad phi`, anisotropy, the phase-field flux `F`, `tau(n)` and the
/// thermal Laplacian. Mirrors `Stepper::stage_a_`.
void alloy_stage_a_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, int dim);

/// Stage B: `div F` and the complete phase-field right-hand side, including
/// the elastic hook. Mirrors `Stepper::stage_b_`.
void alloy_stage_b_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, int dim);

/// Stage C: the solute flux `J = D_l q(phi) grad U + j_at`. Mirrors
/// `Stepper::stage_c_`.
void alloy_stage_c_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, int dim);

/// Stage D: `div J`, the simultaneous update of `psi`, `phi` and `theta`, and
/// the recovery `U = psi / P(phi_new)`. Mirrors `Stepper::stage_d_`.
void alloy_stage_d_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, double dt,
                       int dim);

} // namespace alloy_dendrite::hip
