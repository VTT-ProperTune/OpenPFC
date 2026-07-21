// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_gradient_device.hpp
 * @brief GPU-side per-point FD gradient evaluator parameterized on a
 *        model-owned grads aggregate `G` (mirror of
 *        `pfc::field::FdGradient<G>`).
 *
 * @details
 * The CPU evaluator [`pfc::field::FdGradient<G>`](
 * ../../kernel/field/fd_gradient.hpp) is the single-pass workhorse that
 * the heat3d / wave / Kobayashi-style point-wise drivers use to turn a
 * raw field buffer into a model-owned `G` aggregate at every owned cell.
 * `FdGradientDevice<G>` is its **GPU twin**: same per-member concept
 * introspection, same FD stencils (orders 2..14 for D1, orders 2..20
 * for D2), same compile-time pruning of the catalog `{value, x, y, z,
 * xx, yy, zz}`. Only the moving parts differ:
 *
 *   - The evaluator reads from a **PaddedBrick**-layout device buffer,
 *     so the input pointer points at cell `(0, 0, 0)` of the padded box
 *     `(nx + 2hw, ny + 2hw, nz + 2hw)`. Owned-cell coordinates `(ix, iy,
 *     iz) ∈ [0, n)` are translated to the linear index `(ix + hw) +
 *     (iy + hw) * nxp + (iz + hw) * nxp * nyp` inside `operator()`.
 *   - All the FD weights are **pre-scaled** with `1 / (h * denom_d1)` /
 *     `1 / (h^2 * denom_d2)` and stored as a small `double` array on
 *     the host-side wrapper. The `pod()` accessor returns a trivially
 *     copyable struct that the launcher can pass to a `__global__`
 *     kernel by value (the pre-scaling lifts the `int64_t * double`
 *     conversion out of the inner loop).
 *
 * Like the CPU twin, mixed second derivatives `xy / xz / yz` are
 * **rejected at compile time**. Lifting that requires both the
 * corner-filled halo (already shipped on the GPU side via
 * [`FullPaddedDeviceHalo`](full_padded_device_halo.hpp)) and a tensor
 * stencil device routine — a follow-up commit.
 *
 * **Maximum half-widths** baked into the POD: `MAX_HW1 = 7` (D1 orders
 * 2..14) and `MAX_HW2 = 10` (D2 orders 2..20). Higher orders simply
 * trigger the same `std::invalid_argument` the CPU twin throws today;
 * raising the caps requires extending the table sizes here and the
 * stencil tables in `fd_stencils.hpp`.
 *
 * **Usage** (typical for a `.cu` translation unit):
 *
 * @code
 * #include <openpfc/runtime/cuda/fd_gradient_device.hpp>
 * #include <openpfc/runtime/cuda/for_each_interior_device.hpp>
 *
 * pfc::cuda::FdGradientDevice<MyGrads> eval(d_padded_u, nx, ny, nz, dx, dy, dz,
 *                                           hw, order);
 * pfc::sim::cuda::for_each_interior_device(model, eval.pod(), d_du, t,
 *                                          nx, ny, nz, stream);
 * @endcode
 *
 * @see openpfc/kernel/field/fd_gradient.hpp — CPU twin
 * @see openpfc/runtime/cuda/full_padded_device_halo.hpp — corner-filled
 *      halo that unblocks the future `xy / xz / yz` extension
 * @see openpfc/kernel/data/host_device.hpp — `OPENPFC_HD` portable
 *      annotation used by user-defined `model.rhs(t, g)`
 */

#if defined(OpenPFC_ENABLE_CUDA)

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/grad_concepts.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

namespace pfc::cuda {

/**
 * @brief Compile-time caps on the per-derivative half-widths the device
 *        evaluator can carry in its POD payload. D1 caps at order 14
 *        (`half_width = 7`); D2 caps at order 20 (`half_width = 10`).
 */
inline constexpr int kFdDeviceMaxHw1 = 7;
inline constexpr int kFdDeviceMaxHw2 = 10;

/**
 * @brief Trivially-copyable POD that fully describes a per-point FD
 *        evaluator on the device.
 *
 * The host-side `FdGradientDevice<G>` populates this struct from a
 * `LocalField`-style triple `(d_core, padded_extents, spacing)` and a
 * runtime `order`. The kernel takes a copy of this struct by value (it
 * fits comfortably on the stack of a CUDA thread).
 *
 * **Coefficient layout**: `cx1[k]` is the **already-scaled** D1 weight
 * at offset `+k` along x, with `1 / (dx * denom_d1)` baked in; the
 * matching `-k` offset uses `-cx1[k]` (anti-symmetric) and is computed
 * implicitly by the device evaluator. `cx2[0]` is the centre weight,
 * `cx2[k]` for `k >= 1` is the symmetric weight at offset `±k`, both
 * with `1 / (dx^2 * denom_d2)` baked in. Indices `(0)` of `cx1` are
 * unused (D1 always has zero centre weight); kept for layout symmetry
 * with the CPU view.
 */
struct FdGradientDevicePOD {
  const double *d_core{nullptr}; ///< Padded device buffer (cell `(-hw,-hw,-hw)`).
  int nxp{0};                    ///< Padded x extent.
  int nyp{0};                    ///< Padded y extent.
  int nzp{0};                    ///< Padded z extent.
  int hw{0};                     ///< Halo width (= `order / 2`).

  // Strides through the padded buffer.
  std::ptrdiff_t sx{0}; ///< stride along x = 1
  std::ptrdiff_t sy{0}; ///< stride along y = nxp
  std::ptrdiff_t sz{0}; ///< stride along z = nxp * nyp

  // D1 — pre-scaled weights, indices `[1..hw1]` only (index 0 unused).
  int hw1{0};
  double cx1[kFdDeviceMaxHw1 + 1]{};
  double cy1[kFdDeviceMaxHw1 + 1]{};
  double cz1[kFdDeviceMaxHw1 + 1]{};

  // D2 — pre-scaled weights, indices `[0..hw2]`.
  int hw2{0};
  double cx2[kFdDeviceMaxHw2 + 1]{};
  double cy2[kFdDeviceMaxHw2 + 1]{};
  double cz2[kFdDeviceMaxHw2 + 1]{};
};

/**
 * @brief Device-side per-point evaluator: returns `G` populated according
 *        to `has_*<G>`.
 *
 * Owned-cell indices `(ix, iy, iz) ∈ [0, n)`; the function adds `hw` to
 * each axis to index into the padded buffer. `prepare()` is a no-op for
 * FD; included only so the evaluator satisfies the same interface as
 * `pfc::field::FdGradient<G>` and can be dropped into a hypothetical
 * generic device driver loop.
 *
 * @note Mixed second derivatives `g.xy / g.xz / g.yz` are
 *       `static_assert`-rejected here, mirroring the CPU twin.
 */
template <class G>
OPENPFC_INLINE_HD G evaluate_fd_grad(const FdGradientDevicePOD &e, int ix, int iy,
                                     int iz) noexcept {
  static_assert(!pfc::field::has_xy<G> && !pfc::field::has_xz<G> &&
                    !pfc::field::has_yz<G>,
                "FdGradientDevice: mixed second derivatives (xy/xz/yz) are not "
                "yet implemented; the corner-filled halo "
                "(`pfc::cuda::FullPaddedDeviceHalo`) makes them feasible but "
                "the tensor-product device kernel is a follow-up.");

  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix + e.hw) +
                           static_cast<std::ptrdiff_t>(iy + e.hw) * e.sy +
                           static_cast<std::ptrdiff_t>(iz + e.hw) * e.sz;

  G g{};

  if constexpr (pfc::field::has_value<G>) {
    g.value = e.d_core[c];
  }

  if constexpr (pfc::field::has_x<G>) {
    double acc = 0.0;
    for (int k = 1; k <= e.hw1; ++k) {
      const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * e.sx;
      acc += e.cx1[k] * (e.d_core[c + ks] - e.d_core[c - ks]);
    }
    g.x = acc;
  }
  if constexpr (pfc::field::has_y<G>) {
    double acc = 0.0;
    for (int k = 1; k <= e.hw1; ++k) {
      const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * e.sy;
      acc += e.cy1[k] * (e.d_core[c + ks] - e.d_core[c - ks]);
    }
    g.y = acc;
  }
  if constexpr (pfc::field::has_z<G>) {
    double acc = 0.0;
    for (int k = 1; k <= e.hw1; ++k) {
      const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * e.sz;
      acc += e.cz1[k] * (e.d_core[c + ks] - e.d_core[c - ks]);
    }
    g.z = acc;
  }

  if constexpr (pfc::field::has_xx<G>) {
    double acc = e.cx2[0] * e.d_core[c];
    for (int k = 1; k <= e.hw2; ++k) {
      const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * e.sx;
      acc += e.cx2[k] * (e.d_core[c + ks] + e.d_core[c - ks]);
    }
    g.xx = acc;
  }
  if constexpr (pfc::field::has_yy<G>) {
    double acc = e.cy2[0] * e.d_core[c];
    for (int k = 1; k <= e.hw2; ++k) {
      const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * e.sy;
      acc += e.cy2[k] * (e.d_core[c + ks] + e.d_core[c - ks]);
    }
    g.yy = acc;
  }
  if constexpr (pfc::field::has_zz<G>) {
    double acc = e.cz2[0] * e.d_core[c];
    for (int k = 1; k <= e.hw2; ++k) {
      const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * e.sz;
      acc += e.cz2[k] * (e.d_core[c + ks] + e.d_core[c - ks]);
    }
    g.zz = acc;
  }

  return g;
}

/**
 * @brief Maximum number of per-field evaluators in a device composite.
 *
 * Covers in-tree multi-field FD CUDA models (wave2d / kobayashi: 2 fields)
 * with headroom up to 4.
 */
inline constexpr int kMaxCompositeFields = 4;

/**
 * @brief Trivially-copyable pack of up to `kMaxCompositeFields` per-field
 *        `FdGradientDevicePOD` payloads for multi-field device kernels.
 *
 * Layout is an array of the existing single-field POD (not a parallel-array
 * redesign). `n_fields` records how many leading slots are live.
 */
struct CompositeGradientDevicePOD {
  FdGradientDevicePOD fields[kMaxCompositeFields]{};
  int n_fields{0};
};

static_assert(sizeof(CompositeGradientDevicePOD) == 2088,
              "CompositeGradientDevicePOD size mismatch");

namespace detail {

template <class Composite, class... PerFieldGrads, std::size_t... Is>
__device__ inline Composite
evaluate_fd_grad_composite_impl(const CompositeGradientDevicePOD &eval, int ix,
                                int iy, int iz, std::index_sequence<Is...>) {
  return Composite{
      ::pfc::cuda::evaluate_fd_grad<PerFieldGrads>(eval.fields[Is], ix, iy,
                                                   iz)...};
}

} // namespace detail

/**
 * @brief Device-side multi-field evaluator: build a brace-initializable
 *        `Composite` from per-field catalog grads aggregates.
 *
 * @tparam Composite      Model-owned aggregate listing one member per field
 *                        in pack order (same contract as CPU
 *                        `pfc::field::CompositeGradient`).
 * @tparam PerFieldGrads  Per-field grads types passed to `evaluate_fd_grad`
 *                        (must use catalog names `{value,x,y,z,xx,...}`).
 */
template <class Composite, class... PerFieldGrads>
__device__ inline Composite
evaluate_fd_grad_composite(const CompositeGradientDevicePOD &eval, int ix,
                           int iy, int iz) {
  static_assert(sizeof...(PerFieldGrads) >= 1 &&
                    sizeof...(PerFieldGrads) <= kMaxCompositeFields,
                "evaluate_fd_grad_composite: N must be in [1, kMaxCompositeFields]");
  return detail::evaluate_fd_grad_composite_impl<Composite, PerFieldGrads...>(
      eval, ix, iy, iz, std::index_sequence_for<PerFieldGrads...>{});
}

/**
 * @brief Host-side wrapper: build a populated `FdGradientDevicePOD` from a
 *        padded device buffer + grid spacing + runtime FD order.
 *
 * Constructed with the same template parameter `G` as the model's grads
 * aggregate; the constructor consults `pfc::field::has_*<G>` to populate
 * only the per-axis weight rows that the model actually reads. Asking
 * for a derivative whose stencil order is not tabulated (D1 over 14, D2
 * over 20) produces a `std::invalid_argument` — strictly better than the
 * silent-zero behaviour the early CPU evaluator had.
 *
 * The resulting `FdGradientDevicePOD` is **trivially copyable** and
 * passed by value into device kernels via the `for_each_interior_device`
 * launcher. The wrapper itself stores nothing the kernel cares about and
 * is safe to destroy as soon as the kernel has been launched (the POD
 * carries everything that lives across the kernel call).
 */
template <class G> class FdGradientDevice {
public:
  /**
   * @param d_core      Pointer to the **padded** device buffer, cell
   *                    `(-hw,-hw,-hw)` (i.e. element `0` of the padded
   *                    contiguous storage).
   * @param nx,ny,nz    Owned (non-halo) extents of the local subdomain.
   * @param dx,dy,dz    Grid spacing in physical coordinates.
   * @param halo_width  Halo width on every side; must equal `order / 2`.
   * @param order       Even spatial order. D2 orders 2..20, D1 orders
   *                    2..14 are tabulated.
   *
   * @throws std::invalid_argument if any derivative declared by `G`
   *         requires an order that is not tabulated.
   */
  FdGradientDevice(const double *d_core, int nx, int ny, int nz, double dx,
                   double dy, double dz, int halo_width, int order) {
    static_assert(!pfc::field::has_xy<G> && !pfc::field::has_xz<G> &&
                      !pfc::field::has_yz<G>,
                  "FdGradientDevice: mixed second derivatives (xy/xz/yz) are "
                  "not yet implemented (corner-filled halo is shipped; the "
                  "tensor-product device routine is a follow-up).");

    m_pod.d_core = d_core;
    m_pod.hw = halo_width;
    m_pod.nxp = nx + 2 * halo_width;
    m_pod.nyp = ny + 2 * halo_width;
    m_pod.nzp = nz + 2 * halo_width;
    m_pod.sx = static_cast<std::ptrdiff_t>(1);
    m_pod.sy = static_cast<std::ptrdiff_t>(m_pod.nxp);
    m_pod.sz = static_cast<std::ptrdiff_t>(m_pod.nxp) *
               static_cast<std::ptrdiff_t>(m_pod.nyp);

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_dz = 1.0 / dz;

    if constexpr (pfc::field::has_xx<G> || pfc::field::has_yy<G> ||
                  pfc::field::has_zz<G>) {
      pfc::field::fd::EvenCentralD2View st2{};
      if (!pfc::field::fd::lookup_even_central_d2(order, &st2)) {
        throw std::invalid_argument(
            "FdGradientDevice: order " + std::to_string(order) +
            " has no central second-derivative stencil table "
            "(supported even orders: 2..20).");
      }
      if (st2.half_width > kFdDeviceMaxHw2) {
        throw std::invalid_argument(
            "FdGradientDevice: D2 half_width " + std::to_string(st2.half_width) +
            " exceeds compiled-in cap kFdDeviceMaxHw2 = " +
            std::to_string(kFdDeviceMaxHw2) + "; raise the cap and rebuild.");
      }
      m_pod.hw2 = st2.half_width;
      const double inv_den2 = 1.0 / static_cast<double>(st2.denom);
      const double scale_x = inv_dx * inv_dx * inv_den2;
      const double scale_y = inv_dy * inv_dy * inv_den2;
      const double scale_z = inv_dz * inv_dz * inv_den2;
      for (int k = 0; k <= st2.half_width; ++k) {
        const double ck = static_cast<double>(st2.coeffs[k]);
        m_pod.cx2[k] = scale_x * ck;
        m_pod.cy2[k] = scale_y * ck;
        m_pod.cz2[k] = scale_z * ck;
      }
    }

    if constexpr (pfc::field::has_x<G> || pfc::field::has_y<G> ||
                  pfc::field::has_z<G>) {
      pfc::field::fd::EvenCentralD1View st1{};
      if (!pfc::field::fd::lookup_even_central_d1(order, &st1)) {
        throw std::invalid_argument("FdGradientDevice: order " +
                                    std::to_string(order) +
                                    " has no central first-derivative stencil table "
                                    "(supported even orders: 2..14).");
      }
      if (st1.half_width > kFdDeviceMaxHw1) {
        throw std::invalid_argument(
            "FdGradientDevice: D1 half_width " + std::to_string(st1.half_width) +
            " exceeds compiled-in cap kFdDeviceMaxHw1 = " +
            std::to_string(kFdDeviceMaxHw1) + "; raise the cap and rebuild.");
      }
      m_pod.hw1 = st1.half_width;
      const double inv_den1 = 1.0 / static_cast<double>(st1.denom);
      const double scale_x = inv_dx * inv_den1;
      const double scale_y = inv_dy * inv_den1;
      const double scale_z = inv_dz * inv_den1;
      for (int k = 1; k <= st1.half_width; ++k) {
        const double ck = static_cast<double>(st1.coeffs[k]);
        m_pod.cx1[k] = scale_x * ck;
        m_pod.cy1[k] = scale_y * ck;
        m_pod.cz1[k] = scale_z * ck;
      }
    }
  }

  /// Owned-region extents (host-side accessors, mirror the CPU twin's
  /// `imin/imax/...` interface so a future generic driver template can
  /// deduce the iteration range from either evaluator).
  int imin() const noexcept { return 0; }
  int imax() const noexcept { return m_pod.nxp - 2 * m_pod.hw; }
  int jmin() const noexcept { return 0; }
  int jmax() const noexcept { return m_pod.nyp - 2 * m_pod.hw; }
  int kmin() const noexcept { return 0; }
  int kmax() const noexcept { return m_pod.nzp - 2 * m_pod.hw; }

  /// `prepare()` is a no-op for FD; the application owns the halo
  /// exchange. Kept for interface parity with `SpectralGradient<G>`.
  void prepare() noexcept {}

  /// Trivially-copyable payload to pass into the device kernel.
  [[nodiscard]] const FdGradientDevicePOD &pod() const noexcept { return m_pod; }

private:
  FdGradientDevicePOD m_pod;
};

/**
 * @brief Host-side wrapper: pack per-field `FdGradientDevice` PODs into a
 *        `CompositeGradientDevicePOD` for multi-field device launches.
 *
 * Mirrors CPU `pfc::field::CompositeGradient<Composite, PerField...>`:
 * constructor takes one `FdGradientDevice<PerFieldGrads>` per field,
 * `prepare()` is a no-op fan-out, and `pod()` returns the packed payload.
 */
template <class Composite, class... PerFieldGrads> class CompositeGradientDevice {
public:
  static_assert(sizeof...(PerFieldGrads) >= 1 &&
                    sizeof...(PerFieldGrads) <= kMaxCompositeFields,
                "CompositeGradientDevice: N must be in [1, kMaxCompositeFields]");

  explicit CompositeGradientDevice(const FdGradientDevice<PerFieldGrads> &...evals) {
    const FdGradientDevicePOD pods[] = {evals.pod()...};
    m_pod.n_fields = static_cast<int>(sizeof...(PerFieldGrads));
    for (int i = 0; i < m_pod.n_fields; ++i) {
      m_pod.fields[i] = pods[i];
    }
  }

  void prepare() noexcept {}

  [[nodiscard]] const CompositeGradientDevicePOD &pod() const noexcept {
    return m_pod;
  }

  int imin() const noexcept { return 0; }
  int imax() const noexcept {
    return m_pod.fields[0].nxp - 2 * m_pod.fields[0].hw;
  }
  int jmin() const noexcept { return 0; }
  int jmax() const noexcept {
    return m_pod.fields[0].nyp - 2 * m_pod.fields[0].hw;
  }
  int kmin() const noexcept { return 0; }
  int kmax() const noexcept {
    return m_pod.fields[0].nzp - 2 * m_pod.fields[0].hw;
  }

private:
  CompositeGradientDevicePOD m_pod{};
};

/**
 * @brief Free-function factory: deduces `PerFieldGrads...` from arguments.
 *
 * @code
 * auto composite = pfc::cuda::create_composite_device<WaveLocal>(eval_u, eval_v);
 * @endcode
 */
template <class Composite, class... PerFieldGrads>
[[nodiscard]] inline CompositeGradientDevice<Composite, PerFieldGrads...>
create_composite_device(const FdGradientDevice<PerFieldGrads> &...evals) {
  return CompositeGradientDevice<Composite, PerFieldGrads...>(evals...);
}

/**
 * @brief Build a `FdGradientDevice<G>` over a halo-padded brick.
 *
 * The evaluator indexes the **full** `(nx+2hw)×(ny+2hw)×(nz+2hw)` storage with
 * interior bounds `[hw, nx_pad-hw)` per axis, matching ghost cells populated by
 * `pfc::communication::PaddedHaloExchanger<T>` on `u.data()`.
 *
 * @tparam G     Model-owned grads aggregate (see `grad_concepts.hpp`).
 * @param u      Padded brick (must outlive the returned evaluator).
 * @param order  Even spatial order; should satisfy `order / 2 == u.halo_width()`.
 *               Defaults to 2.
 *
 * @return FdGradientDevice<G> ready for use with for_each_interior_device.
 */
template <class G>
[[nodiscard]] inline FdGradientDevice<G> create(const pfc::field::PaddedBrick<double> &u,
                                                int order = 2) {
  const auto sp = u.spacing();
  return FdGradientDevice<G>(u.data(), u.nx(), u.ny(), u.nz(),
                             sp[0], sp[1], sp[2],
                             u.halo_width(), order);
}

} // namespace pfc::cuda

#endif // OpenPFC_ENABLE_CUDA
