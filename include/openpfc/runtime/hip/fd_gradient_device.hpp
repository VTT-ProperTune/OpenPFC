// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_gradient_device.hpp
 * @brief GPU-side per-point FD gradient evaluator parameterized on a
 *        model-owned grads aggregate `G` (HIP mirror of
 *        `pfc::cuda::FdGradientDevice<G>`).
 *
 * @details
 * The CPU evaluator [`pfc::field::FdGradient<G>`](
 * ../../kernel/field/fd_gradient.hpp) is the single-pass workhorse that
 * the heat3d / wave / Kobayashi-style point-wise drivers use to turn a
 * raw field buffer into a model-owned `G` aggregate at every owned cell.
 * `FdGradientDevice<G>` is its **GPU twin**: same per-member concept
 * introspection, same FD stencils (orders 2..14 for D1, orders 2..20
 * for D2), same compile-time pruning of the catalog `{value, x, y, z,
 * xx, yy, zz, xy, xz, yz}`. Only the moving parts differ:
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
 * Mixed second derivatives `xy / xz / yz` are supported via a separable
 * D1⊗D1 product on the padded buffer (same pre-scaled `cx1`/`cy1`/`cz1`
 * rows as first derivatives). Callers that request those members must
 * ensure edge and corner ghosts are valid — typically
 * [`FullPaddedDeviceHalo`](full_padded_device_halo.hpp), or an equivalent
 * corner fill (e.g. writing the analytic field into the full padded host
 * buffer in single-rank tests).
 *
 * **Maximum half-widths** baked into the POD: `MAX_HW1 = 7` (D1 orders
 * 2..14) and `MAX_HW2 = 10` (D2 orders 2..20). Higher orders simply
 * trigger the same `std::invalid_argument` the CPU twin throws today;
 * raising the caps requires extending the table sizes here and the
 * stencil tables in `fd_stencils.hpp`.
 *
 * **Usage** (typical for a `.hip` translation unit):
 *
 * @code
 * #include <openpfc/runtime/hip/fd_gradient_device.hpp>
 * #include <openpfc/runtime/hip/for_each_interior_device.hpp>
 *
 * pfc::hip::FdGradientDevice<MyGrads> eval(d_padded_u, nx, ny, nz, dx, dy, dz,
 *                                           hw, order);
 * pfc::sim::hip::for_each_interior_device(model, eval.pod(), d_du, t,
 *                                          nx, ny, nz, stream);
 * @endcode
 *
 * @see openpfc/kernel/field/fd_gradient.hpp — CPU twin
 * @see openpfc/runtime/hip/for_each_interior_device.hpp — HIP driver loop
 * @see openpfc/runtime/hip/full_padded_device_halo.hpp — corner-filled
 *      halo required when `G` declares `xy` / `xz` / `yz`
 * @see openpfc/runtime/cuda/fd_gradient_device.hpp — CUDA twin
 */

#if defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/grad_concepts.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

namespace pfc::hip {

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
 * fits comfortably on the stack of a HIP thread).
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
  std::ptrdiff_t sx{0}; ///< stride along x (always 1).
  std::ptrdiff_t sy{0}; ///< stride along y (= nxp).
  std::ptrdiff_t sz{0}; ///< stride along z (= nxp * nyp).

  // Half-widths for first and second derivatives (populated only if the
  // corresponding derivative is present in `G`).
  int hw1{0}; ///< D1 half-width (also when `has_xy|has_xz|has_yz`).
  int hw2{0}; ///< D2 half-width (populated if `has_xx|has_yy|has_zz`).

  // Pre-scaled FD weights. `cx1[k]` is the weight at offset `+k` along x
  // (anti-symmetric: the `-k` offset uses `-cx1[k]`). `cx2[k]` is the
  // symmetric weight at offset `±k` along x (with `cx2[0]` being the centre
  // weight). All weights are already divided by `(dx * denom_d1)` or
  // `(dx^2 * denom_d2)`, so the device code only does a double-precision
  // dot product.
  double cx1[kFdDeviceMaxHw1 + 1]{}; ///< Pre-scaled D1 weights along x.
  double cy1[kFdDeviceMaxHw1 + 1]{}; ///< Pre-scaled D1 weights along y.
  double cz1[kFdDeviceMaxHw1 + 1]{}; ///< Pre-scaled D1 weights along z.
  double cx2[kFdDeviceMaxHw2 + 1]{}; ///< Pre-scaled D2 weights along x.
  double cy2[kFdDeviceMaxHw2 + 1]{}; ///< Pre-scaled D2 weights along y.
  double cz2[kFdDeviceMaxHw2 + 1]{}; ///< Pre-scaled D2 weights along z.
};

/**
 * @brief Device-side evaluator: build a model-owned `G` aggregate from a
 *        padded buffer at owned cell `(ix, iy, iz)`.
 *
 * This function is the HIP twin of `pfc::field::FdGradient<G>::operator()`.
 * It must be callable from device code (annotated `OPENPFC_HD`) and
 * materialize only the derivative fields that `G` actually declares.
 *
 * @tparam G  Model-owned grads aggregate (e.g. `HeatGrads`, `HasXx`).
 * @param eval POD describing the evaluator (passed by value to the kernel).
 * @param ix   Owned x-coordinate in `[0, nx)`.
 * @param iy   Owned y-coordinate in `[0, ny)`.
 * @param iz   Owned z-coordinate in `[0, nz)`.
 *
 * @return A `G` instance with each member populated from the padded buffer.
 *
 * @note Mixed seconds `g.xy` / `g.xz` / `g.yz` use a separable D1⊗D1
 *       product. The padded buffer must have valid edge/corner ghosts
 *       (`pfc::hip::FullPaddedDeviceHalo` or equivalent corner fill).
 */
template <class G>
OPENPFC_HD inline G evaluate_fd_grad(const FdGradientDevicePOD &eval,
                                     int ix, int iy, int iz) {
  G g{};

  // Translate owned coordinates to padded buffer indices.
  const std::ptrdiff_t c0 = static_cast<std::ptrdiff_t>(ix + eval.hw) +
                            static_cast<std::ptrdiff_t>(iy + eval.hw) *
                                eval.sy +
                            static_cast<std::ptrdiff_t>(iz + eval.hw) *
                                eval.sz;

  const double *u = eval.d_core;

  if constexpr (pfc::field::has_value<G>) {
    g.value = u[c0];
  }

  if constexpr (pfc::field::has_x<G>) {
    double acc = 0.0;
    for (int k = 1; k <= eval.hw1; ++k) {
      const std::ptrdiff_t cp = c0 + static_cast<std::ptrdiff_t>(k) * eval.sx;
      const std::ptrdiff_t cm = c0 - static_cast<std::ptrdiff_t>(k) * eval.sx;
      acc += eval.cx1[k] * (u[cp] - u[cm]);
    }
    g.x = acc;
  }

  if constexpr (pfc::field::has_y<G>) {
    double acc = 0.0;
    for (int k = 1; k <= eval.hw1; ++k) {
      const std::ptrdiff_t cp = c0 + static_cast<std::ptrdiff_t>(k) * eval.sy;
      const std::ptrdiff_t cm = c0 - static_cast<std::ptrdiff_t>(k) * eval.sy;
      acc += eval.cy1[k] * (u[cp] - u[cm]);
    }
    g.y = acc;
  }

  if constexpr (pfc::field::has_z<G>) {
    double acc = 0.0;
    for (int k = 1; k <= eval.hw1; ++k) {
      const std::ptrdiff_t cp = c0 + static_cast<std::ptrdiff_t>(k) * eval.sz;
      const std::ptrdiff_t cm = c0 - static_cast<std::ptrdiff_t>(k) * eval.sz;
      acc += eval.cz1[k] * (u[cp] - u[cm]);
    }
    g.z = acc;
  }

  if constexpr (pfc::field::has_xx<G>) {
    double acc = eval.cx2[0] * u[c0];
    for (int k = 1; k <= eval.hw2; ++k) {
      const std::ptrdiff_t cp = c0 + static_cast<std::ptrdiff_t>(k) * eval.sx;
      const std::ptrdiff_t cm = c0 - static_cast<std::ptrdiff_t>(k) * eval.sx;
      acc += eval.cx2[k] * (u[cp] + u[cm]);
    }
    g.xx = acc;
  }

  if constexpr (pfc::field::has_yy<G>) {
    double acc = eval.cy2[0] * u[c0];
    for (int k = 1; k <= eval.hw2; ++k) {
      const std::ptrdiff_t cp = c0 + static_cast<std::ptrdiff_t>(k) * eval.sy;
      const std::ptrdiff_t cm = c0 - static_cast<std::ptrdiff_t>(k) * eval.sy;
      acc += eval.cy2[k] * (u[cp] + u[cm]);
    }
    g.yy = acc;
  }

  if constexpr (pfc::field::has_zz<G>) {
    double acc = eval.cz2[0] * u[c0];
    for (int k = 1; k <= eval.hw2; ++k) {
      const std::ptrdiff_t cp = c0 + static_cast<std::ptrdiff_t>(k) * eval.sz;
      const std::ptrdiff_t cm = c0 - static_cast<std::ptrdiff_t>(k) * eval.sz;
      acc += eval.cz2[k] * (u[cp] + u[cm]);
    }
    g.zz = acc;
  }

  // Mixed seconds: separable D1⊗D1 on the padded buffer.
  if constexpr (pfc::field::has_xy<G>) {
    double acc = 0.0;
    for (int i = 1; i <= eval.hw1; ++i) {
      const std::ptrdiff_t is = static_cast<std::ptrdiff_t>(i) * eval.sx;
      for (int j = 1; j <= eval.hw1; ++j) {
        const std::ptrdiff_t js = static_cast<std::ptrdiff_t>(j) * eval.sy;
        acc += eval.cx1[i] * eval.cy1[j] *
               (u[c0 + is + js] - u[c0 - is + js] - u[c0 + is - js] +
                u[c0 - is - js]);
      }
    }
    g.xy = acc;
  }
  if constexpr (pfc::field::has_xz<G>) {
    double acc = 0.0;
    for (int i = 1; i <= eval.hw1; ++i) {
      const std::ptrdiff_t is = static_cast<std::ptrdiff_t>(i) * eval.sx;
      for (int k = 1; k <= eval.hw1; ++k) {
        const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * eval.sz;
        acc += eval.cx1[i] * eval.cz1[k] *
               (u[c0 + is + ks] - u[c0 - is + ks] - u[c0 + is - ks] +
                u[c0 - is - ks]);
      }
    }
    g.xz = acc;
  }
  if constexpr (pfc::field::has_yz<G>) {
    double acc = 0.0;
    for (int j = 1; j <= eval.hw1; ++j) {
      const std::ptrdiff_t js = static_cast<std::ptrdiff_t>(j) * eval.sy;
      for (int k = 1; k <= eval.hw1; ++k) {
        const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * eval.sz;
        acc += eval.cy1[j] * eval.cz1[k] *
               (u[c0 + js + ks] - u[c0 - js + ks] - u[c0 + js - ks] +
                u[c0 - js - ks]);
      }
    }
    g.yz = acc;
  }

  return g;
}

/**
 * @brief Host-side wrapper for a device-side FD gradient evaluator.
 *
 * The constructor populates a `FdGradientDevicePOD` struct from a
 * `LocalField`-style triple `(d_core, padded_extents, spacing)` and a
 * runtime `order`. The POD can then be passed by value to a HIP kernel,
 * where `evaluate_fd_grad<G>` uses it to materialize the grads aggregate
 * at each owned cell.
 *
 * @tparam G  Model-owned grads aggregate (e.g. `HeatGrads`, `HasXx`).
 */
template <class G> class FdGradientDevice {
public:
  /**
   * @brief Construct a device evaluator from a padded device buffer.
   *
   * @param d_core       Pointer to the **start of the padded buffer**
   *                    (cell `(-hw,-hw,-hw)`), not the `(0,0,0)` owned cell.
   * @param nx,ny,nz    Owned (non-halo) extents of the local subdomain.
   * @param dx,dy,dz    Grid spacing in physical coordinates.
   * @param halo_width  Halo width on every side; must be `>=` the stencil
   *                    half-width required by members `G` declares
   *                    (typically `order / 2`). Larger halos remain valid.
   * @param order       Even spatial order. D2 orders 2..20, D1 orders
   *                    2..14 are tabulated.
   *
   * @throws std::invalid_argument if any derivative declared by `G`
   *         requires an order that is not tabulated, if a looked-up
   *         half-width exceeds the compiled-in POD caps, or if
   *         `halo_width` is strictly less than the required half-width.
   */
  FdGradientDevice(const double *d_core, int nx, int ny, int nz, double dx,
                   double dy, double dz, int halo_width, int order) {
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

    // D1 weights also feed mixed seconds (D1⊗D1); load whenever any first
    // or mixed member is requested so mixed-only aggregates still get
    // populated cx1/cy1/cz1 rows.
    if constexpr (pfc::field::has_x<G> || pfc::field::has_y<G> ||
                  pfc::field::has_z<G> || pfc::field::has_xy<G> ||
                  pfc::field::has_xz<G> || pfc::field::has_yz<G>) {
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

    const int required =
        m_pod.hw1 > m_pod.hw2 ? m_pod.hw1 : m_pod.hw2;
    if (halo_width < required) {
      throw std::invalid_argument(
          "FdGradientDevice: halo_width " + std::to_string(halo_width) +
          " < required half_width " + std::to_string(required) +
          " for order " + std::to_string(order));
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
 * @brief Build a `FdGradientDevice<G>` over a halo-padded brick.
 *
 * The evaluator indexes the **full** `(nx+2hw)×(ny+2hw)×(nz+2hw)` storage with
 * interior bounds `[hw, nx_pad-hw)` per axis, matching ghost cells populated by
 * `pfc::communication::PaddedHaloExchanger<T>` on `u.data()`.
 *
 * @tparam G     Model-owned grads aggregate (see `grad_concepts.hpp`).
 * @param u      Padded brick (must outlive the returned evaluator).
 * @param order  Even spatial order; requires `u.halo_width() >= order / 2`.
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

} // namespace pfc::hip

#endif // OpenPFC_ENABLE_HIP
