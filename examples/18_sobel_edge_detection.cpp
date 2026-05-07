// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** \example 18_sobel_edge_detection.cpp
 *
 * **Laboratory example** for `pfc::field::stencil::*` — the **generic**,
 * runtime-coefficient stencil layer.
 *
 * `pfc::field::FdGradient<G>` is the PDE-specialised entry point: it
 * consumes the central FD tables in `fd_stencils.hpp` and returns the
 * standard partial derivatives a model expects (`g.x`, `g.xx`, ...). For
 * apps that want **arbitrary** stencils — Sobel-style edge detection,
 * learned CNN kernels, anisotropic discretisations on a non-uniform
 * grid, separable Gaussian smoothing — OpenPFC ships a parallel
 * **generic stencil layer** in `kernel/field/stencil_apply.hpp` with
 * three primitives that take **any** weights:
 *
 *   - `apply_1d_along<Axis>(coeffs, hw, core, c, sx, sy, sz)` — one-axis
 *     convolution with arbitrary asymmetric weights.
 *   - `apply_separable(cx, Hx, cy, Hy, cz, Hz, ...)` — separable
 *     tensor-product convolution `cx ⊗ cy ⊗ cz`.
 *   - `apply_dense<Nz, Ny, Nx>(weights, ...)` — fully general dense 3D
 *     box stencil with compile-time extents.
 *
 * This single-rank example synthesises a 3D field with a sharp step
 * along x, then applies all three primitives and prints a 1D slice of
 * the responses. The numbers should match: Sobel-x via `apply_dense`
 * (3×3×3 weight tensor), Sobel-x via `apply_separable` (`[-1,0,+1] ⊗
 * [1,2,1] ⊗ [1,2,1]`), and a plain central D1 via `apply_1d_along` —
 * three different abstractions, three call shapes, three uses. The
 * "lab vs fortress" trade-off for stencil work in OpenPFC: the kernel
 * primitive surface is **as small as it is honest**, and any custom
 * evaluator the user wants to plug into `pfc::sim::for_each_interior`
 * is just a small struct around one of these three calls.
 *
 * Run:
 *   `./18_sobel_edge_detection`
 *
 * (No MPI; the example is a stand-alone demonstration of the stencil
 *  layer. The same primitives work unchanged on a per-rank padded brick
 *  after a halo exchange — just substitute the local `nxp / nyp` for
 *  the buffer dimensions and shift `c` by `(hw, hw, hw)`.)
 */

#include <array>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#include <openpfc/kernel/field/stencil_apply.hpp>

namespace stencil = pfc::field::stencil;

namespace {

/// Build an `N x N x N` field with a sharp step in x at `i_step`:
/// `u(x, y, z) = (x >= i_step) ? +1.0 : 0.0`. Slightly smoothed in
/// y / z to make Sobel-y / Sobel-z signals visible later.
std::vector<double> make_step_field(int N, int i_step) {
  std::vector<double> u(static_cast<std::size_t>(N) * N * N, 0.0);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const std::size_t c =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(N) * N;
        u[c] = (ix >= i_step) ? 1.0 : 0.0;
      }
    }
  }
  return u;
}

void print_x_line(const char *tag, const std::vector<double> &v, int N, int iy,
                  int iz) {
  std::cout << std::setw(28) << tag << " :  ";
  for (int ix = 0; ix < N; ++ix) {
    const std::size_t c =
        static_cast<std::size_t>(ix) +
        static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
        static_cast<std::size_t>(iz) * static_cast<std::size_t>(N) * N;
    std::cout << std::setw(6) << std::fixed << std::setprecision(2) << v[c] << ' ';
  }
  std::cout << '\n';
}

} // namespace

int main() {
  constexpr int N = 16;
  constexpr int i_step = N / 2;
  constexpr std::ptrdiff_t SX = 1;
  constexpr std::ptrdiff_t SY = N;
  constexpr std::ptrdiff_t SZ = static_cast<std::ptrdiff_t>(N) * N;

  const auto u = make_step_field(N, i_step);

  // Centre row used for visual inspection.
  const int iy = N / 2;
  const int iz = N / 2;

  // -----------------------------------------------------------------
  // 1. apply_1d_along: plain order-2 central D1 along x.
  //
  // Weights `[-0.5, 0, +0.5]` reproduce `f'(x) ≈ (f(x+h) - f(x-h)) / 2h`
  // with `h = 1`. The response is non-zero only at the two cells
  // straddling the step.
  // -----------------------------------------------------------------
  const std::array<double, 3> d1 = {-0.5, 0.0, +0.5};

  std::vector<double> r_d1(u.size(), 0.0);
  for (int ix = 1; ix < N - 1; ++ix) {
    const std::size_t c =
        static_cast<std::size_t>(ix) +
        static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
        static_cast<std::size_t>(iz) * static_cast<std::size_t>(N) * N;
    r_d1[c] = stencil::apply_1d_along<0>(d1.data(), 1, u.data(),
                                         static_cast<std::ptrdiff_t>(c), SX, SY, SZ);
  }

  // -----------------------------------------------------------------
  // 2. apply_separable: classical 3D Sobel-x = [-1, 0, +1]_x ⊗
  //    [1, 2, 1]_y ⊗ [1, 2, 1]_z.
  //
  // The y / z weights smooth across the edge, so the response is a
  // factor of 16 larger than the plain D1 above.
  // -----------------------------------------------------------------
  const std::array<double, 3> sob_x = {-1.0, 0.0, +1.0};
  const std::array<double, 3> smo = {1.0, 2.0, 1.0};

  std::vector<double> r_sep(u.size(), 0.0);
  for (int ix = 1; ix < N - 1; ++ix) {
    const std::size_t c =
        static_cast<std::size_t>(ix) +
        static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
        static_cast<std::size_t>(iz) * static_cast<std::size_t>(N) * N;
    r_sep[c] = stencil::apply_separable(sob_x.data(), 1, smo.data(), 1, smo.data(),
                                        1, u.data(), static_cast<std::ptrdiff_t>(c),
                                        SX, SY, SZ);
  }

  // -----------------------------------------------------------------
  // 3. apply_dense: same Sobel-x kernel as a fully-general 3x3x3 box.
  //    This is the path you would use for non-separable kernels like
  //    a learned CNN filter or a rotationally-invariant Laplacian.
  //    The result must agree with `apply_separable` on this kernel.
  // -----------------------------------------------------------------
  constexpr double sobel_x_dense[3][3][3] = {
      {{-1, 0, +1}, {-2, 0, +2}, {-1, 0, +1}},
      {{-2, 0, +2}, {-4, 0, +4}, {-2, 0, +2}},
      {{-1, 0, +1}, {-2, 0, +2}, {-1, 0, +1}},
  };

  std::vector<double> r_dense(u.size(), 0.0);
  for (int ix = 1; ix < N - 1; ++ix) {
    const std::size_t c =
        static_cast<std::size_t>(ix) +
        static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
        static_cast<std::size_t>(iz) * static_cast<std::size_t>(N) * N;
    r_dense[c] = stencil::apply_dense(sobel_x_dense, u.data(),
                                      static_cast<std::ptrdiff_t>(c), SX, SY, SZ);
  }

  // -----------------------------------------------------------------
  // Print one x-line (centre y, centre z). The step starts at i_step.
  // -----------------------------------------------------------------
  std::cout << "Sobel edge detection on a 3D step field (N=" << N
            << ", step at ix=" << i_step << "):\n\n";
  print_x_line("u (input field)", u, N, iy, iz);
  print_x_line("apply_1d_along (D1, hw=1)", r_d1, N, iy, iz);
  print_x_line("apply_separable (Sobel-x)", r_sep, N, iy, iz);
  print_x_line("apply_dense (Sobel-x box)", r_dense, N, iy, iz);

  // Cross-check: apply_separable and apply_dense must agree (same kernel).
  bool agree = true;
  for (std::size_t i = 0; i < u.size(); ++i) {
    if (r_sep[i] != r_dense[i]) {
      agree = false;
      break;
    }
  }
  std::cout << "\napply_separable == apply_dense (Sobel-x kernel): "
            << (agree ? "yes" : "MISMATCH (kernel mismatch)") << "\n";

  return agree ? 0 : 1;
}
