// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_multigrid.hpp
 * @brief Matrix-free variable-coefficient geometric multigrid on padded Fields.
 *
 * Solves (approximately) A u = f for the cell-centred operator
 *
 *     A u = c u - div(D grad u)   at free cells,   A u = u   at pinned cells,
 *
 * with a cell coefficient c >= 0, per-face coefficients D > 0 (one per axis,
 * the +face of each cell) and a cell mask m (1 free, 0 pinned). It is meant as
 * the preconditioner M^{-1} of a Krylov method (`pfc::solvers::cg`, `gmres`,
 * `newton`): `apply(r, z)` runs one V-cycle from z = 0.
 *
 * Hierarchy: cell-centred coarsening by 2 on every active axis while each
 * local extent and local offset is even and the coarse extent is at least
 * `Controls::min_extent`, on every rank. Coarse levels keep the fine proc grid
 * (`decomposition::create(domain, grid)`), so the hierarchy is distributed and
 * needs no gathering. Coarse coefficients are restricted from the fine ones
 * (c averaged, face D averaged over the fine faces of a coarse face, a coarse
 * cell is free if any child is free); no mean is projected out, so c > 0 with
 * periodic boundaries is fine. This rediscretisation is grid-independent for
 * smooth coefficients but not robust to large jumps: with a 10^3 jump the PCG
 * iteration count grows slowly with the grid, and with 10^6 it degrades badly
 * (operator-dependent transfers would be needed there). `update()` refreshes every
 * level from the fine coefficients without reallocating, so a Newton loop can call
 * it once per linearisation.
 *
 * Cycle: red-black Gauss-Seidel, `pre` sweeps (red then black) before and
 * `post` sweeps (black then red) after the coarse correction, so the V-cycle is
 * symmetric and can precondition CG; residual restriction is the adjoint of the
 * prolongation divided by the number of children (1/8, 3/8, 3/8, 1/8 per axis);
 * cell-centred bilinear/trilinear prolongation; `coarse_sweeps` smoothing
 * sweeps on the coarsest level. Halos are exchanged by `pfc::comm::HaloExchange`
 * (Full connectivity: the prolongation reads diagonal u halos and the
 * restriction diagonal mask halos).
 *
 * Backends: host (`HostSpace`) and CUDA/HIP (`CUDASpace`/`HIPSpace`; kernels in
 * src/openpfc/runtime/gpu/field_multigrid_gpu.inc). The point arithmetic is
 * shared (field_multigrid_kernels.hpp), so host and device give the same
 * result up to floating-point contraction. Device coefficients written on the
 * host (`with_host_view`) are pushed by `update()`; vectors passed to `apply`
 * must be current on the device. Non-periodic axes keep zero halos
 * (homogeneous Dirichlet on the correction).
 */

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/solvers/field_multigrid_kernels.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/field_multigrid_gpu.hpp>
#endif

namespace pfc::solvers {

template <class MemorySpace> class FieldMultigrid {
public:
  using FieldT = data::Field<double, MemorySpace>;
  using Exchange = comm::HaloExchange<MemorySpace, double>;

  struct Controls {
    int pre = 2;            ///< red-black sweeps before the coarse correction
    int post = 2;           ///< red-black sweeps after it
    int coarse_sweeps = 40; ///< red-black sweeps on the coarsest level
    int min_extent = 2;     ///< smallest coarse local extent per active axis
    int max_levels = 32;
    int tag_base = 7000; ///< first MPI tag; each level uses a block of 300
  };

  FieldMultigrid(const decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
                 Controls controls = {})
      : m_comm(comm), m_rank(rank), m_c(controls) {
    if (m_c.pre < 0 || m_c.post < 0 || m_c.coarse_sweeps < 1 || m_c.min_extent < 2)
      throw std::invalid_argument("FieldMultigrid: bad controls");
    const Domain fine = decomposition::domain(decomp);
    const auto gsize = domain::get_size(fine);
    for (int a = 0; a < 3; ++a) m_active[a] = gsize[a] > 1;
    const Int3 grid = decomposition::get_grid(decomp);
    m_levels.push_back(make_level(fine, decomp, 0));
    while (static_cast<int>(m_levels.size()) < m_c.max_levels) {
      const Level &F = *m_levels.back();
      Int3 csize{}, fs = F.box.size;
      int ok = 1;
      for (int a = 0; a < 3; ++a) {
        if (!m_active[a]) {
          csize[a] = domain::get_size(F.domain)[a];
          continue;
        }
        if (fs[a] % 2 != 0 || F.box.low[a] % 2 != 0 || fs[a] / 2 < m_c.min_extent)
          ok = 0;
        csize[a] = domain::get_size(F.domain)[a] / 2;
      }
      int all = 0;
      MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, m_comm);
      if (!all) break;
      Real3 sp = domain::get_spacing(F.domain);
      for (int a = 0; a < 3; ++a)
        if (m_active[a]) sp[a] *= 2;
      const Domain cd =
          domain::with_spacing(csize, sp, domain::get_periodic(F.domain));
      auto cdec = decomposition::create(cd, grid);
      const Box3i cb = decomposition::local_box(cdec, m_rank);
      for (int a = 0; a < 3; ++a) {
        const int r = m_active[a] ? 2 : 1;
        if (cb.low[a] * r != F.box.low[a] || cb.size[a] * r != fs[a]) ok = 0;
      }
      MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, m_comm);
      if (!all) break;
      m_levels.push_back(make_level(cd, cdec, static_cast<int>(m_levels.size())));
    }
  }

  [[nodiscard]] int levels() const noexcept {
    return static_cast<int>(m_levels.size());
  }
  [[nodiscard]] Int3 level_size(int l) const { return m_levels.at(l)->box.size; }

  /// Fine-level coefficient fields; write owned cells, then call update().
  FieldT &c() noexcept { return m_levels[0]->c; }
  FieldT &dx() noexcept { return m_levels[0]->dx; }
  FieldT &dy() noexcept { return m_levels[0]->dy; }
  FieldT &dz() noexcept { return m_levels[0]->dz; }
  FieldT &mask() noexcept { return m_levels[0]->m; }

  /// Exchange the fine face coefficients and restrict all coefficients to every
  /// coarse level. Call after changing c, dx, dy, dz or mask.
  void update() {
    if constexpr (!is_host) {
      Level &L = *m_levels[0];
      for (FieldT *f : {&L.c, &L.dx, &L.dy, &L.dz, &L.m}) f->sync_to_device();
    }
    for (std::size_t l = 0; l < m_levels.size(); ++l) {
      Level &L = *m_levels[l];
      if (l > 0) {
        Level &F = *m_levels[l - 1];
        restrict_coefficients(F, L);
      }
      mark_written(L.dx, L.dy, L.dz);
      mark_written(L.m, L.m, L.m);
      L.coef_x->exchange();
    }
  }

  /// out = A u on the fine level (owned cells). u's halos are exchanged here.
  void apply_operator(FieldT &u, FieldT &out) {
    Level &L = *m_levels[0];
    copy_owned(L, u, L.u);
    exchange_u(L);
    if constexpr (is_host) {
      for_owned(L, [&](int i, int j, int k) {
        out.data()[fmg::index(L.g, i, j, k)] =
            fmg::apply(L.g, coeffs(L), L.u.data(), i, j, k);
      });
    } else {
      gpu_ops::apply(L.g, coeffs(L), L.u.data(), out.data());
      out.note_device_write();
    }
  }

  /// z = M^{-1} r: one V-cycle from zero (owned cells of r read, of z written).
  void apply(const FieldT &r, FieldT &z) {
    Level &L = *m_levels[0];
    copy_owned(L, r, L.f);
    zero(L.u);
    cycle(0);
    copy_owned(L, L.u, z);
  }

  /// Convenience: `precondition(r, z, k)` signature of the Krylov solvers.
  void operator()(const FieldT &r, FieldT &z, int = 0) { apply(r, z); }

private:
  static constexpr bool is_host = std::is_same_v<MemorySpace, HostSpace>;

  struct Level {
    Domain domain;
    decomposition::Decomposition decomp;
    Box3i box;
    fmg::Level g;
    FieldT u, f, r, c, dx, dy, dz, m;
    std::unique_ptr<Exchange> u_x, r_x, coef_x;
    Level(const Domain &d, decomposition::Decomposition dc, const Box3i &b)
        : domain(d), decomp(std::move(dc)), box(b), u(d, b, 1), f(d, b, 1),
          r(d, b, 1), c(d, b, 1), dx(d, b, 1), dy(d, b, 1), dz(d, b, 1), m(d, b, 1) {
    }
  };

  struct gpu_ops;

  std::unique_ptr<Level> make_level(const Domain &d,
                                    const decomposition::Decomposition &dc, int l) {
    const Box3i b = decomposition::local_box(dc, m_rank);
    auto L = std::make_unique<Level>(d, dc, b);
    const auto sp = domain::get_spacing(d);
    for (int a = 0; a < 3; ++a) {
      L->g.n[a] = b.size[a];
      L->g.pitch[a] = b.size[a] + 2;
      L->g.active[a] = m_active[a] ? 1 : 0;
      L->g.low[a] = b.low[a];
      L->g.ih2[a] = m_active[a] ? 1.0 / (sp[a] * sp[a]) : 0.0;
    }
    const bool two_d = !m_active[2];
    comm::HaloExchangeOptions ou, oc;
    ou.connectivity = comm::HaloConnectivity::Full;
    ou.directions = two_d ? halo::presets::Full2D() : halo::presets::Full3D();
    ou.exchange_base = m_c.tag_base + 300 * l;
    oc.connectivity = comm::HaloConnectivity::Full;
    oc.directions = two_d ? halo::presets::Full2D() : halo::presets::Full3D();
    oc.exchange_base = m_c.tag_base + 300 * l + 80;
    for (FieldT *f : {&L->u, &L->f, &L->r, &L->c, &L->dx, &L->dy, &L->dz, &L->m})
      zero(*f);
    fill_owned(*L, L->m, 1.0);
    L->u_x = std::make_unique<Exchange>(std::vector<FieldT *>{&L->u}, L->decomp,
                                        m_rank, m_comm, ou);
    comm::HaloExchangeOptions orr = ou;
    orr.exchange_base = m_c.tag_base + 300 * l + 40;
    L->r_x = std::make_unique<Exchange>(std::vector<FieldT *>{&L->r}, L->decomp,
                                        m_rank, m_comm, orr);
    L->coef_x = std::make_unique<Exchange>(
        std::vector<FieldT *>{&L->dx, &L->dy, &L->dz, &L->m}, L->decomp, m_rank,
        m_comm, oc);
    return L;
  }

  static fmg::Coefficients coeffs(Level &L) {
    return fmg::Coefficients{L.c.data(), L.dx.data(), L.dy.data(), L.dz.data(),
                             L.m.data()};
  }

  template <class Fn> static void for_owned(const Level &L, Fn &&fn) {
    for (int k = 0; k < L.g.n[2]; ++k)
      for (int j = 0; j < L.g.n[1]; ++j)
        for (int i = 0; i < L.g.n[0]; ++i) fn(i, j, k);
  }

  static void mark_written(FieldT &a, FieldT &b, FieldT &c) {
    if constexpr (!is_host) {
      a.note_device_write();
      b.note_device_write();
      c.note_device_write();
    }
  }
  void exchange_u(Level &L) {
    if constexpr (!is_host) L.u.note_device_write();
    L.u_x->exchange();
  }
  static void zero(FieldT &f) {
    if constexpr (is_host) {
      std::fill(f.data(), f.data() + f.size(), 0.0);
    } else {
      gpu_ops::fill(f.data(), f.size(), 0.0);
      f.note_device_write();
    }
  }
  static void fill_owned(Level &L, FieldT &f, double v) {
    if constexpr (is_host) {
      for_owned(
          L, [&](int i, int j, int k) { f.data()[fmg::index(L.g, i, j, k)] = v; });
    } else {
      gpu_ops::fill_owned(L.g, f.data(), v);
      f.note_device_write();
    }
  }
  static void copy_owned(Level &L, const FieldT &src, FieldT &dst) {
    if constexpr (is_host) {
      for_owned(L, [&](int i, int j, int k) {
        const auto s = fmg::index(L.g, i, j, k);
        dst.data()[s] = src.data()[s];
      });
    } else {
      gpu_ops::copy_owned(L.g, src.data(), dst.data());
      dst.note_device_write();
    }
  }

  void restrict_coefficients(Level &F, Level &G) {
    if constexpr (is_host) {
      const auto Cf = coeffs(F);
      for_owned(G, [&](int I, int J, int K) {
        fmg::restrict_coefficients(F.g, Cf, G.g, G.c.data(), G.dx.data(),
                                   G.dy.data(), G.dz.data(), G.m.data(), I, J, K);
      });
    } else {
      gpu_ops::restrict_coefficients(F.g, coeffs(F), G.g, G.c.data(), G.dx.data(),
                                     G.dy.data(), G.dz.data(), G.m.data());
    }
  }

  void sweep(Level &L, int colour) {
    exchange_u(L);
    if constexpr (is_host) {
      const auto C = coeffs(L);
      for_owned(L, [&](int i, int j, int k) {
        fmg::smooth(L.g, C, L.f.data(), L.u.data(), i, j, k, colour);
      });
    } else {
      gpu_ops::smooth(L.g, coeffs(L), L.f.data(), L.u.data(), colour);
    }
  }

  void residual(Level &L) {
    exchange_u(L);
    if constexpr (is_host) {
      const auto C = coeffs(L);
      for_owned(L, [&](int i, int j, int k) {
        const auto s = fmg::index(L.g, i, j, k);
        L.r.data()[s] = L.f.data()[s] - fmg::apply(L.g, C, L.u.data(), i, j, k);
      });
    } else {
      gpu_ops::residual(L.g, coeffs(L), L.f.data(), L.u.data(), L.r.data());
    }
  }

  void cycle(std::size_t l) {
    Level &L = *m_levels[l];
    if (l + 1 == m_levels.size()) {
      for (int s = 0; s < m_c.coarse_sweeps; ++s) { // symmetric: red black black red
        sweep(L, 0);
        sweep(L, 1);
        sweep(L, 1);
        sweep(L, 0);
      }
      return;
    }
    for (int s = 0; s < m_c.pre; ++s) {
      sweep(L, 0);
      sweep(L, 1);
    }
    residual(L);
    Level &G = *m_levels[l + 1];
    if constexpr (!is_host) L.r.note_device_write();
    L.r_x->exchange();
    if constexpr (is_host) {
      for_owned(G, [&](int I, int J, int K) {
        fmg::restrict_residual(L.g, L.r.data(), L.m.data(), G.g, G.m.data(),
                               G.f.data(), I, J, K);
      });
    } else {
      gpu_ops::restrict_residual(L.g, L.r.data(), L.m.data(), G.g, G.m.data(),
                                 G.f.data());
    }
    zero(G.u);
    cycle(l + 1);
    exchange_u(G);
    if constexpr (is_host) {
      for_owned(L, [&](int i, int j, int k) {
        fmg::prolong_add(L.g, L.m.data(), L.u.data(), G.g, G.u.data(), i, j, k);
      });
    } else {
      gpu_ops::prolong_add(L.g, L.m.data(), L.u.data(), G.g, G.u.data());
    }
    for (int s = 0; s < m_c.post; ++s) {
      sweep(L, 1);
      sweep(L, 0);
    }
  }

  MPI_Comm m_comm;
  int m_rank;
  Controls m_c;
  bool m_active[3] = {false, false, false};
  std::vector<std::unique_ptr<Level>> m_levels;
};

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
/// Device launchers (CUDA or HIP by memory space).
template <class MemorySpace> struct FieldMultigrid<MemorySpace>::gpu_ops {
  static constexpr bool cuda =
#if defined(OpenPFC_ENABLE_CUDA)
      std::is_same_v<MemorySpace, CUDASpace>;
#else
      false;
#endif
#if defined(OpenPFC_ENABLE_CUDA)
#define OPENPFC_FMG_CALL(fn, ...)                                                   \
  do {                                                                              \
    if constexpr (cuda)                                                             \
      gpu::fmg::fn##_cuda(__VA_ARGS__);                                             \
    else                                                                            \
      OPENPFC_FMG_CALL_HIP(fn, __VA_ARGS__);                                        \
  } while (0)
#else
#define OPENPFC_FMG_CALL(fn, ...) OPENPFC_FMG_CALL_HIP(fn, __VA_ARGS__)
#endif
#if defined(OpenPFC_ENABLE_HIP)
#define OPENPFC_FMG_CALL_HIP(fn, ...) gpu::fmg::fn##_hip(__VA_ARGS__)
#else
#define OPENPFC_FMG_CALL_HIP(fn, ...)                                               \
  throw std::logic_error("FieldMultigrid: no device backend")
#endif
  static void fill(double *p, std::size_t n, double v) {
    OPENPFC_FMG_CALL(fill, p, n, v);
  }
  static void fill_owned(const fmg::Level &L, double *p, double v) {
    OPENPFC_FMG_CALL(fill_owned, L, p, v);
  }
  static void copy_owned(const fmg::Level &L, const double *s, double *d) {
    OPENPFC_FMG_CALL(copy_owned, L, s, d);
  }
  static void apply(const fmg::Level &L, const fmg::Coefficients &C, const double *u,
                    double *out) {
    OPENPFC_FMG_CALL(apply, L, C, u, out);
  }
  static void smooth(const fmg::Level &L, const fmg::Coefficients &C,
                     const double *f, double *u, int colour) {
    OPENPFC_FMG_CALL(smooth, L, C, f, u, colour);
  }
  static void residual(const fmg::Level &L, const fmg::Coefficients &C,
                       const double *f, const double *u, double *r) {
    OPENPFC_FMG_CALL(residual, L, C, f, u, r);
  }
  static void restrict_coefficients(const fmg::Level &F, const fmg::Coefficients &Cf,
                                    const fmg::Level &G, double *c, double *dx,
                                    double *dy, double *dz, double *m) {
    OPENPFC_FMG_CALL(restrict_coefficients, F, Cf, G, c, dx, dy, dz, m);
  }
  static void restrict_residual(const fmg::Level &F, const double *rf,
                                const double *mf, const fmg::Level &G,
                                const double *mg, double *fg) {
    OPENPFC_FMG_CALL(restrict_residual, F, rf, mf, G, mg, fg);
  }
  static void prolong_add(const fmg::Level &F, const double *mf, double *uf,
                          const fmg::Level &G, const double *ug) {
    OPENPFC_FMG_CALL(prolong_add, F, mf, uf, G, ug);
  }
#undef OPENPFC_FMG_CALL
#undef OPENPFC_FMG_CALL_HIP
};
#else
template <class MemorySpace> struct FieldMultigrid<MemorySpace>::gpu_ops {
  template <class... A> static void fill(A &&...) {}
  template <class... A> static void fill_owned(A &&...) {}
  template <class... A> static void copy_owned(A &&...) {}
  template <class... A> static void apply(A &&...) {}
  template <class... A> static void smooth(A &&...) {}
  template <class... A> static void residual(A &&...) {}
  template <class... A> static void restrict_coefficients(A &&...) {}
  template <class... A> static void restrict_residual(A &&...) {}
  template <class... A> static void prolong_add(A &&...) {}
};
#endif

} // namespace pfc::solvers
