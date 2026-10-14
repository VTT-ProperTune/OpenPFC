// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file nodal_multigrid.hpp
 * @brief Matrix-free distributed geometric multigrid for the nodal Q1 operator.
 *
 * Preconditions (one V-cycle from zero per `apply`) the periodic Q1 (trilinear)
 * finite element operator
 *
 *     A = sum_e a_e K_e + sigma M
 *
 * on a uniform Cartesian node grid with spacing (hx, hy, hz): a_e > 0 is one
 * coefficient per element (element e is indexed by its lowest node), K_e is the
 * exact element stiffness and M the exact consistent mass. Every axis must be
 * periodic. The element matrices factor into 1-D ones,
 *
 *     K_e = sum_d (V / h_d^2) K1_d (x) M1 (x) M1,   M_e = V M1 (x) M1 (x) M1,
 *     K1 = [1 -1; -1 1],  M1 = [1/3 1/6; 1/6 1/3],  V = hx hy hz,
 *
 * so the fine operator is a 27-point row whose weights fold the 8 adjacent
 * element coefficients (nodal_multigrid_kernels.hpp). It is never assembled.
 *
 * Transfers: trilinear nodal prolongation P (coarse node I sits on fine node 2I,
 * an odd fine index averages its two coarse neighbours) and restriction R = P^T.
 * Both are gathers that read one halo layer; nothing is accumulated across ranks.
 *
 * Coarse operators: level l >= `galerkin_from_level` stores the Galerkin product
 * P^T A P as 27 weights per coarse node. Earlier coarse levels are rediscretized
 * with each coarse element coefficient the mean of its 8 children. For a constant
 * coefficient both coincide, because the Q1 spaces are nested: P^T K_h P = K_2h
 * and P^T M_h P = M_2h. With a varying coefficient only the Galerkin product
 * keeps that variational identity.
 *
 * Smoothers: Chebyshev on D^{-1} A over [lambda/range, lambda], lambda being
 * `eigenvalue_safety` times a power-iteration estimate per level whose start
 * vector is a hash of the global node index (so the estimate, and the whole
 * cycle, does not depend on the decomposition); or symmetric 8-colour
 * Gauss-Seidel (colours by the parity of the global index; every level must have
 * even global extents). Pre and post smoothing use the same polynomial or reverse
 * colour order, so the V-cycle is symmetric and can precondition CG.
 *
 * Hierarchy: a level is halved while every global extent is even and its half is
 * at least `coarsest_extent`. A distributed level keeps the fine process grid
 * while every local extent and offset stays even and the coarse local extent is
 * at least 2. Once the coarse level has at most `agglomerate_max` nodes, or the
 * distributed halving is no longer possible, it is agglomerated: each rank
 * restricts into its own region, one allgather gives every rank the whole
 * coarse level, and the rest of the hierarchy is computed redundantly on each
 * rank. Only that bounded coarse part is replicated (`replicated_bytes`). The
 * coarsest level is solved exactly by a dense Cholesky factorization on the host
 * when it is whole on each rank and has at most `dense_max` nodes; otherwise it
 * gets `coarse_sweeps` smoother applications (`coarse_exact()` is then false).
 *
 * Nullspace: with sigma == 0 the constant is in the kernel of every level. The
 * dense coarse matrix is regularized with a rank-one constant term, its
 * right-hand side is projected to zero mean, and `apply` projects the output to
 * zero global mean with one communicator-wide reduction.
 *
 * Vectors passed to `apply_operator` and `apply` must live on the fine owned box
 * with storage halo 1. `apply_operator` writes the halo of its input (the owned
 * values are not changed). Backends: host, CUDA and HIP; the point arithmetic is
 * shared (nodal_multigrid_kernels.hpp).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/solvers/nodal_multigrid_kernels.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/nodal_multigrid_gpu.hpp>
#endif

namespace pfc::solvers {

namespace nmg_device {

/// Device launchers by memory space; the primary template is never called.
template <class Space> struct Launch {
  [[noreturn]] static void unavailable() {
    throw std::logic_error("NodalMultigrid: no device backend");
  }
  template <class... A> static void fill(A &&...) { unavailable(); }
  template <class... A> static void axpy_owned(A &&...) { unavailable(); }
  template <class... A> static void add_owned(A &&...) { unavailable(); }
  template <class... A> static void hash_owned(A &&...) { unavailable(); }
  template <class... A> static void copy_coefficient(A &&...) { unavailable(); }
  template <class... A> static void apply(A &&...) { unavailable(); }
  template <class... A> static void scaled_apply(A &&...) { unavailable(); }
  template <class... A> static void inverse_diagonal(A &&...) { unavailable(); }
  template <class... A> static void residual(A &&...) { unavailable(); }
  template <class... A> static void chebyshev(A &&...) { unavailable(); }
  template <class... A> static void gauss_seidel(A &&...) { unavailable(); }
  template <class... A> static void restrict_residual(A &&...) { unavailable(); }
  template <class... A> static void prolong_add(A &&...) { unavailable(); }
  template <class... A> static void restrict_coefficient(A &&...) { unavailable(); }
  template <class... A> static void galerkin(A &&...) { unavailable(); }
  template <class... A> static double dot_owned(A &&...) {
    unavailable();
    return 0.0;
  }
};

#if defined(OpenPFC_ENABLE_CUDA)
template <> struct Launch<CUDASpace> {
  static void fill(double *p, std::size_t n, double v) {
    gpu::nmg::fill_cuda(p, n, v);
  }
  static void axpy_owned(const nmg::Level &L, double a, const double *x, double *y) {
    gpu::nmg::axpy_owned_cuda(L, a, x, y);
  }
  static void add_owned(const nmg::Level &L, double *y, double v) {
    gpu::nmg::add_owned_cuda(L, y, v);
  }
  static void hash_owned(const nmg::Level &L, double *x) {
    gpu::nmg::hash_owned_cuda(L, x);
  }
  static void copy_coefficient(const nmg::Level &L, const double *src, int h,
                               double *dst) {
    gpu::nmg::copy_coefficient_cuda(L, src, h, dst);
  }
  static void apply(const nmg::Level &L, const nmg::Operator &A, const double *u,
                    double *out) {
    gpu::nmg::apply_cuda(L, A, u, out);
  }
  static void scaled_apply(const nmg::Level &L, const nmg::Operator &A,
                           const double *invd, const double *u, double *out) {
    gpu::nmg::scaled_apply_cuda(L, A, invd, u, out);
  }
  static void inverse_diagonal(const nmg::Level &L, const nmg::Operator &A,
                               double *invd) {
    gpu::nmg::inverse_diagonal_cuda(L, A, invd);
  }
  static void residual(const nmg::Level &L, const nmg::Operator &A, const double *f,
                       const double *u, double *r, int z) {
    gpu::nmg::residual_cuda(L, A, f, u, r, z);
  }
  static void chebyshev(const nmg::Level &L, const nmg::Operator &A, const double *f,
                        const double *u, const double *invd, double *d, double c1,
                        double c2, int z) {
    gpu::nmg::chebyshev_cuda(L, A, f, u, invd, d, c1, c2, z);
  }
  static void gauss_seidel(const nmg::Level &L, const nmg::Operator &A,
                           const double *f, const double *invd, double *u, int c) {
    gpu::nmg::gauss_seidel_cuda(L, A, f, invd, u, c);
  }
  static void restrict_residual(const nmg::Level &F, const double *rf,
                                const nmg::Level &G, double *fg, const int *off) {
    gpu::nmg::restrict_residual_cuda(F, rf, G, fg, off);
  }
  static void prolong_add(const nmg::Level &F, double *uf, const nmg::Level &G,
                          const double *ug, const int *off) {
    gpu::nmg::prolong_add_cuda(F, uf, G, ug, off);
  }
  static void restrict_coefficient(const nmg::Level &F, const double *af,
                                   const nmg::Level &G, double *ag, const int *off) {
    gpu::nmg::restrict_coefficient_cuda(F, af, G, ag, off);
  }
  static void galerkin(const nmg::Level &F, const nmg::Operator &A,
                       const nmg::Level &G, double *sg, std::size_t gp,
                       const int *off) {
    gpu::nmg::galerkin_cuda(F, A, G, sg, gp, off);
  }
  static double dot_owned(const nmg::Level &L, const double *x, const double *y) {
    return gpu::nmg::dot_owned_cuda(L, x, y);
  }
};
#endif
#if defined(OpenPFC_ENABLE_HIP)
template <> struct Launch<HIPSpace> {
  static void fill(double *p, std::size_t n, double v) {
    gpu::nmg::fill_hip(p, n, v);
  }
  static void axpy_owned(const nmg::Level &L, double a, const double *x, double *y) {
    gpu::nmg::axpy_owned_hip(L, a, x, y);
  }
  static void add_owned(const nmg::Level &L, double *y, double v) {
    gpu::nmg::add_owned_hip(L, y, v);
  }
  static void hash_owned(const nmg::Level &L, double *x) {
    gpu::nmg::hash_owned_hip(L, x);
  }
  static void copy_coefficient(const nmg::Level &L, const double *src, int h,
                               double *dst) {
    gpu::nmg::copy_coefficient_hip(L, src, h, dst);
  }
  static void apply(const nmg::Level &L, const nmg::Operator &A, const double *u,
                    double *out) {
    gpu::nmg::apply_hip(L, A, u, out);
  }
  static void scaled_apply(const nmg::Level &L, const nmg::Operator &A,
                           const double *invd, const double *u, double *out) {
    gpu::nmg::scaled_apply_hip(L, A, invd, u, out);
  }
  static void inverse_diagonal(const nmg::Level &L, const nmg::Operator &A,
                               double *invd) {
    gpu::nmg::inverse_diagonal_hip(L, A, invd);
  }
  static void residual(const nmg::Level &L, const nmg::Operator &A, const double *f,
                       const double *u, double *r, int z) {
    gpu::nmg::residual_hip(L, A, f, u, r, z);
  }
  static void chebyshev(const nmg::Level &L, const nmg::Operator &A, const double *f,
                        const double *u, const double *invd, double *d, double c1,
                        double c2, int z) {
    gpu::nmg::chebyshev_hip(L, A, f, u, invd, d, c1, c2, z);
  }
  static void gauss_seidel(const nmg::Level &L, const nmg::Operator &A,
                           const double *f, const double *invd, double *u, int c) {
    gpu::nmg::gauss_seidel_hip(L, A, f, invd, u, c);
  }
  static void restrict_residual(const nmg::Level &F, const double *rf,
                                const nmg::Level &G, double *fg, const int *off) {
    gpu::nmg::restrict_residual_hip(F, rf, G, fg, off);
  }
  static void prolong_add(const nmg::Level &F, double *uf, const nmg::Level &G,
                          const double *ug, const int *off) {
    gpu::nmg::prolong_add_hip(F, uf, G, ug, off);
  }
  static void restrict_coefficient(const nmg::Level &F, const double *af,
                                   const nmg::Level &G, double *ag, const int *off) {
    gpu::nmg::restrict_coefficient_hip(F, af, G, ag, off);
  }
  static void galerkin(const nmg::Level &F, const nmg::Operator &A,
                       const nmg::Level &G, double *sg, std::size_t gp,
                       const int *off) {
    gpu::nmg::galerkin_hip(F, A, G, sg, gp, off);
  }
  static double dot_owned(const nmg::Level &L, const double *x, const double *y) {
    return gpu::nmg::dot_owned_hip(L, x, y);
  }
};
#endif

} // namespace nmg_device

#define OPENPFC_NMG_CALL(fn, ...) nmg_device::Launch<MemorySpace>::fn(__VA_ARGS__)
#define OPENPFC_NMG_RET(fn, ...) nmg_device::Launch<MemorySpace>::fn(__VA_ARGS__)

template <class MemorySpace> class NodalMultigrid {
public:
  using FieldT = data::Field<double, MemorySpace>;

  enum class Smoother { Chebyshev, ColouredGaussSeidel };

  struct Controls {
    double sigma = 0.0; ///< reaction coefficient (>= 0)
    Smoother smoother = Smoother::Chebyshev;
    int chebyshev_degree = 3;    ///< operator applications per smoothing
    double smoothing_range = 15; ///< lambda_max / lambda_min of the target band
    double eigenvalue_safety = 1.1;
    int power_iterations = 15;
    int pre = 1;  ///< smoothings (Chebyshev) or symmetric half-sweeps (GS) before
    int post = 1; ///< and after the coarse correction
    int galerkin_from_level = 1;  ///< first Galerkin level (larger: rediscretize)
    long agglomerate_max = 32768; ///< largest agglomerated coarse level (nodes)
    int coarsest_extent = 4;      ///< smallest global extent created by halving
    long dense_max = 1728;        ///< largest exact (dense Cholesky) coarse solve
    int coarse_sweeps = 20;       ///< smoothings when the coarse solve is inexact
    int max_levels = 32;
    bool profile = false; ///< synchronize and time every phase
    int tag_base = 9000;  ///< first MPI tag; each level uses a block of 1000
  };

  /// Seconds per phase (phases are timed only with `Controls::profile`) and
  /// the setup cost of the last `update`.
  struct Stats {
    double setup = 0, eigenvalues = 0, coarse_operators = 0, factorization = 0;
    double smooth = 0, residual = 0, transfer = 0, halo = 0, gather = 0, coarse = 0;
    long cycles = 0;
  };

  struct LevelInfo {
    Int3 global{}, local{};
    bool replicated = false; ///< every rank holds the whole level
    bool stencil = false;    ///< stored Galerkin stencil (else element coefficients)
    std::size_t bytes = 0;   ///< bytes this rank stores for the level
    double lambda_max = 0;   ///< smoothing bound on D^{-1} A
  };

  NodalMultigrid(const decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
                 Controls controls = {})
      : m_comm(comm), m_rank(rank), m_c(controls) {
    if (!(m_c.sigma >= 0) || m_c.chebyshev_degree < 1 ||
        !(m_c.smoothing_range > 1) || !(m_c.eigenvalue_safety >= 1) ||
        m_c.power_iterations < 1 || m_c.pre < 0 || m_c.post < 0 ||
        m_c.galerkin_from_level < 1 || m_c.agglomerate_max < 1 ||
        m_c.coarsest_extent < 3 || m_c.coarse_sweeps < 1 || m_c.max_levels < 1)
      throw std::invalid_argument("NodalMultigrid: bad controls");
    MPI_Comm_size(comm, &m_size);
    const Domain fine = decomposition::domain(decomp);
    const auto &periodic = domain::get_periodic(fine);
    const auto &gsize = domain::get_size(fine);
    for (int a = 0; a < 3; ++a) {
      if (!periodic[a])
        throw std::invalid_argument("NodalMultigrid: every axis must be periodic");
      if (gsize[a] < 3)
        throw std::invalid_argument("NodalMultigrid: global extents must be >= 3");
    }
    const Int3 grid = decomposition::get_grid(decomp);
    m_levels.push_back(m_size == 1 ? make_whole(fine, 0, false)
                                   : make_distributed(fine, decomp, 0));
    while (static_cast<int>(m_levels.size()) < m_c.max_levels) {
      const Lev &F = *m_levels.back();
      Int3 cs{};
      bool halve = true;
      for (int a = 0; a < 3; ++a) {
        halve = halve && F.g.gn[a] % 2 == 0 && F.g.gn[a] / 2 >= m_c.coarsest_extent;
        cs[a] = F.g.gn[a] / 2;
      }
      if (!halve) break;
      Real3 sp = domain::get_spacing(F.domain);
      for (int a = 0; a < 3; ++a) sp[a] *= 2;
      const Domain cd = domain::with_spacing(cs, sp, {true, true, true});
      const int l = static_cast<int>(m_levels.size());
      if (F.whole) {
        m_levels.push_back(make_whole(cd, l, F.replicated));
        continue;
      }
      // Distributed fine level: owned extents and offsets must halve on every rank.
      int ok = 1;
      for (int a = 0; a < 3; ++a)
        if (F.g.n[a] % 2 != 0 || F.g.low[a] % 2 != 0) ok = 0;
      if (!all(ok)) break;
      const long nodes = static_cast<long>(cs[0]) * cs[1] * cs[2];
      int dist = 1;
      for (int a = 0; a < 3; ++a)
        if (F.g.n[a] / 2 < 2) dist = 0;
      dist = all(dist) && nodes > m_c.agglomerate_max;
      if (dist) {
        auto cdec = decomposition::create(cd, grid);
        const Box3i cb = decomposition::local_box(cdec, m_rank);
        int same = 1;
        for (int a = 0; a < 3; ++a)
          if (cb.low[a] * 2 != F.g.low[a] || cb.size[a] * 2 != F.g.n[a]) same = 0;
        if (all(same)) {
          m_levels.push_back(make_distributed(cd, cdec, l));
          continue;
        }
      }
      if (nodes > m_c.agglomerate_max) break;
      m_levels.push_back(make_whole(cd, l, true));
      Lev &G = *m_levels.back();
      G.regions.resize(static_cast<std::size_t>(m_size));
      for (int p = 0; p < m_size; ++p) {
        const Box3i fb = decomposition::local_box(F.decomp, p);
        Box3i r = fb;
        for (int a = 0; a < 3; ++a) {
          r.low[a] = fb.low[a] / 2;
          r.size[a] = fb.size[a] / 2;
        }
        G.regions[static_cast<std::size_t>(p)] = r;
      }
      for (int a = 0; a < 3; ++a) G.off[a] = F.g.low[a] / 2;
    }
    if (m_c.smoother == Smoother::ColouredGaussSeidel)
      for (const auto &L : m_levels)
        for (int a = 0; a < 3; ++a)
          if (L->g.gn[a] % 2 != 0 && L.get() != m_levels.back().get())
            throw std::invalid_argument(
                "NodalMultigrid: coloured Gauss-Seidel needs even global extents");
    const Lev &C = *m_levels.back();
    m_dense = C.whole && static_cast<long>(nodes_of(C)) <= m_c.dense_max;
  }

  [[nodiscard]] int levels() const noexcept {
    return static_cast<int>(m_levels.size());
  }
  [[nodiscard]] bool coarse_exact() const noexcept { return m_dense; }
  [[nodiscard]] const Stats &stats() const noexcept { return m_stats; }
  void reset_stats() {
    const Stats s = m_stats;
    m_stats = Stats{};
    m_stats.setup = s.setup;
    m_stats.eigenvalues = s.eigenvalues;
    m_stats.coarse_operators = s.coarse_operators;
    m_stats.factorization = s.factorization;
  }

  [[nodiscard]] LevelInfo level_info(int l) const {
    const Lev &L = *m_levels.at(static_cast<std::size_t>(l));
    LevelInfo info;
    for (int a = 0; a < 3; ++a) {
      info.global[a] = L.g.gn[a];
      info.local[a] = L.g.n[a];
    }
    info.replicated = L.replicated;
    info.stencil = L.g.stencil != 0;
    info.bytes = level_bytes(L);
    info.lambda_max = L.lmax;
    return info;
  }

  /// Bytes this rank stores for levels owned by one rank each (distributed).
  [[nodiscard]] std::size_t distributed_bytes() const {
    std::size_t b = 0;
    for (const auto &L : m_levels)
      if (!L->replicated) b += level_bytes(*L);
    return b;
  }
  /// Bytes this rank stores for agglomerated levels held by every rank.
  [[nodiscard]] std::size_t replicated_bytes() const {
    std::size_t b = 0;
    for (const auto &L : m_levels)
      if (L->replicated) b += level_bytes(*L);
    return b + m_chol.size() * sizeof(double);
  }

  /// Copy the owned element coefficients (any storage halo) and build every
  /// coarse level, smoother bound and the coarse factorization.
  void update(const FieldT &coefficient) {
    const auto t0 = std::chrono::steady_clock::now();
    Lev &L0 = *m_levels[0];
    check_vector(coefficient, "coefficient", -1);
    if constexpr (is_host) {
      for_nodes(L0.g.n, [&](int i, int j, int k) {
        L0.a.data()[nmg::index(L0.g, nmg::kCoefficientHalo, i, j, k)] =
            coefficient
                .data()[nmg::index(L0.g, coefficient.storage_halo(), i, j, k)];
      });
    } else {
      const_cast<FieldT &>(coefficient).sync_to_device();
      OPENPFC_NMG_CALL(copy_coefficient, L0.g, coefficient.data(),
                       coefficient.storage_halo(), L0.a.data());
    }
    exchange_coefficients(L0);
    const auto tc = std::chrono::steady_clock::now();
    for (std::size_t l = 1; l < m_levels.size(); ++l)
      build_operator(*m_levels[l - 1], *m_levels[l]);
    sync();
    m_stats.coarse_operators = seconds(tc);
    const auto te = std::chrono::steady_clock::now();
    for (auto &L : m_levels) {
      inverse_diagonal(*L);
      if (L.get() != m_levels.back().get() || !m_dense) estimate_lambda(*L);
    }
    sync();
    m_stats.eigenvalues = seconds(te);
    const auto tf = std::chrono::steady_clock::now();
    if (m_dense) factorize(*m_levels.back());
    m_stats.factorization = seconds(tf);
    m_stats.setup = seconds(t0);
  }

  /// out = A u at the owned fine nodes (u's halo is exchanged in place).
  void apply_operator(const FieldT &u, FieldT &out) {
    Lev &L = *m_levels[0];
    check_vector(u, "u", 1);
    check_vector(out, "out", 1);
    double *up = const_cast<double *>(u.data());
    exchange_vector(L, up);
    if constexpr (is_host) {
      const auto A = op(L);
      for_nodes(L.g.n, [&](int i, int j, int k) {
        out.data()[nmg::vindex(L.g, i, j, k)] = nmg::apply(L.g, A, up, i, j, k);
      });
    } else {
      OPENPFC_NMG_CALL(apply, L.g, op(L), up, out.data());
      out.note_device_write();
    }
  }

  /// z = M^{-1} r: one V-cycle from z = 0 (owned nodes of r read, of z written).
  void apply(const FieldT &r, FieldT &z) {
    Lev &L = *m_levels[0];
    check_vector(r, "r", 1);
    check_vector(z, "z", 1);
    zero(z.data(), z.size());
    cycle(0, r.data(), z.data());
    if (m_c.sigma == 0.0) project_mean(L, z.data());
    if constexpr (!is_host) z.note_device_write();
    ++m_stats.cycles;
  }

  /// Convenience: `precondition(r, z, k)` signature of the Krylov solvers.
  void operator()(const FieldT &r, FieldT &z, int = 0) { apply(r, z); }

private:
  static constexpr bool is_host = std::is_same_v<MemorySpace, HostSpace>;
  using Buf = core::DataBuffer<memory_space_to_backend_t<MemorySpace>, double>;

  /// Full 26-direction halo exchange of raw padded buffers of one level.
  class Halo {
  public:
    Halo(const Domain &d, const Box3i &box, const decomposition::Decomposition &dec,
         int rank, MPI_Comm comm, int width, int nfields, int tag)
        : m_n(nfields) {
      if constexpr (is_host) {
        for (int f = 0; f < nfields; ++f)
          m_host.push_back(std::make_unique<comm::detail::HostFullHalo<double>>(
              box, d, dec, rank, width, comm, tag + 6 * f));
      } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
        m_dev = std::make_unique<DevHalo>(dec, rank, width, comm,
                                          static_cast<std::size_t>(nfields), tag);
#endif
      }
    }
    void exchange(double *const *p, std::size_t padded) {
      if constexpr (is_host) {
        for (int f = 0; f < m_n; ++f)
          m_host[static_cast<std::size_t>(f)]->exchange_halos(p[f], padded);
      } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
        m_dev->exchange(p, nullptr);
#endif
      }
    }

  private:
#if defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_ENABLE_HIP)
    using DevHalo =
        std::conditional_t<std::is_same_v<MemorySpace, CUDASpace>,
                           ::pfc::gpu::DeviceFullHalo<::pfc::cuda::CUDAHaloOps>,
                           ::pfc::gpu::DeviceFullHalo<::pfc::hip::HIPHaloOps>>;
#elif defined(OpenPFC_ENABLE_CUDA)
    using DevHalo = ::pfc::gpu::DeviceFullHalo<::pfc::cuda::CUDAHaloOps>;
#elif defined(OpenPFC_ENABLE_HIP)
    using DevHalo = ::pfc::gpu::DeviceFullHalo<::pfc::hip::HIPHaloOps>;
#endif
    int m_n;
    std::vector<std::unique_ptr<comm::detail::HostFullHalo<double>>> m_host;
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
    std::unique_ptr<DevHalo> m_dev;
#endif
  };

  struct Lev {
    Domain domain;
    decomposition::Decomposition decomp;
    nmg::Level g;
    bool whole = false;      ///< this rank holds the whole global level
    bool replicated = false; ///< agglomerated: every rank holds it
    Buf u, f, r, d, invd, a, s;
    std::unique_ptr<Halo> vx, ax, sx;
    double lmax = 0;
    std::vector<Box3i> regions; ///< per rank, region restricted into (agglomerated)
    int off[3] = {0, 0, 0};     ///< this rank's region in this level
    Lev(const Domain &dm, decomposition::Decomposition dc)
        : domain(dm), decomp(std::move(dc)) {}
  };

  // ---- construction ------------------------------------------------------

  bool all(int v) const {
    int g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_INT, MPI_MIN, m_comm);
    return g != 0;
  }
  static std::size_t nodes_of(const Lev &L) { return nmg::owned_count(L.g); }

  std::unique_ptr<Lev>
  make_distributed(const Domain &d, const decomposition::Decomposition &dc, int l) {
    auto L = std::make_unique<Lev>(d, dc);
    const Box3i b = decomposition::local_box(dc, m_rank);
    init_level(*L, d, b, l == 0);
    L->g.stencil = l >= m_c.galerkin_from_level ? 1 : 0;
    allocate(*L, l);
    make_halos(*L, b, m_rank, m_comm, l);
    return L;
  }

  std::unique_ptr<Lev> make_whole(const Domain &d, int l, bool replicated) {
    auto dec = decomposition::create(d, Int3{1, 1, 1});
    auto L = std::make_unique<Lev>(d, dec);
    L->whole = true;
    L->replicated = replicated;
    const Box3i b = decomposition::local_box(dec, 0);
    init_level(*L, d, b, l == 0);
    L->g.stencil = l >= m_c.galerkin_from_level ? 1 : 0;
    allocate(*L, l);
    make_halos(*L, b, 0, MPI_COMM_SELF, l);
    return L;
  }

  void init_level(Lev &L, const Domain &d, const Box3i &b, bool) {
    const auto &gs = domain::get_size(d);
    const auto &sp = domain::get_spacing(d);
    double h[3];
    for (int a = 0; a < 3; ++a) {
      L.g.n[a] = b.size[a];
      L.g.low[a] = b.low[a];
      L.g.gn[a] = gs[a];
      h[a] = sp[a];
    }
    L.g.sigma = m_c.sigma;
    nmg::set_element_data(L.g, h);
  }

  void allocate(Lev &L, int l) {
    const std::size_t pv = nmg::padded_size(L.g, nmg::kVectorHalo);
    if (l > 0) {
      L.u = Buf(pv);
      L.f = Buf(pv);
    }
    L.r = Buf(pv);
    L.invd = Buf(pv);
    if (m_c.smoother == Smoother::Chebyshev) L.d = Buf(pv);
    if (L.g.stencil)
      L.s = Buf(27 * pv);
    else
      L.a = Buf(nmg::padded_size(L.g, nmg::kCoefficientHalo));
    for (Buf *b : {&L.u, &L.f, &L.r, &L.d, &L.invd, &L.a, &L.s})
      if (b->size()) zero(b->data(), b->size());
  }

  void make_halos(Lev &L, const Box3i &b, int rank, MPI_Comm comm, int l) {
    const int base = m_c.tag_base + 1000 * l;
    L.vx = std::make_unique<Halo>(L.domain, b, L.decomp, rank, comm,
                                  nmg::kVectorHalo, 1, base);
    if (L.g.stencil)
      L.sx = std::make_unique<Halo>(L.domain, b, L.decomp, rank, comm,
                                    nmg::kVectorHalo, 27, base + 100);
    else
      L.ax = std::make_unique<Halo>(L.domain, b, L.decomp, rank, comm,
                                    nmg::kCoefficientHalo, 1, base + 20);
  }

  static std::size_t level_bytes(const Lev &L) {
    std::size_t n = 0;
    for (const Buf *b : {&L.u, &L.f, &L.r, &L.d, &L.invd, &L.a, &L.s})
      n += b->size();
    return n * sizeof(double);
  }

  // ---- small helpers -----------------------------------------------------

  template <class Fn> static void for_nodes(const int n[3], Fn &&fn) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) fn(i, j, k);
  }

  void check_vector(const FieldT &v, const char *what, int halo) const {
    const Lev &L = *m_levels[0];
    for (int a = 0; a < 3; ++a)
      if (v.local_size()[a] != L.g.n[a] || v.box().low[a] != L.g.low[a])
        throw std::invalid_argument(std::string("NodalMultigrid: ") + what +
                                    " is not on the fine owned box");
    if (halo >= 0 && v.storage_halo() != halo)
      throw std::invalid_argument(std::string("NodalMultigrid: ") + what +
                                  " must have storage halo 1");
  }

  static nmg::Operator op(Lev &L) {
    nmg::Operator A;
    A.a = L.a.size() ? L.a.data() : nullptr;
    A.s = L.s.size() ? L.s.data() : nullptr;
    A.plane = nmg::padded_size(L.g, nmg::kVectorHalo);
    return A;
  }

  static double seconds(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
        .count();
  }

  static void sync() {
    if constexpr (!is_host) {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
      pfc::gpuDeviceSynchronize();
#endif
    }
  }

  /// Timer of one phase when profiling (synchronizes before and after).
  class Phase {
  public:
    Phase(const NodalMultigrid &mg, double &acc) : m_on(mg.m_c.profile), m_acc(acc) {
      if (m_on) {
        sync();
        m_t = std::chrono::steady_clock::now();
      }
    }
    ~Phase() {
      if (m_on) {
        sync();
        m_acc += seconds(m_t);
      }
    }

  private:
    bool m_on;
    double &m_acc;
    std::chrono::steady_clock::time_point m_t{};
  };

  static void zero(double *p, std::size_t n) {
    if constexpr (is_host)
      std::fill(p, p + n, 0.0);
    else
      OPENPFC_NMG_CALL(fill, p, n, 0.0);
  }

  void exchange_vector(Lev &L, double *p) {
    Phase t(*this, m_stats.halo);
    double *ptr[1] = {p};
    L.vx->exchange(ptr, nmg::padded_size(L.g, nmg::kVectorHalo));
  }

  void exchange_coefficients(Lev &L) {
    if (L.g.stencil) {
      double *ptr[27];
      const std::size_t pv = nmg::padded_size(L.g, nmg::kVectorHalo);
      for (int o = 0; o < 27; ++o)
        ptr[o] = L.s.data() + static_cast<std::size_t>(o) * pv;
      L.sx->exchange(ptr, pv);
    } else {
      double *ptr[1] = {L.a.data()};
      L.ax->exchange(ptr, nmg::padded_size(L.g, nmg::kCoefficientHalo));
    }
  }

  /// Global sum over the level's owners: local on replicated levels.
  double reduce(const Lev &L, double v) const {
    if (L.whole) return v;
    double g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    return g;
  }

  double dot(Lev &L, const double *x, const double *y) {
    double s = 0;
    if constexpr (is_host) {
      for_nodes(L.g.n, [&](int i, int j, int k) {
        const std::size_t q = nmg::vindex(L.g, i, j, k);
        s += x[q] * y[q];
      });
    } else {
      s = OPENPFC_NMG_RET(dot_owned, L.g, x, y);
    }
    return reduce(L, s);
  }

  void project_mean(Lev &L, double *u) {
    double s = 0;
    if constexpr (is_host) {
      for_nodes(L.g.n,
                [&](int i, int j, int k) { s += u[nmg::vindex(L.g, i, j, k)]; });
    } else {
      if (!m_ones.size()) {
        m_ones = Buf(nmg::padded_size(L.g, nmg::kVectorHalo));
        OPENPFC_NMG_CALL(fill, m_ones.data(), m_ones.size(), 1.0);
      }
      s = OPENPFC_NMG_RET(dot_owned, L.g, u, m_ones.data());
    }
    s = reduce(L, s);
    const double mean = s / (static_cast<double>(L.g.gn[0]) * L.g.gn[1] * L.g.gn[2]);
    if constexpr (is_host) {
      for_nodes(L.g.n,
                [&](int i, int j, int k) { u[nmg::vindex(L.g, i, j, k)] -= mean; });
    } else {
      OPENPFC_NMG_CALL(add_owned, L.g, u, -mean);
    }
  }

  // ---- setup ---------------------------------------------------------------

  /// Coarse operator of G from F, for G's region on this rank, then gather when
  /// G is agglomerated from a distributed F, then exchange G's halos.
  void build_operator(Lev &F, Lev &G) {
    const auto A = op(F);
    const std::size_t gplane = nmg::padded_size(G.g, nmg::kVectorHalo);
    const int rh[3] = {F.g.n[0] / 2, F.g.n[1] / 2, F.g.n[2] / 2};
    if (G.g.stencil) {
      if constexpr (is_host) {
        for_nodes(rh, [&](int I, int J, int K) {
          nmg::galerkin(F.g, A, G.g, G.s.data(), gplane, G.off, I, J, K);
        });
      } else {
        OPENPFC_NMG_CALL(galerkin, F.g, A, G.g, G.s.data(), gplane, G.off);
      }
    } else {
      if (F.g.stencil)
        throw std::logic_error(
            "NodalMultigrid: cannot rediscretize below a stencil");
      if constexpr (is_host) {
        for_nodes(rh, [&](int I, int J, int K) {
          nmg::restrict_coefficient(F.g, F.a.data(), G.g, G.a.data(), G.off, I, J,
                                    K);
        });
      } else {
        OPENPFC_NMG_CALL(restrict_coefficient, F.g, F.a.data(), G.g, G.a.data(),
                         G.off);
      }
    }
    if (G.replicated && !F.whole) {
      if (G.g.stencil)
        for (int o = 0; o < 27; ++o)
          gather(G, G.s.data() + static_cast<std::size_t>(o) * gplane,
                 nmg::kVectorHalo);
      else
        gather(G, G.a.data(), nmg::kCoefficientHalo);
    }
    exchange_coefficients(G);
  }

  /// Every rank's region of an agglomerated level -> every rank (host staging).
  void gather(Lev &G, double *p, int H) {
    Phase t(*this, m_stats.gather);
    const std::size_t padded = nmg::padded_size(G.g, H);
    std::vector<double> img(padded);
    to_host(p, img.data(), padded);
    std::vector<int> counts(static_cast<std::size_t>(m_size)),
        displs(static_cast<std::size_t>(m_size));
    int total = 0;
    for (int q = 0; q < m_size; ++q) {
      const Box3i &r = G.regions[static_cast<std::size_t>(q)];
      counts[static_cast<std::size_t>(q)] = r.size[0] * r.size[1] * r.size[2];
      displs[static_cast<std::size_t>(q)] = total;
      total += counts[static_cast<std::size_t>(q)];
    }
    const Box3i &mine = G.regions[static_cast<std::size_t>(m_rank)];
    std::vector<double> send(
        static_cast<std::size_t>(counts[static_cast<std::size_t>(m_rank)]));
    std::size_t c = 0;
    for_nodes(mine.size.data(), [&](int i, int j, int k) {
      send[c++] =
          img[nmg::index(G.g, H, mine.low[0] + i, mine.low[1] + j, mine.low[2] + k)];
    });
    std::vector<double> recv(static_cast<std::size_t>(total));
    MPI_Allgatherv(send.data(), static_cast<int>(send.size()), MPI_DOUBLE,
                   recv.data(), counts.data(), displs.data(), MPI_DOUBLE, m_comm);
    for (int q = 0; q < m_size; ++q) {
      const Box3i &r = G.regions[static_cast<std::size_t>(q)];
      std::size_t e = static_cast<std::size_t>(displs[static_cast<std::size_t>(q)]);
      for_nodes(r.size.data(), [&](int i, int j, int k) {
        img[nmg::index(G.g, H, r.low[0] + i, r.low[1] + j, r.low[2] + k)] =
            recv[e++];
      });
    }
    from_host(img.data(), p, padded);
  }

  static void to_host(const double *src, double *dst, std::size_t n) {
    if constexpr (is_host) {
      std::copy(src, src + n, dst);
    } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
      if (pfc::gpuMemcpy(dst, src, n * sizeof(double), pfc::gpuMemcpyDeviceToHost) !=
          pfc::gpuSuccess)
        throw std::runtime_error("NodalMultigrid: device to host copy failed");
#endif
    }
  }
  static void from_host(const double *src, double *dst, std::size_t n) {
    if constexpr (is_host) {
      std::copy(src, src + n, dst);
    } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
      if (pfc::gpuMemcpy(dst, src, n * sizeof(double), pfc::gpuMemcpyHostToDevice) !=
          pfc::gpuSuccess)
        throw std::runtime_error("NodalMultigrid: host to device copy failed");
#endif
    }
  }

  void inverse_diagonal(Lev &L) {
    const auto A = op(L);
    if constexpr (is_host) {
      for_nodes(L.g.n, [&](int i, int j, int k) {
        L.invd.data()[nmg::vindex(L.g, i, j, k)] =
            1.0 / nmg::diagonal(L.g, A, i, j, k);
      });
    } else {
      OPENPFC_NMG_CALL(inverse_diagonal, L.g, A, L.invd.data());
    }
  }

  /// lambda_max(D^{-1} A) by power iteration from the hashed start vector.
  void estimate_lambda(Lev &L) {
    const auto A = op(L);
    double *x = L.r.data(), *y = L.f.size() ? L.f.data() : nullptr;
    Buf tmp;
    if (!y) {
      tmp = Buf(nmg::padded_size(L.g, nmg::kVectorHalo));
      zero(tmp.data(), tmp.size());
      y = tmp.data();
    }
    if constexpr (is_host) {
      for_nodes(L.g.n, [&](int i, int j, int k) {
        x[nmg::vindex(L.g, i, j, k)] =
            nmg::hash_value(L.g.low[0] + i, L.g.low[1] + j, L.g.low[2] + k);
      });
    } else {
      OPENPFC_NMG_CALL(hash_owned, L.g, x);
    }
    double lambda = 0;
    for (int it = 0; it < m_c.power_iterations; ++it) {
      const double nx = std::sqrt(dot(L, x, x));
      scale(L, x, 1.0 / nx);
      exchange_vector(L, x);
      if constexpr (is_host) {
        for_nodes(L.g.n, [&](int i, int j, int k) {
          const std::size_t q = nmg::vindex(L.g, i, j, k);
          y[q] = L.invd.data()[q] * nmg::apply(L.g, A, x, i, j, k);
        });
      } else {
        OPENPFC_NMG_CALL(scaled_apply, L.g, A, L.invd.data(), x, y);
      }
      lambda = std::sqrt(dot(L, y, y));
      std::swap(x, y);
    }
    L.lmax = m_c.eigenvalue_safety * lambda;
    zero(L.r.data(), L.r.size());
    if (L.f.size()) zero(L.f.data(), L.f.size());
  }

  void scale(Lev &L, double *x, double s) {
    if constexpr (is_host) {
      for_nodes(L.g.n,
                [&](int i, int j, int k) { x[nmg::vindex(L.g, i, j, k)] *= s; });
    } else {
      // x += (s - 1) x
      OPENPFC_NMG_CALL(axpy_owned, L.g, s - 1.0, x, x);
    }
  }

  /// Dense Cholesky of the whole coarsest level (host), constant term if sigma=0.
  void factorize(Lev &L) {
    const int n = static_cast<int>(nodes_of(L));
    const std::size_t pv = nmg::padded_size(L.g, nmg::kVectorHalo);
    std::vector<double> a, s;
    nmg::Operator A;
    A.plane = pv;
    if (L.g.stencil) {
      s.resize(L.s.size());
      to_host(L.s.data(), s.data(), s.size());
      A.s = s.data();
    } else {
      a.resize(L.a.size());
      to_host(L.a.data(), a.data(), a.size());
      A.a = a.data();
    }
    std::vector<double> M(static_cast<std::size_t>(n) * n, 0.0);
    auto id = [&](int i, int j, int k) {
      const auto w = [](int v, int m) { return ((v % m) + m) % m; };
      return w(i, L.g.gn[0]) +
             L.g.gn[0] * (w(j, L.g.gn[1]) + L.g.gn[1] * w(k, L.g.gn[2]));
    };
    double dsum = 0;
    for_nodes(L.g.n, [&](int i, int j, int k) {
      double w[27];
      nmg::weights(L.g, A, i, j, k, w);
      const int row = id(i, j, k);
      for (int oz = -1; oz <= 1; ++oz)
        for (int oy = -1; oy <= 1; ++oy)
          for (int ox = -1; ox <= 1; ++ox)
            M[static_cast<std::size_t>(row) * n + id(i + ox, j + oy, k + oz)] +=
                w[nmg::offset_index(ox, oy, oz)];
      dsum += w[13];
    });
    if (m_c.sigma == 0.0) {
      const double c = dsum / n / n;
      for (double &v : M) v += c;
    }
    for (int j = 0; j < n; ++j) { // M = L L^T in the lower triangle
      double d = M[static_cast<std::size_t>(j) * n + j];
      for (int p = 0; p < j; ++p)
        d -= M[static_cast<std::size_t>(j) * n + p] *
             M[static_cast<std::size_t>(j) * n + p];
      if (!(d > 0))
        throw std::runtime_error("NodalMultigrid: coarse matrix is not SPD");
      d = std::sqrt(d);
      M[static_cast<std::size_t>(j) * n + j] = d;
      for (int i = j + 1; i < n; ++i) {
        double v = M[static_cast<std::size_t>(i) * n + j];
        for (int p = 0; p < j; ++p)
          v -= M[static_cast<std::size_t>(i) * n + p] *
               M[static_cast<std::size_t>(j) * n + p];
        M[static_cast<std::size_t>(i) * n + j] = v / d;
      }
    }
    m_chol = std::move(M);
    m_coarse_img.assign(pv, 0.0);
  }

  void dense_solve(Lev &L, const double *f, double *u) {
    Phase t(*this, m_stats.coarse);
    const int n = static_cast<int>(nodes_of(L));
    const std::size_t pv = nmg::padded_size(L.g, nmg::kVectorHalo);
    to_host(f, m_coarse_img.data(), pv);
    std::vector<double> b(static_cast<std::size_t>(n));
    int q = 0;
    for_nodes(L.g.n, [&](int i, int j, int k) {
      b[static_cast<std::size_t>(q++)] = m_coarse_img[nmg::vindex(L.g, i, j, k)];
    });
    if (m_c.sigma == 0.0) {
      double m = 0;
      for (double v : b) m += v;
      m /= n;
      for (double &v : b) v -= m;
    }
    const double *M = m_chol.data();
    for (int i = 0; i < n; ++i) {
      double v = b[static_cast<std::size_t>(i)];
      for (int p = 0; p < i; ++p)
        v -= M[static_cast<std::size_t>(i) * n + p] * b[static_cast<std::size_t>(p)];
      b[static_cast<std::size_t>(i)] = v / M[static_cast<std::size_t>(i) * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
      double v = b[static_cast<std::size_t>(i)];
      for (int p = i + 1; p < n; ++p)
        v -= M[static_cast<std::size_t>(p) * n + i] * b[static_cast<std::size_t>(p)];
      b[static_cast<std::size_t>(i)] = v / M[static_cast<std::size_t>(i) * n + i];
    }
    q = 0;
    for_nodes(L.g.n, [&](int i, int j, int k) {
      m_coarse_img[nmg::vindex(L.g, i, j, k)] = b[static_cast<std::size_t>(q++)];
    });
    from_host(m_coarse_img.data(), u, pv);
  }

  // ---- cycle ---------------------------------------------------------------

  /// Chebyshev smoothing of u toward A u = f (zero_u: u enters as zero).
  void chebyshev(Lev &L, const double *f, double *u, bool zero_u) {
    const auto A = op(L);
    const double lmax = L.lmax, lmin = lmax / m_c.smoothing_range;
    const double theta = 0.5 * (lmax + lmin), delta = 0.5 * (lmax - lmin);
    const double sigma1 = theta / delta;
    double rho = 1.0 / sigma1;
    for (int k = 0; k < m_c.chebyshev_degree; ++k) {
      double c1, c2;
      if (k == 0) {
        c1 = 0.0;
        c2 = 1.0 / theta;
      } else {
        const double rho1 = 1.0 / (2.0 * sigma1 - rho);
        c1 = rho1 * rho;
        c2 = 2.0 * rho1 / delta;
        rho = rho1;
      }
      const int zu = (zero_u && k == 0) ? 1 : 0;
      if (!zu) exchange_vector(L, u);
      {
        Phase t(*this, m_stats.smooth);
        if constexpr (is_host) {
          for_nodes(L.g.n, [&](int i, int j, int kk) {
            nmg::chebyshev(L.g, A, f, u, L.invd.data(), L.d.data(), c1, c2, i, j, kk,
                           zu);
          });
          for_nodes(L.g.n, [&](int i, int j, int kk) {
            const std::size_t q = nmg::vindex(L.g, i, j, kk);
            u[q] += L.d.data()[q];
          });
        } else {
          OPENPFC_NMG_CALL(chebyshev, L.g, A, f, u, L.invd.data(), L.d.data(), c1,
                           c2, zu);
          OPENPFC_NMG_CALL(axpy_owned, L.g, 1.0, L.d.data(), u);
        }
      }
    }
  }

  void gauss_seidel(Lev &L, const double *f, double *u, bool forward) {
    const auto A = op(L);
    for (int q = 0; q < 8; ++q) {
      const int colour = forward ? q : 7 - q;
      exchange_vector(L, u);
      Phase t(*this, m_stats.smooth);
      if constexpr (is_host) {
        for_nodes(L.g.n, [&](int i, int j, int k) {
          nmg::gauss_seidel(L.g, A, f, L.invd.data(), u, colour, i, j, k);
        });
      } else {
        OPENPFC_NMG_CALL(gauss_seidel, L.g, A, f, L.invd.data(), u, colour);
      }
    }
  }

  void smooth(Lev &L, const double *f, double *u, int count, bool pre, bool zero_u) {
    for (int s = 0; s < count; ++s) {
      if (m_c.smoother == Smoother::Chebyshev)
        chebyshev(L, f, u, zero_u && s == 0);
      else
        gauss_seidel(L, f, u, pre);
    }
  }

  void cycle(std::size_t l, const double *f, double *u) {
    Lev &L = *m_levels[l];
    if (l + 1 == m_levels.size()) {
      if (m_dense) {
        dense_solve(L, f, u);
      } else {
        for (int s = 0; s < m_c.coarse_sweeps; ++s) {
          if (m_c.smoother == Smoother::Chebyshev) {
            chebyshev(L, f, u, s == 0);
          } else {
            gauss_seidel(L, f, u, true);
            gauss_seidel(L, f, u, false);
          }
        }
      }
      return;
    }
    smooth(L, f, u, m_c.pre, true, true);
    exchange_vector(L, u);
    const auto A = op(L);
    {
      Phase t(*this, m_stats.residual);
      if constexpr (is_host) {
        for_nodes(L.g.n, [&](int i, int j, int k) {
          nmg::residual(L.g, A, f, u, L.r.data(), i, j, k, 0);
        });
      } else {
        OPENPFC_NMG_CALL(residual, L.g, A, f, u, L.r.data(), 0);
      }
    }
    exchange_vector(L, L.r.data());
    Lev &G = *m_levels[l + 1];
    {
      Phase t(*this, m_stats.transfer);
      const int rh[3] = {L.g.n[0] / 2, L.g.n[1] / 2, L.g.n[2] / 2};
      if constexpr (is_host) {
        for_nodes(rh, [&](int I, int J, int K) {
          nmg::restrict_residual(L.g, L.r.data(), G.g, G.f.data(), G.off, I, J, K);
        });
      } else {
        OPENPFC_NMG_CALL(restrict_residual, L.g, L.r.data(), G.g, G.f.data(), G.off);
      }
    }
    if (G.replicated && !L.whole) gather(G, G.f.data(), nmg::kVectorHalo);
    zero(G.u.data(), G.u.size());
    cycle(l + 1, G.f.data(), G.u.data());
    exchange_vector(G, G.u.data());
    {
      Phase t(*this, m_stats.transfer);
      if constexpr (is_host) {
        for_nodes(L.g.n, [&](int i, int j, int k) {
          nmg::prolong_add(L.g, u, G.g, G.u.data(), G.off, i, j, k);
        });
      } else {
        OPENPFC_NMG_CALL(prolong_add, L.g, u, G.g, G.u.data(), G.off);
      }
    }
    smooth(L, f, u, m_c.post, false, false);
  }

  MPI_Comm m_comm;
  int m_rank;
  int m_size = 1;
  Controls m_c;
  bool m_dense = false;
  Stats m_stats;
  std::vector<std::unique_ptr<Lev>> m_levels;
  std::vector<double> m_chol, m_coarse_img;
  Buf m_ones;
};

#undef OPENPFC_NMG_CALL
#undef OPENPFC_NMG_RET

} // namespace pfc::solvers
