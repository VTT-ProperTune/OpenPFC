// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file nodal_dirichlet_multigrid.hpp
 * @brief Bounded host/MPI Q1 multigrid for nonperiodic free-node fields.
 *
 * This first nonperiodic qualification uses OpenPFC's decomposition, Field,
 * Host FullHalo and Krylov interfaces. No new partitioner or field runtime
 * is defined here. The existing periodic NodalMultigrid remains unchanged.
 * Coefficients are constant and each physical boundary node is eliminated.
 *
 * The input Domain contains the FREE nodes only: a physical Q1 mesh with
 * 2^k elements has 2^k-1 free nodes on each nonperiodic axis. Spacing is the
 * physical element spacing. Hierarchy transfers are P (trilinear) and P^T,
 * with one halo-exchanged, row-owned action at each distributed level. Only
 * the bounded coarsest problem is replicated for exact Cholesky.
 *
 * CPU only. The later integration into the shared NodalMultigrid family must
 * reuse this numerical acceptance, rather than claim HIP/CUDA parity here.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/solvers/nodal_multigrid_kernels.hpp>

namespace pfc::solvers {

class NodalDirichletMultigrid {
public:
  using FieldT = data::Field<double, HostSpace>;

  struct Controls {
    double diffusion = 1.0;          ///< Positive, spatially constant Q1 diffusion.
    double reaction = 0.0;           ///< Nonnegative consistent Q1 mass term.
    double jacobi_weight = 0.65;     ///< Capped by a Gershgorin spectral bound.
    int pre = 2;
    int post = 2;
    int coarsest_free_extent = 7;    ///< Current nested Q1 qualification gate.
    int max_coarse_unknowns = 1024;  ///< Guard against unbounded dense replication.
  };

  NodalDirichletMultigrid(const decomposition::Decomposition &fine, int rank,
                          MPI_Comm comm)
      : NodalDirichletMultigrid(fine, rank, comm, Controls{}) {}

  NodalDirichletMultigrid(const decomposition::Decomposition &fine, int rank,
                          MPI_Comm comm, Controls controls)
      : m_comm(comm), m_rank(rank), m_controls(controls) {
    if (comm == MPI_COMM_NULL)
      throw std::invalid_argument("NodalDirichletMultigrid: null communicator");
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);
    if (rank != m_rank)
      throw std::invalid_argument("NodalDirichletMultigrid: MPI rank mismatch");
    if (decomposition::get_num_domains(fine) != m_size)
      throw std::invalid_argument("NodalDirichletMultigrid: decomposition size mismatch");
    if (!(controls.diffusion > 0.0) || !std::isfinite(controls.diffusion) ||
        !(controls.reaction >= 0.0) || !std::isfinite(controls.reaction) ||
        !(controls.jacobi_weight > 0.0 && controls.jacobi_weight < 1.0) ||
        !std::isfinite(controls.jacobi_weight) || controls.pre < 1 ||
        controls.post < 1 || controls.pre != controls.post ||
        controls.coarsest_free_extent < 3 || controls.max_coarse_unknowns < 1)
      throw std::invalid_argument("NodalDirichletMultigrid: invalid controls");

    const Domain domain0 = decomposition::domain(fine);
    const auto periodic = domain::get_periodic(domain0);
    const auto size = domain::get_size(domain0);
    for (int d = 0; d < 3; ++d) {
      if (periodic[d])
        throw std::invalid_argument("NodalDirichletMultigrid: every axis must be nonperiodic");
      if (size[d] < 2 * controls.coarsest_free_extent + 1 ||
          !power_of_two(static_cast<long long>(size[d]) + 1))
        throw std::invalid_argument(
            "NodalDirichletMultigrid: free extents must be 2^k-1 and >= 15");
    }

    const auto proc_grid = decomposition::get_grid(fine);
    Domain current = domain0;
    decomposition::Decomposition current_dec = fine;
    while (true) {
      const auto global = domain::get_size(current);
      bool coarsen = true;
      for (int d = 0; d < 3; ++d)
        coarsen = coarsen && global[d] > controls.coarsest_free_extent;
      if (!coarsen) break;
      m_levels.push_back(
          std::make_unique<Level>(current, current_dec, m_rank, m_comm,
                                  make_stencil(current, controls)));
      Int3 next_size{};
      Real3 next_spacing = domain::get_spacing(current);
      for (int d = 0; d < 3; ++d) {
        next_size[d] = (global[d] - 1) / 2;
        next_spacing[d] *= 2.0;
        if (proc_grid[d] > next_size[d])
          throw std::invalid_argument(
              "NodalDirichletMultigrid: process grid exceeds coarse free extent");
      }
      Domain next = domain::with_spacing(next_size, next_spacing,
                                         {false, false, false});
      auto next_dec = decomposition::create(next, proc_grid);
      validate_transfer_reach(current_dec, next_dec, global, next_size);
      current = next;
      current_dec = std::move(next_dec);
    }
    if (m_levels.empty())
      throw std::invalid_argument("NodalDirichletMultigrid: at least one V-cycle level required");
    m_coarse_domain = current;
    const auto n = product(domain::get_size(current));
    if (n > static_cast<std::size_t>(controls.max_coarse_unknowns) ||
        n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("NodalDirichletMultigrid: coarse dense capacity exceeded");
    m_coarse_rhs.assign(n, 0.0);
    m_coarse_x.assign(n, 0.0);
    factor_coarse(make_stencil(current, controls));
  }

  [[nodiscard]] int levels() const noexcept {
    return static_cast<int>(m_levels.size()) + 1;
  }
  [[nodiscard]] std::size_t coarse_unknowns() const noexcept {
    return m_coarse_rhs.size();
  }
  [[nodiscard]] std::size_t local_unknowns() const noexcept {
    return owned_count(m_levels.front()->box);
  }

  /// Matrix-free Q1 action, writing unique owned free nodes only.
  void apply_operator(const FieldT &x, FieldT &y) {
    check_field(x);
    check_field(y);
    action(*m_levels.front(), x, y);
  }

  /// Apply one zero-initialized SPD V-cycle as a fixed CG preconditioner.
  void apply(const FieldT &rhs, FieldT &out) {
    check_field(rhs);
    check_field(out);
    auto &fine = *m_levels.front();
    each_owned(fine.box, [&](int i, int j, int k) {
      fine.rhs(i, j, k) = rhs(i, j, k);
      fine.x(i, j, k) = 0.0;
    });
    cycle(0);
    each_owned(fine.box, [&](int i, int j, int k) {
      out(i, j, k) = fine.x(i, j, k);
    });
  }

  void operator()(const FieldT &r, FieldT &z, int = 0) { apply(r, z); }

private:
  static bool power_of_two(long long n) {
    return n > 0 && (n & (n - 1)) == 0;
  }

  static std::size_t product(const Int3 &n) {
    std::size_t value = 1;
    for (int d = 0; d < 3; ++d) {
      if (n[d] < 1 || value > std::numeric_limits<std::size_t>::max() /
                                    static_cast<std::size_t>(n[d]))
        throw std::invalid_argument("NodalDirichletMultigrid: invalid product");
      value *= static_cast<std::size_t>(n[d]);
    }
    return value;
  }

  static std::size_t owned_count(const Box3i &b) { return product(b.size); }

  static std::size_t linear(const Int3 &n, int i, int j, int k) {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(n[0]) *
           (static_cast<std::size_t>(j) + static_cast<std::size_t>(n[1]) * k);
  }

  template <typename Fn>
  static void each_owned(const Box3i &b, Fn fn) {
    for (int k = 0; k < b.size[2]; ++k)
      for (int j = 0; j < b.size[1]; ++j)
        for (int i = 0; i < b.size[0]; ++i) fn(i, j, k);
  }

  static std::array<double, 27> make_stencil(const Domain &d,
                                               const Controls &controls) {
    nmg::Level reference;
    const auto spacing = domain::get_spacing(d);
    const double h[3] = {spacing[0], spacing[1], spacing[2]};
    nmg::set_element_data(reference, h);
    std::array<double, 27> weights{};
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int slot = nmg::offset_index(dx, dy, dz);
          const double multiplicity =
              (dx == 0 ? 2.0 : 1.0) * (dy == 0 ? 2.0 : 1.0) *
              (dz == 0 ? 2.0 : 1.0);
          weights[slot] = controls.diffusion * multiplicity *
                              reference.ke[slot] +
                          controls.reaction * reference.mass[slot];
        }
    return weights;
  }

  static comm::HaloExchangeOptions full_halo() {
    comm::HaloExchangeOptions opt;
    opt.connectivity = comm::HaloConnectivity::Full;
    opt.persistent = false;
    return opt;
  }

  struct Level {
    Domain domain;
    decomposition::Decomposition decomp;
    Box3i box;
    FieldT x, rhs, residual, ax, scratch;
    comm::HaloExchange<HostSpace, double> halo;
    std::array<double, 27> weights;
    double jacobi = 0.0;

    Level(const Domain &d, const decomposition::Decomposition &dc, int rank,
          MPI_Comm comm, const std::array<double, 27> &w)
        : domain(d), decomp(dc),
          box(decomposition::local_box(decomp, rank)),
          x(d, box, 0), rhs(d, box, 0), residual(d, box, 0), ax(d, box, 0),
          scratch(d, box, 2), halo(scratch, decomp, rank, comm, full_halo()),
          weights(w) {
      const double diag = weights[13];
      double abs_sum = 0.0;
      for (double v : weights) abs_sum += std::abs(v);
      if (!(diag > 0.0) || !(abs_sum >= diag))
        throw std::logic_error("NodalDirichletMultigrid: nonpositive Q1 diagonal");
      jacobi = std::min(0.65, 1.5 * diag / abs_sum) / diag;
    }
  };

  /// Only OpenPFC's Decomposition chooses boxes. The solver checks whether
  /// geometric Q1 transfers fit the established 2-cell halo for every rank.
  static void validate_transfer_reach(const decomposition::Decomposition &fine,
                                      const decomposition::Decomposition &coarse,
                                      const Int3 &nf, const Int3 &nc) {
    const int nproc = decomposition::get_num_domains(fine);
    if (decomposition::get_num_domains(coarse) != nproc)
      throw std::invalid_argument("NodalDirichletMultigrid: process count changed");
    constexpr int H = 2;
    for (int rank = 0; rank < nproc; ++rank) {
      const Box3i f = decomposition::local_box(fine, rank);
      const Box3i c = decomposition::local_box(coarse, rank);
      for (int d = 0; d < 3; ++d) {
        const int rlo = std::max(0, 2 * c.low[d]);
        const int rhi = std::min(nf[d] - 1, 2 * c.high[d] + 2);
        const int plo = std::max(0, (f.low[d] + 1) / 2 - 1);
        const int phi = std::min(nc[d] - 1, (f.high[d] + 1) / 2);
        if (rlo < f.low[d] - H || rhi > f.high[d] + H ||
            plo < c.low[d] - H || phi > c.high[d] + H)
          throw std::invalid_argument(
              "NodalDirichletMultigrid: nested Q1 transfer exceeds 2-cell halo");
      }
    }
  }

  void check_field(const FieldT &field) const {
    const auto &fine = *m_levels.front();
    if (field.domain() != fine.domain ||
        field.box().low != fine.box.low ||
        field.box().size != fine.box.size)
      throw std::invalid_argument("NodalDirichletMultigrid: field box mismatch");
  }

  static void load_halo(Level &level, const FieldT &src) {
    std::fill(level.scratch.vec().begin(), level.scratch.vec().end(), 0.0);
    each_owned(level.box, [&](int i, int j, int k) {
      level.scratch(i, j, k) = src(i, j, k);
    });
    level.halo.exchange();
  }

  static double sampled(const Level &L, int i, int j, int k) {
    const auto size = domain::get_size(L.domain);
    if (i < 0 || j < 0 || k < 0 ||
        i >= size[0] || j >= size[1] || k >= size[2]) return 0.0;
    const int li = i - L.box.low[0], lj = j - L.box.low[1],
              lk = k - L.box.low[2];
    for (int d = 0; d < 3; ++d) {
      const int idx = d == 0 ? li : (d == 1 ? lj : lk);
      if (idx < -2 || idx >= L.box.size[d] + 2)
        throw std::logic_error("NodalDirichletMultigrid: missing Q1 halo");
    }
    return L.scratch(li, lj, lk);
  }

  static void action(Level &L, const FieldT &input, FieldT &output) {
    load_halo(L, input);
    each_owned(L.box, [&](int i, int j, int k) {
      double sum = 0.0;
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            sum += L.weights[nmg::offset_index(dx, dy, dz)] *
                   L.scratch(i + dx, j + dy, k + dz);
      output(i, j, k) = sum;
    });
  }

  static void smooth(Level &L, int count) {
    for (int sweep = 0; sweep < count; ++sweep) {
      action(L, L.x, L.ax);
      each_owned(L.box, [&](int i, int j, int k) {
        L.x(i, j, k) += L.jacobi * (L.rhs(i, j, k) - L.ax(i, j, k));
      });
    }
  }

  template <typename Sample>
  static double interpolate(int fi, int fj, int fk, Sample get) {
    const int f[3] = {fi + 1, fj + 1, fk + 1};
    const int base[3] = {f[0] / 2 - 1, f[1] / 2 - 1, f[2] / 2 - 1};
    const int cnt[3] = {(f[0] & 1) ? 2 : 1, (f[1] & 1) ? 2 : 1,
                        (f[2] & 1) ? 2 : 1};
    const double w = ((f[0] & 1) ? 0.5 : 1.0) *
                     ((f[1] & 1) ? 0.5 : 1.0) *
                     ((f[2] & 1) ? 0.5 : 1.0);
    double value = 0.0;
    for (int dz = 0; dz < cnt[2]; ++dz)
      for (int dy = 0; dy < cnt[1]; ++dy)
        for (int dx = 0; dx < cnt[0]; ++dx)
          value += w * get(base[0] + dx, base[1] + dy, base[2] + dz);
    return value;
  }

  static void restrict_distributed(Level &fine, Level &coarse) {
    load_halo(fine, fine.residual);
    each_owned(coarse.box, [&](int i, int j, int k) {
      const int ci = coarse.box.low[0] + i;
      const int cj = coarse.box.low[1] + j;
      const int ck = coarse.box.low[2] + k;
      const int fi = 2 * ci + 1, fj = 2 * cj + 1, fk = 2 * ck + 1;
      double v = 0.0;
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const double weight = (dx ? 0.5 : 1.0) *
                                  (dy ? 0.5 : 1.0) * (dz ? 0.5 : 1.0);
            v += weight * sampled(fine, fi + dx, fj + dy, fk + dz);
          }
      coarse.rhs(i, j, k) = v;
    });
  }

  static void prolong_distributed(Level &fine, Level &coarse) {
    load_halo(coarse, coarse.x);
    each_owned(fine.box, [&](int i, int j, int k) {
      const int fi = fine.box.low[0] + i, fj = fine.box.low[1] + j,
                fk = fine.box.low[2] + k;
      fine.x(i, j, k) += interpolate(fi, fj, fk, [&](int ci, int cj, int ck) {
        return sampled(coarse, ci, cj, ck);
      });
    });
  }

  void restrict_coarse(Level &fine) {
    std::vector<double> local(m_coarse_rhs.size(), 0.0);
    const auto csize = domain::get_size(m_coarse_domain);
    each_owned(fine.box, [&](int i, int j, int k) {
      const int f[3] = {fine.box.low[0] + i + 1,
                        fine.box.low[1] + j + 1,
                        fine.box.low[2] + k + 1};
      const int base[3] = {f[0] / 2 - 1, f[1] / 2 - 1, f[2] / 2 - 1};
      const int cnt[3] = {(f[0] & 1) ? 2 : 1, (f[1] & 1) ? 2 : 1,
                          (f[2] & 1) ? 2 : 1};
      const double weight = ((f[0] & 1) ? 0.5 : 1.0) *
                            ((f[1] & 1) ? 0.5 : 1.0) *
                            ((f[2] & 1) ? 0.5 : 1.0);
      const double residual = fine.residual(i, j, k);
      for (int dz = 0; dz < cnt[2]; ++dz)
        for (int dy = 0; dy < cnt[1]; ++dy)
          for (int dx = 0; dx < cnt[0]; ++dx) {
            const int ci = base[0] + dx, cj = base[1] + dy,
                      ck = base[2] + dz;
            if (ci >= 0 && cj >= 0 && ck >= 0 &&
                ci < csize[0] && cj < csize[1] && ck < csize[2])
              local[linear(csize, ci, cj, ck)] += weight * residual;
          }
    });
    MPI_Allreduce(local.data(), m_coarse_rhs.data(),
                  static_cast<int>(m_coarse_rhs.size()), MPI_DOUBLE, MPI_SUM,
                  m_comm);
  }

  void prolong_coarse(Level &fine) {
    const auto csize = domain::get_size(m_coarse_domain);
    each_owned(fine.box, [&](int i, int j, int k) {
      const int fi = fine.box.low[0] + i, fj = fine.box.low[1] + j,
                fk = fine.box.low[2] + k;
      fine.x(i, j, k) += interpolate(fi, fj, fk, [&](int ci, int cj, int ck) {
        if (ci < 0 || cj < 0 || ck < 0 ||
            ci >= csize[0] || cj >= csize[1] || ck >= csize[2]) return 0.0;
        return m_coarse_x[linear(csize, ci, cj, ck)];
      });
    });
  }

  void factor_coarse(const std::array<double, 27> &weights) {
    const auto n = domain::get_size(m_coarse_domain);
    const std::size_t N = m_coarse_rhs.size();
    m_cholesky.assign(N * N, 0.0);
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const std::size_t row = linear(n, i, j, k);
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const int ci = i + dx, cj = j + dy, ck = k + dz;
                if (ci >= 0 && cj >= 0 && ck >= 0 &&
                    ci < n[0] && cj < n[1] && ck < n[2])
                  m_cholesky[row * N + linear(n, ci, cj, ck)] +=
                      weights[nmg::offset_index(dx, dy, dz)];
              }
        }
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j <= i; ++j) {
        double entry = m_cholesky[i * N + j];
        for (std::size_t k = 0; k < j; ++k)
          entry -= m_cholesky[i * N + k] * m_cholesky[j * N + k];
        if (i == j) {
          if (!(entry > 0.0) || !std::isfinite(entry))
            throw std::runtime_error("NodalDirichletMultigrid: coarse matrix not SPD");
          m_cholesky[i * N + i] = std::sqrt(entry);
        } else {
          m_cholesky[i * N + j] = entry / m_cholesky[j * N + j];
        }
      }
  }

  void solve_coarse() {
    const std::size_t N = m_coarse_rhs.size();
    for (std::size_t i = 0; i < N; ++i) {
      double x = m_coarse_rhs[i];
      for (std::size_t j = 0; j < i; ++j)
        x -= m_cholesky[i * N + j] * m_coarse_x[j];
      m_coarse_x[i] = x / m_cholesky[i * N + i];
    }
    for (std::size_t i = N; i-- > 0;) {
      double x = m_coarse_x[i];
      for (std::size_t j = i + 1; j < N; ++j)
        x -= m_cholesky[j * N + i] * m_coarse_x[j];
      m_coarse_x[i] = x / m_cholesky[i * N + i];
    }
  }

  void cycle(std::size_t l) {
    Level &fine = *m_levels[l];
    smooth(fine, m_controls.pre);
    action(fine, fine.x, fine.ax);
    each_owned(fine.box, [&](int i, int j, int k) {
      fine.residual(i, j, k) = fine.rhs(i, j, k) - fine.ax(i, j, k);
    });
    if (l + 1 < m_levels.size()) {
      Level &coarse = *m_levels[l + 1];
      restrict_distributed(fine, coarse);
      std::fill(coarse.x.vec().begin(), coarse.x.vec().end(), 0.0);
      cycle(l + 1);
      prolong_distributed(fine, coarse);
    } else {
      restrict_coarse(fine);
      solve_coarse();
      prolong_coarse(fine);
    }
    smooth(fine, m_controls.post);
  }

  MPI_Comm m_comm;
  int m_rank, m_size = 1;
  Controls m_controls;
  std::vector<std::unique_ptr<Level>> m_levels;
  Domain m_coarse_domain;
  std::vector<double> m_coarse_rhs, m_coarse_x, m_cholesky;
};

} // namespace pfc::solvers
