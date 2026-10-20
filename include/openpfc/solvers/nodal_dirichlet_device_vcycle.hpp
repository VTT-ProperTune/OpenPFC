// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file nodal_dirichlet_device_vcycle.hpp
 * @brief Arbitrary-depth nested Dirichlet Q1 GPU V-cycle using OpenPFC fields.
 *
 * Caller-provided nested OpenPFC Decompositions, never another partitioner.
 * Every fine-grid operation is device resident and calls OpenPFC FieldOps,
 * FieldHalo and the shared Q1 CUDA/HIP kernels. Only the capacity-bounded
 * replicated coarse free-node RHS and correction cross the host boundary.
 */

#include <openpfc/solvers/nodal_dirichlet_coarse.hpp>
#include <openpfc/solvers/nodal_dirichlet_device_transfers.hpp>
#include <openpfc/solvers/nodal_dirichlet_device_operator.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

namespace pfc::solvers {

/**
 * Fixed SPD preconditioner over an OpenPFC-owned nested Q1 hierarchy.
 *
 * This is a deliberately bounded validation gate: the coarsest free-node
 * system has at most 1024 unknowns, replicated and factored with the SAME
 * DenseCoarseCholesky used by the CPU Dirichlet GMG. Global fine/coarse
 * partitions must already have been created by OpenPFC Decomposition.
 *
 * Fine/intermediate action, damping, residual, transfers and Krylov vectors
 * remain CUDA/HIP resident. A recursive instance on the next nested level
 * invokes the identical numerical V-cycle rather than cloning a solver.
 * Only the tiny terminal coarse problem is reduced and solved on the host.
 * Hardware numerical qualification remains distinct from compile-only CI.
 */
template <class Space> class NodalDirichletDeviceVcycle {
public:
  using FieldT = data::Field<double, Space>;
  using Ops = FieldOperations<Space>;
  using Vector = typename Ops::Vector;
  using Halo = FieldHalo<Space>;
  using Launch = detail::DirichletDevicePointLaunch<Space>;

  struct Controls {
    double diffusion = 1.0;
    double reaction = 0.0;
    double jacobi_weight = 0.65;
    int pre = 2;
    int post = 2;
    std::size_t max_coarse_unknowns = 1024;
  };

  using Hierarchy = std::vector<decomposition::Decomposition>;

  /// Backwards-compatible two-level Q1 constructor.
  NodalDirichletDeviceVcycle(
      const decomposition::Decomposition &fine,
      const decomposition::Decomposition &coarse, int rank, MPI_Comm comm)
      : NodalDirichletDeviceVcycle(
            Hierarchy{fine, coarse}, rank, comm, Controls{}) {}

  NodalDirichletDeviceVcycle(
      const decomposition::Decomposition &fine,
      const decomposition::Decomposition &coarse, int rank, MPI_Comm comm,
      Controls controls)
      : NodalDirichletDeviceVcycle(
            Hierarchy{fine, coarse}, rank, comm, controls) {}

  /// Backwards-compatible three-level constructor.
  NodalDirichletDeviceVcycle(
      const decomposition::Decomposition &fine,
      const decomposition::Decomposition &middle,
      const decomposition::Decomposition &coarse, int rank, MPI_Comm comm)
      : NodalDirichletDeviceVcycle(
            Hierarchy{fine, middle, coarse}, rank, comm, Controls{}) {}

  NodalDirichletDeviceVcycle(
      const decomposition::Decomposition &fine,
      const decomposition::Decomposition &middle,
      const decomposition::Decomposition &coarse, int rank, MPI_Comm comm,
      Controls controls)
      : NodalDirichletDeviceVcycle(
            Hierarchy{fine, middle, coarse}, rank, comm, controls) {}

  /// Any nested level count >=2. Levels are OpenPFC Decompositions ordered
  /// finest to coarsest; callers may supply nontrivial MPI brick grids.
  NodalDirichletDeviceVcycle(const Hierarchy &levels, int rank, MPI_Comm comm)
      : NodalDirichletDeviceVcycle(levels, rank, comm, Controls{}) {}

  NodalDirichletDeviceVcycle(const Hierarchy &levels, int rank, MPI_Comm comm,
                             Controls controls)
      : NodalDirichletDeviceVcycle(levels, 0, rank, comm, controls) {}

  /// Reuse OpenPFC's host-testable geometry factory: no device-specific
  /// splitter and no rank-local partition policy is defined in the solver.
  [[nodiscard]] static Hierarchy
  make_hierarchy(const decomposition::Decomposition &fine) {
    return nmg::dirichlet::make_nested_decompositions(fine);
  }

private:
  static const decomposition::Decomposition &at(const Hierarchy &levels,
                                                 std::size_t index) {
    if (index >= levels.size())
      throw std::invalid_argument("DirichletDeviceVcycle: missing nested grid level");
    return levels[index];
  }

  NodalDirichletDeviceVcycle(
      const Hierarchy &levels, std::size_t index, int rank, MPI_Comm comm,
      Controls controls)
      : m_fd(decomposition::domain(at(levels, index))),
        m_cd(decomposition::domain(at(levels, index + 1))),
        m_fdec(at(levels, index)), m_cdec(at(levels, index + 1)),
        m_fb(validated_box(levels, index, rank, comm, controls)),
        m_cb(decomposition::local_box(m_cdec, rank)),
        m_comm(comm), m_controls(controls),
        m_fops(m_fd, m_fb, 1, nmg::dirichlet::kHalo, comm),
        m_cops(m_cd, m_cb, 1, nmg::dirichlet::kHalo, comm),
        m_halo(m_fops, m_fdec, rank, comm, full_halo(9200)),
        m_operator(m_fdec, rank, comm,
                   controls.diffusion, controls.reaction),
        m_transfers(m_fdec, m_cdec, rank, comm),
        m_rhs(m_fops.make(m_fops.size())),
        m_x(m_fops.make(m_fops.size())),
        m_next(m_fops.make(m_fops.size())),
        m_ax(m_fops.make(m_fops.size())),
        m_residual(m_fops.make(m_fops.size())),
        m_crhs(m_cops.make(m_cops.size())),
        m_cx(m_cops.make(m_cops.size())) {
    const auto spacing = domain::get_spacing(m_fd);
    const double h[3] = {spacing[0], spacing[1], spacing[2]};
    m_weights = nmg::dirichlet::make_weights(
        h, controls.diffusion, controls.reaction);

    const auto n = domain::get_size(m_fd);
    for (int d = 0; d < 3; ++d) {
      m_shape.n[d] = m_fb.size[d];
      m_shape.low[d] = m_fb.low[d];
      m_shape.gn[d] = n[d];
    }

    const double diag = m_weights.v[13];
    double row_bound = 0.0;
    for (double a : m_weights.v) row_bound += std::abs(a);
    if (!(diag > 0.0) || !(row_bound >= diag))
      throw std::logic_error("DirichletDeviceVcycle: invalid Q1 Jacobi row");
    m_jacobi = std::min(controls.jacobi_weight,
                         1.5 * diag / row_bound) / diag;

    if (index + 2 < levels.size()) {
      // The same GPU hierarchy on the next level. All fine/intermediate
      // Fields remain resident; only the leaf reduces a bounded coarse RHS.
      m_child.reset(new NodalDirichletDeviceVcycle(
          levels, index + 1, rank, comm, controls));
    } else {
      const auto nc = domain::get_size(m_cd);
      const auto hc = domain::get_spacing(m_cd);
      const double coarse_spacing[3] = {hc[0], hc[1], hc[2]};
      m_factor = nmg::dirichlet::DenseCoarseCholesky(
          nc, nmg::dirichlet::make_weights(
                  coarse_spacing, controls.diffusion, controls.reaction),
          controls.max_coarse_unknowns);
      m_local_coarse.assign(m_factor.unknowns(), 0.0);
      m_global_coarse.assign(m_factor.unknowns(), 0.0);
      m_coarse_solution.assign(m_factor.unknowns(), 0.0);
    }
  }

public:
  NodalDirichletDeviceVcycle(
      const NodalDirichletDeviceVcycle &) = delete;
  NodalDirichletDeviceVcycle &operator=(
      const NodalDirichletDeviceVcycle &) = delete;

  [[nodiscard]] std::size_t coarse_unknowns() const noexcept {
    return m_child ? m_child->coarse_unknowns() : m_factor.unknowns();
  }
  [[nodiscard]] int levels() const noexcept {
    return m_child ? 1 + m_child->levels() : 2;
  }
  [[nodiscard]] const Box3i &owned_box() const noexcept { return m_fb; }

  /// Apply a zero-initialized symmetric nested V-cycle. Input and
  /// output may alias, and either may have a different nonnegative halo.
  /// All ranks of m_comm must participate in the same call order.
  void apply(const FieldT &rhs, FieldT &out) {
    validate_field(rhs, m_fd, m_fb, "RHS");
    validate_field(out, m_fd, m_fb, "output");
    const_cast<FieldT &>(rhs).sync_to_device();
    Launch::owned_copy(m_shape, rhs.data(), rhs.storage_halo(),
                       m_rhs[0].data(), nmg::dirichlet::kHalo);
    m_rhs[0].note_device_write();

    // Fresh zero initial guess on every call: required for a fixed linear
    // symmetric preconditioner usable by standard (non-flexible) PCG.
    m_fops.scale(m_x, 0.0);
    smooth(m_controls.pre);

    m_operator.apply_operator(m_x[0], m_ax[0]);
    m_fops.copy(m_rhs, m_residual);
    m_fops.axpy(m_residual, -1.0, m_ax);

    m_transfers.restrict_pt(m_residual[0], m_crhs[0]);
    solve_coarse();
    m_transfers.prolong_add(m_cx[0], m_x[0]);

    smooth(m_controls.post);
    out.sync_to_device();
    Launch::owned_copy(m_shape, m_x[0].data(),
                       nmg::dirichlet::kHalo, out.data(),
                       out.storage_halo());
    out.note_device_write();
  }

  void operator()(const FieldT &rhs, FieldT &out, int = 0) {
    apply(rhs, out);
  }

private:
  static comm::HaloExchangeOptions full_halo(int tag) {
    comm::HaloExchangeOptions opts;
    opts.connectivity = comm::HaloConnectivity::Full;
    opts.exchange_base = tag;
    return opts;
  }

  static Box3i validated_box(const Hierarchy &levels, std::size_t index,
                             int rank, MPI_Comm comm, const Controls &c) {
    if (comm == MPI_COMM_NULL)
      throw std::invalid_argument("DirichletDeviceVcycle: null MPI communicator");
    int actual = -1, size = 0;
    MPI_Comm_rank(comm, &actual);
    MPI_Comm_size(comm, &size);
    if (actual != rank || levels.size() < 2 || index + 1 >= levels.size())
      throw std::invalid_argument("DirichletDeviceVcycle: invalid MPI level/index");
    if (!(c.diffusion > 0.0) || !std::isfinite(c.diffusion) ||
        !(c.reaction >= 0.0) || !std::isfinite(c.reaction) ||
        !(c.jacobi_weight > 0.0 && c.jacobi_weight < 1.0) ||
        !std::isfinite(c.jacobi_weight) || c.pre < 1 ||
        c.post != c.pre || c.max_coarse_unknowns < 1 ||
        c.max_coarse_unknowns > 1024)
      throw std::invalid_argument("DirichletDeviceVcycle: unsupported controls");

    // All ranks inspect the SAME hierarchy BEFORE any rank creates a
    // device HaloExchange. Fail closed on an incompatible intermediate
    // partition, rather than hanging one rank inside a collective.
    for (std::size_t k = 0; k + 1 < levels.size(); ++k) {
      if (decomposition::get_num_domains(levels[k]) != size ||
          decomposition::get_num_domains(levels[k + 1]) != size)
        throw std::invalid_argument(
            "DirichletDeviceVcycle: MPI rank count differs among levels");
      nmg::dirichlet::validate_nested_transfer_reach(
          levels[k], levels[k + 1]);
    }
    const auto leaf = domain::get_size(decomposition::domain(levels.back()));
    if (!(leaf == Int3{7, 7, 7} || leaf == Int3{7, 7, 15}))
      throw std::invalid_argument(
          "DirichletDeviceVcycle: expected 7x7x7 or 7x7x15 terminal free nodes");
    const std::size_t N = static_cast<std::size_t>(leaf[0]) * leaf[1] * leaf[2];
    if (N > c.max_coarse_unknowns)
      throw std::invalid_argument(
          "DirichletDeviceVcycle: leaf coarse factor exceeds capacity");
    return decomposition::local_box(levels[index], rank);
  }

  static void validate_field(const FieldT &v, const Domain &d,
                             const Box3i &box, const char *name) {
    if (v.domain() != d || v.box().low != box.low ||
        v.box().size != box.size || v.storage_halo() < 0)
      throw std::invalid_argument(
          std::string("DirichletDeviceVcycle: bad ") + name + " field");
  }

  void smooth(int sweeps) {
    for (int pass = 0; pass < sweeps; ++pass) {
      const auto &exchanged = m_halo(m_x);
      Launch::jacobi(m_shape, m_weights, exchanged[0].data(),
                     m_rhs[0].data(), m_next[0].data(), m_jacobi);
      m_next[0].note_device_write();
      std::swap(m_x, m_next);
    }
  }

  static std::size_t linear(const Int3 &n, int i, int j, int k) {
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(n[0]) *
               (static_cast<std::size_t>(j) +
                static_cast<std::size_t>(n[1]) * k);
  }

  void solve_coarse() {
    if (m_child) {
      // The middle-level correction is computed wholly on the device.
      // Only the terminal 7x7x7 or 7x7x15 solve inside m_child sees host.
      m_child->apply(m_crhs[0], m_cx[0]);
      return;
    }
    const auto global = domain::get_size(m_cd);
    std::fill(m_local_coarse.begin(), m_local_coarse.end(), 0.0);
    // Only the <=1024 coarse coefficients are downloaded; no fine-grid
    // transfer or replicated intermediate level is possible through here.
    m_crhs[0].with_host_read([&](const double *p, std::size_t) {
      for (int k = 0; k < m_cb.size[2]; ++k)
        for (int j = 0; j < m_cb.size[1]; ++j)
          for (int i = 0; i < m_cb.size[0]; ++i)
            m_local_coarse[linear(global, m_cb.low[0] + i,
                                 m_cb.low[1] + j, m_cb.low[2] + k)] =
                p[m_crhs[0].idx(i, j, k)];
    });
    if (MPI_Allreduce(m_local_coarse.data(), m_global_coarse.data(),
                      static_cast<int>(m_factor.unknowns()), MPI_DOUBLE,
                      MPI_SUM, m_comm) != MPI_SUCCESS)
      throw std::runtime_error("DirichletDeviceVcycle: coarse MPI reduction failed");
    m_factor.solve(m_global_coarse, m_coarse_solution);

    m_cx[0].with_host_view([&](double *p, std::size_t) {
      for (int k = 0; k < m_cb.size[2]; ++k)
        for (int j = 0; j < m_cb.size[1]; ++j)
          for (int i = 0; i < m_cb.size[0]; ++i)
            p[m_cx[0].idx(i, j, k)] =
                m_coarse_solution[linear(global, m_cb.low[0] + i,
                                         m_cb.low[1] + j, m_cb.low[2] + k)];
    });
    m_cx[0].sync_to_device();
  }

  Domain m_fd, m_cd;
  decomposition::Decomposition m_fdec, m_cdec;
  Box3i m_fb, m_cb;
  MPI_Comm m_comm;
  Controls m_controls;
  Ops m_fops, m_cops;
  Halo m_halo;
  NodalDirichletDeviceOperator<Space> m_operator;
  NodalDirichletDeviceTransfers<Space> m_transfers;
  Vector m_rhs, m_x, m_next, m_ax, m_residual, m_crhs, m_cx;
  nmg::dirichlet::DenseCoarseCholesky m_factor;
  std::unique_ptr<NodalDirichletDeviceVcycle> m_child;
  nmg::Level m_shape{};
  nmg::dirichlet::Weights m_weights{};
  double m_jacobi = 0.0;
  std::vector<double> m_local_coarse, m_global_coarse, m_coarse_solution;
};

/// Preserve the original qualified two-level public source API.
template <class Space>
using NodalDirichletDeviceTwoLevelVcycle = NodalDirichletDeviceVcycle<Space>;

} // namespace pfc::solvers

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
