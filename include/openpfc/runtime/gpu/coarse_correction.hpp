// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/runtime/gpu/multigrid.hpp>
#include <openpfc/solvers/coarse_correction.hpp>
#include <optional>
namespace pfc::solvers::gpu {
/// Device-resident shared coarse correction. The coarse unknown and the
/// Krylov vectors stay in device buffers. Each application stages only the
/// small coarse residual for the existing MPI reduction, then discards that
/// temporary. No host coarse unknown is retained.
template <class Tag> class SharedCoarseCorrection {
public:
  using Vector = typename VectorOperations<Tag>::Vector;
  using Controls = solvers::CoarseControls;
  double communication_seconds = 0;
  std::size_t collective_bytes = 0;
  int applications = 0;

  SharedCoarseCorrection(solvers::StructuredGrid fine, std::vector<int> owned,
                         solvers::Multigrid::Factory factory, MPI_Comm comm,
                         Controls controls = Controls())
      : ops_(comm), comm_(comm), components_(fine.components), nodes_(fine.nodes()),
        periodic_(fine.boundary == solvers::Boundary::Periodic) {
    solvers::SharedCoarseCorrection host(fine, owned, factory, comm, controls);
    coarse_count_ = host.coarse().size();
    build(host.rows(), static_cast<int>(owned.size()), coarse_count_);
    solvers::MultigridControls hierarchy;
    hierarchy.coarse_nodes =
        *std::max_element(host.coarse().shape.begin(), host.coarse().shape.end());
    auto coarse_operator = host.coarse_operator();
    auto grid = host.coarse();
    solver_.emplace(
        grid,
        [coarse_operator](const solvers::StructuredGrid &) {
          return coarse_operator;
        },
        hierarchy);
    coarse_residual_ = ops_.make(coarse_count_);
    coarse_update_ = ops_.make(coarse_count_);
    if (periodic_) {
      for (int component = 0; component < components_; ++component) {
        std::vector<double> mask(owned.size());
        for (std::size_t i = 0; i < owned.size(); ++i)
          mask[i] = owned[i] % components_ == component ? 1 : 0;
        masks_.push_back(ops_.make(mask.size()));
        masks_.back().copy_from_host(mask);
      }
    }
  }
  std::size_t retained_host_coarse_bytes() const { return 0; }
  int coarse_dofs() const { return coarse_count_; }
  void operator()(const Vector &r, Vector &z, int = 0) {
    int valid = static_cast<int>(r.size()) == prolongation_.rows() ? 1 : 0,
        all = valid;
    if (comm_ != MPI_COMM_NULL &&
        MPI_Allreduce(&valid, &all, 1, MPI_INT, MPI_MIN, comm_) != MPI_SUCCESS)
      throw std::runtime_error("coarse ownership check failed");
    if (!all) throw std::invalid_argument("owned residual size mismatch");
    restriction_(r, coarse_residual_);
    std::vector<double> host(coarse_count_), reduced(coarse_count_);
    auto start = std::chrono::steady_clock::now();
    coarse_residual_.copy_to_host(host.data(), host.size());
    if (comm_ != MPI_COMM_NULL) {
      if (MPI_Allreduce(host.data(), reduced.data(), coarse_count_, MPI_DOUBLE,
                        MPI_SUM, comm_) != MPI_SUCCESS)
        throw std::runtime_error("coarse residual reduction failed");
      host.swap(reduced);
    }
    coarse_residual_.copy_from_host(host);
    collective_bytes += 2 * static_cast<std::size_t>(coarse_count_) * sizeof(double);
    communication_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    (*solver_)(coarse_residual_, coarse_update_);
    prolongation_(coarse_update_, z);
    if (periodic_)
      for (int component = 0; component < components_; ++component)
        ops_.axpy(z, -ops_.dot(z, masks_[component]) / nodes_, masks_[component]);
    ++applications;
  }

private:
  VectorOperations<Tag> ops_;
  MPI_Comm comm_;
  int components_, nodes_, coarse_count_ = 0;
  bool periodic_;
  SparseOperator<Tag> prolongation_{0, 0, {0}, {}, {}};
  SparseOperator<Tag> restriction_{0, 0, {0}, {}, {}};
  std::optional<Multigrid<Tag>> solver_;
  Vector coarse_residual_{0}, coarse_update_{0};
  std::vector<Vector> masks_;
  static SparseOperator<Tag>
  csr(int rows, int columns,
      const std::vector<std::vector<std::pair<int, double>>> &entries) {
    std::vector<int> offsets{0}, indices;
    std::vector<double> values;
    for (const auto &row : entries) {
      for (auto [column, weight] : row) {
        indices.push_back(column);
        values.push_back(weight);
      }
      offsets.push_back(static_cast<int>(indices.size()));
    }
    return SparseOperator<Tag>(rows, columns, offsets, indices, values);
  }
  void build(const std::vector<std::vector<std::pair<int, double>>> &rows, int owned,
             int coarse) {
    prolongation_ = csr(owned, coarse, rows);
    std::vector<std::vector<std::pair<int, double>>> columns(coarse);
    for (int row = 0; row < owned; ++row)
      for (auto [column, weight] : rows[row])
        columns[column].push_back({row, weight});
    restriction_ = csr(coarse, owned, columns);
  }
};
} // namespace pfc::solvers::gpu
