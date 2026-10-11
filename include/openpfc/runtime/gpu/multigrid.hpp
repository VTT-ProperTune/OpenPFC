// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/runtime/gpu/vector_operations.hpp>
#include <openpfc/solvers/multigrid.hpp>
namespace pfc::solvers::gpu {
namespace detail {
struct CsrView {
  int rows;
  const int *offsets, *columns;
  const double *values;
};
#if defined(OpenPFC_ENABLE_CUDA)
void sparse_cuda(CsrView, const double *, double *);
void dense_cuda(const double *, const double *, double *, int);
void smooth_cuda(double *, const double *, const double *, const double *, double,
                 int);
void boundary_cuda(double *, StructuredGrid);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void sparse_hip(CsrView, const double *, double *);
void dense_hip(const double *, const double *, double *, int);
void smooth_hip(double *, const double *, const double *, const double *, double,
                int);
void boundary_hip(double *, StructuredGrid);
#endif
} // namespace detail
/// Resident CSR adapter, including rectangular transfer operators.
template <class Tag> class SparseOperator {
  pfc::core::DataBuffer<Tag, int> offsets_, columns_;
  pfc::core::DataBuffer<Tag, double> values_;
  int rows_, columns_count_;

public:
  using Vector = typename VectorOperations<Tag>::Vector;
  Vector diagonal;
  explicit SparseOperator(const solvers::SparseOperator &a)
      : SparseOperator(a.size(), a.size(), a.offsets, a.columns, a.values,
                       a.diagonal) {}
  SparseOperator(int rows, int columns, const std::vector<int> &offsets,
                 const std::vector<int> &indices, const std::vector<double> &values,
                 const std::vector<double> &diag = {})
      : offsets_(offsets.size()), columns_(indices.size()), values_(values.size()),
        rows_(rows), columns_count_(columns), diagonal(diag.size()) {
    if (rows < 0 || columns < 0 || offsets.size() != std::size_t(rows + 1) ||
        offsets.front() != 0 || offsets.back() != int(indices.size()) ||
        indices.size() != values.size())
      throw std::invalid_argument("invalid sparse device adapter");
    for (std::size_t i = 1; i < offsets.size(); ++i)
      if (offsets[i] < offsets[i - 1] || offsets[i] > int(indices.size()))
        throw std::invalid_argument("invalid sparse row offsets");
    if (!diag.empty() && diag.size() != std::size_t(rows))
      throw std::invalid_argument("invalid sparse diagonal size");
    for (int j : indices)
      if (j < 0 || j >= columns)
        throw std::invalid_argument("sparse column out of range");
    offsets_.copy_from_host(offsets);
    columns_.copy_from_host(indices);
    values_.copy_from_host(values);
    if (!diag.empty()) diagonal.copy_from_host(diag);
  }
  int rows() const { return rows_; }
  int columns() const { return columns_count_; }
  void operator()(const Vector &x, Vector &y) const {
    if (x.size() != std::size_t(columns_count_))
      throw std::invalid_argument("sparse vector size mismatch");
    if (x.data() == y.data())
      throw std::invalid_argument("sparse action forbids input alias");
    if (y.size() != std::size_t(rows_)) y.resize(rows_);
    detail::CsrView view{rows_, offsets_.data(), columns_.data(), values_.data()};
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::sparse_cuda(view, x.data(), y.data());
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::sparse_hip(view, x.data(), y.data());
#endif
  }
  std::size_t storage_bytes() const {
    return (offsets_.size() + columns_.size()) * sizeof(int) +
           (values_.size() + diagonal.size()) * sizeof(double);
  }
};
/// One-device V-cycle. Build the verified host hierarchy once, then upload its
/// operators and transfers. No field leaves the device during application.
/// Host coarse factorization is a setup cost: the device applies its fixed
/// projected inverse. This hierarchy is local; MPI scaling needs a distributed
/// level adapter and must not be inferred from a replicated test.
template <class Tag> class Multigrid {
public:
  using Vector = typename VectorOperations<Tag>::Vector;

private:
  struct Level {
    StructuredGrid grid;
    SparseOperator<Tag> action;
    Vector b, x, ax, r, correction;
    Level(StructuredGrid g, const solvers::SparseOperator &a,
          VectorOperations<Tag> &ops)
        : grid(g), action(a), b(ops.make(g.size())), x(ops.make(g.size())),
          ax(ops.make(g.size())), r(ops.make(g.size())),
          correction(ops.make(g.size())) {}
  };
  struct Transfer {
    SparseOperator<Tag> prolongation, restriction;
    static SparseOperator<Tag> make(const GridTransfer &t, bool transpose) {
      int rows = transpose ? t.coarse.size() : t.fine.size(),
          cols = transpose ? t.fine.size() : t.coarse.size();
      std::vector<std::vector<std::pair<int, double>>> entries(rows);
      for (int i = 0; i < t.fine.size(); ++i)
        for (auto [j, w] : t.weights[i])
          entries[transpose ? j : i].push_back({transpose ? i : j, w});
      std::vector<int> offsets{0}, indices;
      std::vector<double> values;
      for (auto &row : entries) {
        for (auto [j, w] : row) {
          indices.push_back(j);
          values.push_back(w);
        }
        offsets.push_back(indices.size());
      }
      return SparseOperator<Tag>(rows, cols, offsets, indices, values);
    }
    explicit Transfer(const GridTransfer &t)
        : prolongation(make(t, false)), restriction(make(t, true)) {}
  };
  VectorOperations<Tag> ops_;
  std::vector<Level> levels_;
  std::vector<Transfer> transfers_;
  MultigridControls controls_;
  Vector inverse_;
  void project(Vector &v, StructuredGrid grid) {
    if (grid.boundary == Boundary::Periodic) {
      ops_.project(v, grid.components);
      return;
    }
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
      return detail::boundary_cuda(v.data(), grid);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
      return detail::boundary_hip(v.data(), grid);
#endif
  }
  void smooth(Level &l, int sweeps) {
    for (int i = 0; i < sweeps; ++i) {
      l.action(l.x, l.ax);
#if defined(OpenPFC_ENABLE_CUDA)
      if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
        detail::smooth_cuda(l.x.data(), l.b.data(), l.ax.data(),
                            l.action.diagonal.data(), controls_.jacobi_weight,
                            l.grid.size());
#endif
#if defined(OpenPFC_ENABLE_HIP)
      if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
        detail::smooth_hip(l.x.data(), l.b.data(), l.ax.data(),
                           l.action.diagonal.data(), controls_.jacobi_weight,
                           l.grid.size());
#endif
      project(l.x, l.grid);
    }
  }
  void cycle(int k) {
    auto &l = levels_[k];
    if (k + 1 == int(levels_.size())) {
#if defined(OpenPFC_ENABLE_CUDA)
      if constexpr (std::is_same_v<Tag, pfc::backend::CUDATag>)
        detail::dense_cuda(inverse_.data(), l.b.data(), l.x.data(), l.grid.size());
#endif
#if defined(OpenPFC_ENABLE_HIP)
      if constexpr (std::is_same_v<Tag, pfc::backend::HIPTag>)
        detail::dense_hip(inverse_.data(), l.b.data(), l.x.data(), l.grid.size());
#endif
      project(l.x, l.grid);
      return;
    }
    smooth(l, controls_.pre_smooth);
    l.action(l.x, l.ax);
    ops_.copy(l.b, l.r);
    ops_.axpy(l.r, -1, l.ax);
    project(l.r, l.grid);
    auto &coarse = levels_[k + 1];
    transfers_[k].restriction(l.r, coarse.b);
    project(coarse.b, coarse.grid);
    ops_.scale(coarse.x, 0);
    cycle(k + 1);
    transfers_[k].prolongation(coarse.x, l.correction);
    ops_.axpy(l.x, 1, l.correction);
    smooth(l, controls_.post_smooth);
  }

public:
  Multigrid(StructuredGrid grid, solvers::Multigrid::Factory factory,
            MultigridControls controls = {})
      : controls_(controls) {
    solvers::Multigrid host(grid, factory, controls);
    for (int k = 0; k < host.levels(); ++k) {
      if (k) grid = host.transfer_at(k - 1).coarse;
      levels_.emplace_back(grid, host.operator_at(k), ops_);
      if (k + 1 < host.levels()) transfers_.emplace_back(host.transfer_at(k));
    }
    const auto &a = host.operator_at(host.levels() - 1);
    // Reuse the existing factored coarse solve, including its nullspace contract.
    MultigridControls coarse_controls = controls;
    coarse_controls.coarse_nodes =
        *std::max_element(grid.shape.begin(), grid.shape.end());
    solvers::Multigrid coarse(
        grid, [&](const StructuredGrid &) { return a; }, coarse_controls);
    int n = grid.size();
    std::vector<double> inverse(n * n), b(n), x;
    for (int j = 0; j < n; ++j) {
      std::fill(b.begin(), b.end(), 0);
      b[j] = 1;
      coarse(b, x);
      for (int i = 0; i < n; ++i) inverse[i * n + j] = x[i];
    }
    inverse_ = Vector(n * n);
    inverse_.copy_from_host(inverse);
  }
  void operator()(const Vector &r, Vector &z, int = 0) {
    auto &l = levels_.front();
    ops_.copy(r, l.b);
    project(l.b, l.grid);
    ops_.scale(l.x, 0);
    cycle(0);
    ops_.copy(l.x, z);
  }
  const SparseOperator<Tag> &operator_at(int level) const {
    return levels_.at(level).action;
  }
  std::size_t storage_bytes() const {
    std::size_t n = inverse_.size() * sizeof(double);
    for (auto &l : levels_)
      n += l.action.storage_bytes() + 5 * l.grid.size() * sizeof(double);
    for (auto &t : transfers_)
      n += t.prolongation.storage_bytes() + t.restriction.storage_bytes();
    return n;
  }
};
} // namespace pfc::solvers::gpu
