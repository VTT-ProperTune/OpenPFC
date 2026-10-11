// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace pfc::solvers {
enum class Boundary { Periodic, Dirichlet };
enum class CoarseOperator { Rediscretized, Galerkin };
struct StructuredGrid {
  std::array<int, 3> shape;
  int components = 1;
  Boundary boundary = Boundary::Periodic;
  int nodes() const { return shape[0] * shape[1] * shape[2]; }
  int size() const { return components * nodes(); }
  int node(int x, int y, int z) const { return x + shape[0] * (y + shape[1] * z); }
  bool constrained(int x, int y, int z) const {
    return boundary == Boundary::Dirichlet &&
           (x == 0 || y == 0 || z == 0 || x == shape[0] - 1 || y == shape[1] - 1 ||
            z == shape[2] - 1);
  }
  void validate() const {
    long long n = components;
    if (components < 1) throw std::invalid_argument("grid needs components");
    for (int a : shape) {
      if (a < 3)
        throw std::invalid_argument("grid needs at least three nodes per axis");
      if (n > std::numeric_limits<int>::max() / a)
        throw std::invalid_argument("structured grid exceeds index capacity");
      n *= a;
    }
    if (n > std::numeric_limits<int>::max())
      throw std::invalid_argument("structured grid exceeds index capacity");
  }
  void project(std::vector<double> &v) const {
    if (v.size() != static_cast<std::size_t>(size()))
      throw std::invalid_argument("grid vector mismatch");
    if (boundary == Boundary::Periodic) {
      std::vector<double> mean(components);
      for (int i = 0; i < size(); ++i) mean[i % components] += v[i] / nodes();
      for (int i = 0; i < size(); ++i) v[i] -= mean[i % components];
    } else {
      for (int z = 0; z < shape[2]; ++z)
        for (int y = 0; y < shape[1]; ++y)
          for (int x = 0; x < shape[0]; ++x)
            if (constrained(x, y, z))
              for (int c = 0; c < components; ++c)
                v[components * node(x, y, z) + c] = 0;
    }
  }
};
/// Compact host sparse operator. Experimental physics never enters this type.
struct SparseOperator {
  std::vector<int> offsets, columns;
  std::vector<double> values, diagonal;
  int size() const { return diagonal.size(); }
  void operator()(const std::vector<double> &x, std::vector<double> &y) const {
    if (x.size() != diagonal.size())
      throw std::invalid_argument("sparse action size mismatch");
    y.assign(size(), 0);
    for (int i = 0; i < size(); ++i)
      for (int p = offsets[i]; p < offsets[i + 1]; ++p)
        y[i] += values[p] * x[columns[p]];
  }
  static SparseOperator from_rows(const std::vector<std::map<int, double>> &rows) {
    SparseOperator out;
    out.offsets = {0};
    out.diagonal.resize(rows.size());
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
      for (auto [j, v] : rows[i])
        if (v != 0) {
          out.columns.push_back(j);
          out.values.push_back(v);
          if (i == j) out.diagonal[i] = v;
        }
      out.offsets.push_back(out.values.size());
    }
    return out;
  }
};
/** Trilinear interpolation in fixed physical coordinates; odd periodic grids
 * need not be nested. Restriction is exactly P^T, appropriate for weak-form
 * (volume-weighted) operators. Dirichlet boundary values remain zero.
 */
class GridTransfer {
public:
  StructuredGrid fine, coarse;
  std::vector<std::vector<std::pair<int, double>>> weights;
  GridTransfer(StructuredGrid f, StructuredGrid c)
      : fine(f), coarse(c), weights(f.size()) {
    f.validate();
    c.validate();
    if (f.components != c.components || f.boundary != c.boundary)
      throw std::invalid_argument("incompatible transfer grids");
    for (int z = 0; z < f.shape[2]; ++z)
      for (int y = 0; y < f.shape[1]; ++y)
        for (int x = 0; x < f.shape[0]; ++x) {
          if (f.constrained(x, y, z)) continue;
          int index[3] = {x, y, z}, low[3];
          double fraction[3];
          for (int d = 0; d < 3; ++d) {
            double t = static_cast<double>(index[d]) *
                       (c.shape[d] - (c.boundary == Boundary::Dirichlet)) /
                       (f.shape[d] - (f.boundary == Boundary::Dirichlet));
            low[d] = static_cast<int>(std::floor(t));
            fraction[d] = t - low[d];
          }
          for (int mask = 0; mask < 8; ++mask) {
            int q[3];
            double w = 1;
            for (int d = 0; d < 3; ++d) {
              int bit = (mask >> d) & 1;
              q[d] = (low[d] + bit) % c.shape[d];
              w *= bit ? fraction[d] : 1 - fraction[d];
            }
            if (c.constrained(q[0], q[1], q[2]) || w == 0) continue;
            for (int component = 0; component < f.components; ++component)
              weights[f.components * f.node(x, y, z) + component].push_back(
                  {c.components * c.node(q[0], q[1], q[2]) + component, w});
          }
        }
  }
  void prolong(const std::vector<double> &v, std::vector<double> &out) const {
    if (v.size() != static_cast<std::size_t>(coarse.size()))
      throw std::invalid_argument("prolongation size mismatch");
    out.assign(fine.size(), 0);
    for (int i = 0; i < fine.size(); ++i)
      for (auto [j, w] : weights[i]) out[i] += w * v[j];
  }
  void restrict(const std::vector<double> &v, std::vector<double> &out) const {
    if (v.size() != static_cast<std::size_t>(fine.size()))
      throw std::invalid_argument("restriction size mismatch");
    out.assign(coarse.size(), 0);
    for (int i = 0; i < fine.size(); ++i)
      for (auto [j, w] : weights[i]) out[j] += w * v[i];
  }
  SparseOperator galerkin(const SparseOperator &a) const {
    std::vector<std::map<int, double>> rows(coarse.size());
    for (int i = 0; i < a.size(); ++i)
      for (int p = a.offsets[i]; p < a.offsets[i + 1]; ++p)
        for (auto [r, wr] : weights[i])
          for (auto [c, wc] : weights[a.columns[p]])
            rows[r][c] += wr * a.values[p] * wc;
    if (coarse.boundary == Boundary::Dirichlet)
      for (int z = 0; z < coarse.shape[2]; ++z)
        for (int y = 0; y < coarse.shape[1]; ++y)
          for (int x = 0; x < coarse.shape[0]; ++x)
            if (coarse.constrained(x, y, z))
              for (int c = 0; c < coarse.components; ++c) {
                int row = coarse.components * coarse.node(x, y, z) + c;
                rows[row][row] = 1;
              }
    return SparseOperator::from_rows(rows);
  }
};
struct MultigridControls {
  int pre_smooth = 3, post_smooth = 3, coarse_nodes = 3;
  double jacobi_weight = .6;
  CoarseOperator coarse_operator = CoarseOperator::Rediscretized;
};
/** Host geometric V-cycle with point Jacobi, exact gauged coarse solve and
 * explicit rediscretization/Galerkin choice. The factory supplies an auxiliary
 * operator on each grid; it need not be the true nonsymmetric PDE Jacobian.
 * Periodic componentwise constants are projected out at every level.
 * Construction is separate from repeated applications. This serial hierarchy
 * can be wrapped by the explicit replicated MPI adapter for verification;
 * that adapter is not a scalable distributed coarse hierarchy.
 */
class Multigrid {
  struct Level {
    StructuredGrid grid;
    SparseOperator a;
  };
  std::vector<Level> levels_;
  std::vector<GridTransfer> transfers_;
  MultigridControls controls_;
  std::vector<double> lu_;
  std::vector<int> pivots_;
  void factor_coarse() {
    const auto &level = levels_.back();
    int n = level.a.size();
    lu_.assign(n * n, 0);
    pivots_.resize(n);
    for (int i = 0; i < n; ++i)
      for (int p = level.a.offsets[i]; p < level.a.offsets[i + 1]; ++p)
        lu_[i * n + level.a.columns[p]] = level.a.values[p];
    if (level.grid.boundary == Boundary::Periodic)
      for (int i = 0; i < n; ++i)
        for (int j = i % level.grid.components; j < n; j += level.grid.components)
          lu_[i * n + j] += 1.0 / level.grid.nodes();
    double scale = 0;
    for (double v : lu_) scale = std::max(scale, std::abs(v));
    for (int k = 0; k < n; ++k) {
      int pivot = k;
      for (int i = k + 1; i < n; ++i)
        if (std::abs(lu_[i * n + k]) > std::abs(lu_[pivot * n + k])) pivot = i;
      if (!(std::abs(lu_[pivot * n + k]) > 1e-13 * scale))
        throw std::runtime_error("singular multigrid coarse operator");
      pivots_[k] = pivot;
      for (int j = 0; j < n; ++j) std::swap(lu_[k * n + j], lu_[pivot * n + j]);
      for (int i = k + 1; i < n; ++i) {
        lu_[i * n + k] /= lu_[k * n + k];
        for (int j = k + 1; j < n; ++j)
          lu_[i * n + j] -= lu_[i * n + k] * lu_[k * n + j];
      }
    }
  }
  void smooth(const Level &l, const std::vector<double> &b, std::vector<double> &x,
              int sweeps) const {
    std::vector<double> ax;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      l.a(x, ax);
      for (int i = 0; i < l.a.size(); ++i)
        x[i] += controls_.jacobi_weight * (b[i] - ax[i]) / l.a.diagonal[i];
      l.grid.project(x);
    }
  }
  void cycle(int k, const std::vector<double> &b, std::vector<double> &x) const {
    const auto &l = levels_[k];
    if (k == static_cast<int>(levels_.size()) - 1) {
      int n = b.size();
      x = b;
      for (int i = 0; i < n; ++i) std::swap(x[i], x[pivots_[i]]);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j) x[i] -= lu_[i * n + j] * x[j];
      for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) x[i] -= lu_[i * n + j] * x[j];
        x[i] /= lu_[i * n + i];
      }
      l.grid.project(x);
      return;
    }
    smooth(l, b, x, controls_.pre_smooth);
    std::vector<double> r, bc, ec(levels_[k + 1].grid.size(), 0), ef;
    l.a(x, r);
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
    l.grid.project(r);
    transfers_[k].restrict(r, bc);
    levels_[k + 1].grid.project(bc);
    cycle(k + 1, bc, ec);
    transfers_[k].prolong(ec, ef);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += ef[i];
    smooth(l, b, x, controls_.post_smooth);
  }

public:
  using Factory = std::function<SparseOperator(const StructuredGrid &)>;
  Multigrid(StructuredGrid grid, Factory factory, MultigridControls controls = {})
      : controls_(controls) {
    grid.validate();
    if (controls.pre_smooth < 1 || controls.post_smooth < 1 ||
        controls.coarse_nodes < 3 ||
        !(controls.jacobi_weight > 0 && controls.jacobi_weight < 1))
      throw std::invalid_argument("invalid multigrid controls");
    levels_.push_back({grid, factory(grid)});
    while (*std::max_element(grid.shape.begin(), grid.shape.end()) >
           controls.coarse_nodes) {
      auto coarse = grid;
      for (int d = 0; d < 3; ++d)
        coarse.shape[d] = std::max(3, (grid.shape[d] + 1) / 2);
      transfers_.emplace_back(grid, coarse);
      auto a = controls.coarse_operator == CoarseOperator::Galerkin
                   ? transfers_.back().galerkin(levels_.back().a)
                   : factory(coarse);
      levels_.push_back({coarse, std::move(a)});
      grid = coarse;
    }
    for (const auto &l : levels_) {
      if (l.a.size() != l.grid.size())
        throw std::invalid_argument("hierarchy operator size mismatch");
      for (double d : l.a.diagonal)
        if (!(d > 0) || !std::isfinite(d))
          throw std::invalid_argument("multigrid requires positive finite diagonal");
    }
    factor_coarse();
  }
  void operator()(const std::vector<double> &r, std::vector<double> &z,
                  int = 0) const {
    auto b = r;
    levels_.front().grid.project(b);
    z.assign(b.size(), 0);
    cycle(0, b, z);
  }
  const SparseOperator &operator_at(int level) const { return levels_.at(level).a; }
  const GridTransfer &transfer_at(int level) const { return transfers_.at(level); }
  int levels() const { return levels_.size(); }
  std::size_t storage_bytes() const {
    std::size_t n = lu_.size() * sizeof(double) + pivots_.size() * sizeof(int);
    for (const auto &l : levels_)
      n += (l.a.values.size() + l.a.diagonal.size()) * sizeof(double) +
           (l.a.offsets.size() + l.a.columns.size()) * sizeof(int);
    for (const auto &t : transfers_)
      for (const auto &w : t.weights) n += w.size() * sizeof(std::pair<int, double>);
    return n;
  }
};
} // namespace pfc::solvers
