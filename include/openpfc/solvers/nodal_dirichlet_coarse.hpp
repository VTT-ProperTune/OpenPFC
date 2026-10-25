// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file nodal_dirichlet_coarse.hpp
 * @brief The single bounded replicated Q1 coarse Cholesky shared by CPU/GPU GMG.
 *
 * The factor lives on the host. It is ONLY for a tiny, explicit free-node
 * Dirichlet coarse grid. GPU multigrid may reduce and download that coarse
 * right-hand side but must never send a fine/intermediate field here.
 */
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/types.hpp>
#include <openpfc/solvers/nodal_dirichlet_kernels.hpp>

namespace pfc::solvers::nmg::dirichlet {

class DenseCoarseCholesky {
public:
  DenseCoarseCholesky() = default;

  /// Rows are free Q1 nodes, so physical Dirichlet boundary nodes have
  /// already been eliminated. The capacity gate is checked BEFORE dense
  /// allocation. Stencil weights must yield a symmetric positive matrix.
  DenseCoarseCholesky(const Int3 &free_nodes, const Weights &weights,
                      std::size_t max_unknowns) {
    factor(free_nodes, weights, max_unknowns);
  }

  void factor(const Int3 &free_nodes, const Weights &weights,
              std::size_t max_unknowns) {
    if (max_unknowns == 0 ||
        max_unknowns > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("DenseCoarseCholesky: invalid coarse capacity");
    std::size_t N = 1;
    for (int d = 0; d < 3; ++d) {
      if (free_nodes[d] < 1 ||
          static_cast<std::size_t>(free_nodes[d]) > max_unknowns / N)
        throw std::invalid_argument(
            "DenseCoarseCholesky: coarse grid exceeds capacity");
      N *= static_cast<std::size_t>(free_nodes[d]);
    }
    if (N > max_unknowns || N > std::numeric_limits<std::size_t>::max() / N)
      throw std::invalid_argument("DenseCoarseCholesky: dense allocation overflow");

    // Do not mutate the existing valid factor until the new candidate has
    // passed its Cholesky pivots. Failed numerical admission fails closed.
    std::vector<double> L(N * N, 0.0);
    for (int k = 0; k < free_nodes[2]; ++k)
      for (int j = 0; j < free_nodes[1]; ++j)
        for (int i = 0; i < free_nodes[0]; ++i) {
          const std::size_t row = linear(free_nodes, i, j, k);
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const int ci = i + dx, cj = j + dy, ck = k + dz;
                if (ci >= 0 && cj >= 0 && ck >= 0 && ci < free_nodes[0] &&
                    cj < free_nodes[1] && ck < free_nodes[2]) {
                  const double a = weights.v[offset_index(dx, dy, dz)];
                  if (!std::isfinite(a))
                    throw std::invalid_argument(
                        "DenseCoarseCholesky: nonfinite Q1 stencil");
                  L[row * N + linear(free_nodes, ci, cj, ck)] += a;
                }
              }
        }

    // The coarse solve is the fixed SPD inverse required by PCG. The
    // positive Cholesky pivots alone do not detect an asymmetric stencil;
    // verify the actual bounded matrix separately.
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < i; ++j) {
        const double a = L[i * N + j], b = L[j * N + i];
        if (std::abs(a - b) > 128.0 * std::numeric_limits<double>::epsilon() *
                                  (1.0 + std::abs(a) + std::abs(b)))
          throw std::invalid_argument(
              "DenseCoarseCholesky: asymmetric Q1 coarse operator");
      }

    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j <= i; ++j) {
        double entry = L[i * N + j];
        for (std::size_t k = 0; k < j; ++k) entry -= L[i * N + k] * L[j * N + k];
        if (i == j) {
          if (!(entry > 0.0) || !std::isfinite(entry))
            throw std::runtime_error(
                "DenseCoarseCholesky: Q1 coarse matrix is not SPD");
          L[i * N + i] = std::sqrt(entry);
        } else {
          L[i * N + j] = entry / L[j * N + j];
        }
      }
    m_nodes = free_nodes;
    m_N = N;
    m_lower = std::move(L);
  }

  [[nodiscard]] std::size_t unknowns() const noexcept { return m_N; }
  [[nodiscard]] const Int3 &free_nodes() const noexcept { return m_nodes; }

  /// Solve using one pre-factored host matrix; no assembly or network
  /// transfers, no MPI, and no access to fine-grid data.
  void solve(const std::vector<double> &rhs, std::vector<double> &x) const {
    if (!m_N || rhs.size() != m_N)
      throw std::invalid_argument("DenseCoarseCholesky: wrong coarse RHS size");
    if (&rhs == &x)
      throw std::invalid_argument(
          "DenseCoarseCholesky: RHS and destination must not alias");

    x.resize(m_N);
    for (std::size_t i = 0; i < m_N; ++i) {
      double value = rhs[i];
      if (!std::isfinite(value))
        throw std::invalid_argument("DenseCoarseCholesky: nonfinite coarse RHS");
      for (std::size_t j = 0; j < i; ++j) value -= m_lower[i * m_N + j] * x[j];
      x[i] = value / m_lower[i * m_N + i];
    }
    for (std::size_t i = m_N; i-- > 0;) {
      double value = x[i];
      for (std::size_t j = i + 1; j < m_N; ++j) value -= m_lower[j * m_N + i] * x[j];
      x[i] = value / m_lower[i * m_N + i];
    }
  }

private:
  static std::size_t linear(const Int3 &n, int i, int j, int k) {
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(n[0]) *
               (static_cast<std::size_t>(j) +
                static_cast<std::size_t>(n[1]) * static_cast<std::size_t>(k));
  }

  Int3 m_nodes{};
  std::size_t m_N = 0;
  std::vector<double> m_lower;
};

} // namespace pfc::solvers::nmg::dirichlet
