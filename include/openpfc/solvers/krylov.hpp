// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <openpfc/solvers/vector_operations.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc::solvers {
enum class Preconditioning { None, Left, Right };
enum class LinearStatus {
  Converged,
  IterationLimit,
  Breakdown,
  Stagnated,
  NonFinite
};
struct KrylovControls {
  double relative_tolerance = 1e-8;
  double absolute_tolerance = 0;
  int maximum_iterations = 400;
  int restart = 30;
  Preconditioning preconditioning = Preconditioning::Right;
  /// Variable inverse-preconditioners are supported only on the right (FGMRES).
  bool flexible = true;
  int stagnation_restarts = 4;
};
struct LinearRecord {
  int iteration = 0, restart = 0;
  double estimated_residual = 0, true_residual = -1;
};
struct KrylovResult {
  LinearStatus status = LinearStatus::IterationLimit;
  int iterations = 0, restarts = 0, operator_calls = 0, preconditioner_calls = 0;
  double residual = 0, rhs_norm = 0, operator_seconds = 0,
         preconditioner_seconds = 0;
  std::vector<LinearRecord> history;
  bool converged() const { return status == LinearStatus::Converged; }
};
namespace detail {
inline void validate(const KrylovControls &c) {
  if (!(c.relative_tolerance >= 0) || !(c.absolute_tolerance >= 0) ||
      !std::isfinite(c.relative_tolerance) || !std::isfinite(c.absolute_tolerance) ||
      (c.relative_tolerance == 0 && c.absolute_tolerance == 0) ||
      c.maximum_iterations < 0 || c.restart < 1 || c.stagnation_restarts < 1)
    throw std::invalid_argument("invalid Krylov controls");
}
inline double elapsed(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
} // namespace detail

/**
 * Restarted GMRES with two classical Gram-Schmidt passes and true residual
 * admission. The second pass is the reorthogonalization.
 * A(x,y) computes A*x; inverse M(r,z,iteration) computes M_k^{-1}*r.
 * Left preconditioning must be fixed and linear. Right preconditioning stores
 * every z_j, supporting variable/nonlinear M_k when flexible=true (FGMRES).
 * The returned x is always in physical coordinates, never M^{-1} coordinates.
 * Estimated/preconditioned residuals never establish convergence: ||b-A*x||
 * must be <= max(atol,rtol*||b||). An incompatible singular RHS fails closed.
 * Nullspace/gauge restrictions belong to the operator/preconditioner adapter.
 */
template <class Ops, class Action, class Preconditioner = IdentityPreconditioner>
KrylovResult gmres(Ops &ops, Action &&action, const typename Ops::Vector &b,
                   typename Ops::Vector &x, KrylovControls c = {},
                   Preconditioner precondition = {}) {
  detail::validate(c);
  if (c.flexible && c.preconditioning == Preconditioning::Left)
    throw std::invalid_argument("flexible left preconditioning is unsupported");
  if (x.size() != b.size())
    throw std::invalid_argument("initial guess size mismatch");
  using Vector = typename Ops::Vector;
  KrylovResult out;
  auto product = ops.make(b.size()), residual = ops.make(b.size());
  auto apply = [&](const Vector &v, Vector &w) {
    auto t = std::chrono::steady_clock::now();
    action(v, w);
    out.operator_seconds += detail::elapsed(t);
    ++out.operator_calls;
    if (w.size() != b.size())
      throw std::invalid_argument("operator output size mismatch");
  };
  auto inverse = [&](const Vector &v, Vector &w) {
    auto t = std::chrono::steady_clock::now();
    precondition(v, w, out.iterations);
    out.preconditioner_seconds += detail::elapsed(t);
    ++out.preconditioner_calls;
    if (w.size() != b.size())
      throw std::invalid_argument("preconditioner output size mismatch");
  };
  auto true_residual = [&]() {
    apply(x, product);
    ops.copy(b, residual);
    ops.axpy(residual, -1, product);
    out.residual = ops.norm(residual);
    return out.residual;
  };
  out.rhs_norm = ops.norm(b);
  const double stop =
      std::max(c.absolute_tolerance, c.relative_tolerance * out.rhs_norm);
  if (!std::isfinite(out.rhs_norm) || !std::isfinite(true_residual())) {
    out.status = LinearStatus::NonFinite;
    return out;
  }
  if (out.residual <= stop) {
    out.status = LinearStatus::Converged;
    return out;
  }
  std::vector<Vector> v(c.restart + 1), z(c.restart);
  for (auto &a : v) a = ops.make(b.size());
  for (auto &a : z) a = ops.make(b.size());
  std::vector<double> h((c.restart + 1) * c.restart), cs(c.restart), sn(c.restart),
      g(c.restart + 1), coeff(c.restart);
  auto H = [&](int i, int j) -> double & { return h[i * c.restart + j]; };
  int stagnant = 0;
  double previous = out.residual;
  while (out.iterations < c.maximum_iterations) {
    std::fill(h.begin(), h.end(), 0);
    std::fill(g.begin(), g.end(), 0);
    if (c.preconditioning == Preconditioning::Left)
      inverse(residual, v[0]);
    else
      ops.copy(residual, v[0]);
    const double beta = ops.norm(v[0]);
    if (!std::isfinite(beta)) {
      out.status = LinearStatus::NonFinite;
      return out;
    }
    if (!(beta > 0)) {
      out.status = LinearStatus::Breakdown;
      return out;
    }
    ops.scale(v[0], 1 / beta);
    g[0] = beta;
    const double estimated_stop = c.preconditioning == Preconditioning::Left
                                      ? c.relative_tolerance * beta
                                      : stop;
    bool breakdown = false;
    int columns = 0;
    for (int j = 0; j < c.restart && out.iterations < c.maximum_iterations; ++j) {
      if (c.preconditioning == Preconditioning::Right)
        inverse(v[j], z[j]);
      else
        ops.copy(v[j], z[j]);
      apply(z[j], product);
      if (c.preconditioning == Preconditioning::Left)
        inverse(product, v[j + 1]);
      else
        ops.copy(product, v[j + 1]);
      const double before = ops.norm(v[j + 1]);
      // Two classical passes. Each pass reads every inner product before any
      // subtraction, so those coefficients share one reduction. The second
      // pass reorthogonalizes. One classical pass is not used: it loses
      // orthogonality on a long basis.
      for (int pass = 0; pass < 2; ++pass) {
        ops.dots(v.data(), j + 1, v[j + 1], coeff.data());
        for (int i = 0; i <= j; ++i) {
          H(i, j) += coeff[i];
          ops.axpy(v[j + 1], -coeff[i], v[i]);
        }
      }
      H(j + 1, j) = ops.norm(v[j + 1]);
      if (!std::isfinite(before) || !std::isfinite(H(j + 1, j))) {
        out.status = LinearStatus::NonFinite;
        return out;
      }
      breakdown =
          H(j + 1, j) <= 64 * std::numeric_limits<double>::epsilon() * before;
      if (!breakdown) ops.scale(v[j + 1], 1 / H(j + 1, j));
      for (int i = 0; i < j; ++i) {
        double a = H(i, j), d = H(i + 1, j);
        H(i, j) = cs[i] * a + sn[i] * d;
        H(i + 1, j) = -sn[i] * a + cs[i] * d;
      }
      const double length = std::hypot(H(j, j), H(j + 1, j));
      if (!(length > 0) || !std::isfinite(length)) {
        out.status = std::isfinite(length) ? LinearStatus::Breakdown
                                           : LinearStatus::NonFinite;
        return out;
      }
      cs[j] = H(j, j) / length;
      sn[j] = H(j + 1, j) / length;
      H(j, j) = length;
      H(j + 1, j) = 0;
      g[j + 1] = -sn[j] * g[j];
      g[j] *= cs[j];
      ++out.iterations;
      columns = j + 1;
      out.history.push_back({out.iterations, out.restarts, std::abs(g[j + 1]), -1});
      if (std::abs(g[j + 1]) <= estimated_stop || breakdown) break;
    }
    std::vector<double> y(columns);
    for (int i = columns - 1; i >= 0; --i) {
      double a = g[i];
      for (int j = i + 1; j < columns; ++j) a -= H(i, j) * y[j];
      y[i] = a / H(i, i);
    }
    for (int i = 0; i < columns; ++i) ops.axpy(x, y[i], z[i]);
    true_residual();
    out.history.back().true_residual = out.residual;
    if (!std::isfinite(out.residual)) {
      out.status = LinearStatus::NonFinite;
      return out;
    }
    if (out.residual <= stop) {
      out.status = LinearStatus::Converged;
      return out;
    }
    if (breakdown) {
      out.status = LinearStatus::Breakdown;
      return out;
    }
    stagnant = out.residual >= previous * (1 - 1e-12) ? stagnant + 1 : 0;
    if (stagnant >= c.stagnation_restarts) {
      out.status = LinearStatus::Stagnated;
      return out;
    }
    previous = out.residual;
    ++out.restarts;
  }
  return out;
}

/** PCG for a caller-established SPD operator and fixed SPD inverse M.
 * None bypasses M; otherwise CG requires a fixed SPD inverse regardless
 * of the GMRES-only flexible/side options.
 * Curvature checks catch some violations but never prove symmetry or SPD.
 * Every convergence claim is checked against the original unpreconditioned RHS.
 */
template <class Ops, class Action, class Preconditioner = IdentityPreconditioner>
KrylovResult cg(Ops &ops, Action &&action, const typename Ops::Vector &b,
                typename Ops::Vector &x, KrylovControls c = {},
                Preconditioner inverse = {}) {
  detail::validate(c);
  if (x.size() != b.size())
    throw std::invalid_argument("initial guess size mismatch");
  KrylovResult out;
  auto product = ops.make(b.size()), r = ops.make(b.size()), z = ops.make(b.size()),
       p = ops.make(b.size());
  auto apply = [&](const auto &v, auto &w) {
    auto t = std::chrono::steady_clock::now();
    action(v, w);
    out.operator_seconds += detail::elapsed(t);
    ++out.operator_calls;
    if (w.size() != b.size())
      throw std::invalid_argument("operator output size mismatch");
  };
  auto precondition = [&]() {
    if (c.preconditioning == Preconditioning::None) {
      ops.copy(r, z);
      return;
    }
    auto t = std::chrono::steady_clock::now();
    inverse(r, z, out.iterations);
    out.preconditioner_seconds += detail::elapsed(t);
    ++out.preconditioner_calls;
    if (z.size() != b.size())
      throw std::invalid_argument("preconditioner output size mismatch");
  };
  apply(x, product);
  ops.copy(b, r);
  ops.axpy(r, -1, product);
  out.rhs_norm = ops.norm(b);
  out.residual = ops.norm(r);
  const double stop =
      std::max(c.absolute_tolerance, c.relative_tolerance * out.rhs_norm);
  if (!std::isfinite(out.residual) || !std::isfinite(out.rhs_norm)) {
    out.status = LinearStatus::NonFinite;
    return out;
  }
  if (out.residual <= stop) {
    out.status = LinearStatus::Converged;
    return out;
  }
  precondition();
  ops.copy(z, p);
  double rho = ops.dot(r, z);
  for (int k = 0; k < c.maximum_iterations; ++k) {
    apply(p, product);
    double curvature = ops.dot(p, product);
    if (!std::isfinite(rho) || !std::isfinite(curvature)) {
      out.status = LinearStatus::NonFinite;
      return out;
    }
    if (!(rho > 0) || !(curvature > 0)) {
      out.status = LinearStatus::Breakdown;
      return out;
    }
    double alpha = rho / curvature;
    ops.axpy(x, alpha, p);
    ops.axpy(r, -alpha, product);
    ++out.iterations;
    out.residual = ops.norm(r);
    if (!std::isfinite(out.residual)) {
      out.status = LinearStatus::NonFinite;
      return out;
    }
    out.history.push_back({out.iterations, 0, out.residual, -1});
    if (out.residual <= stop || out.iterations == c.maximum_iterations) {
      apply(x, product);
      ops.copy(b, r);
      ops.axpy(r, -1, product);
      out.residual = ops.norm(r);
      out.history.back().true_residual = out.residual;
      if (!std::isfinite(out.residual)) {
        out.status = LinearStatus::NonFinite;
        return out;
      }
      if (out.residual <= stop) {
        out.status = LinearStatus::Converged;
        return out;
      }
      // Residual replacement invalidates the recurrence; restart from r.
      precondition();
      ops.copy(z, p);
      rho = ops.dot(r, z);
      continue;
    }
    precondition();
    double next = ops.dot(r, z);
    ops.scale(p, next / rho);
    ops.axpy(p, 1, z);
    rho = next;
  }
  return out;
}
} // namespace pfc::solvers
