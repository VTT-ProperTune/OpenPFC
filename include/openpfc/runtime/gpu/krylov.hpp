// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <limits>
#include <openpfc/runtime/gpu/vector_operations.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <type_traits>

namespace pfc::solvers::gpu {
inline constexpr int FlagBreakdown = 1;
inline constexpr int FlagNonFinite = 2;
inline constexpr int FlagAbort = 3;
inline constexpr int FlagEstimated = 4;
namespace detail {
inline constexpr int SlotBeta = 0, SlotBefore = 1, SlotDot = 2, SlotScale = 3,
                     SlotAlpha = 4, SlotRho = 5, SlotCurvature = 6, SlotNext = 7,
                     SlotStop = 8, SlotOne = 9, SlotFixed = 10;
template <class Tag>
double download(VectorOperations<Tag> &ops, const double *slot) {
  double value = 0;
  check(pfc::gpuMemcpy(&value, slot, sizeof(double), pfc::gpuMemcpyDeviceToHost));
  ops.scalar_transfer_bytes += sizeof(double);
  return value;
}
template <class Tag>
ScalarPoll download_poll(VectorOperations<Tag> &ops, const ScalarPoll *slot) {
  ScalarPoll value{};
  check(pfc::gpuMemcpy(&value, slot, sizeof(value), pfc::gpuMemcpyDeviceToHost));
  ops.scalar_transfer_bytes += sizeof(value);
  return value;
}
template <class Tag>
void norm_into(VectorOperations<Tag> &ops,
               const typename VectorOperations<Tag>::Vector &v, double *slot,
               const ScalarPoll *poll) {
  ops.dot_store(v, v, slot, poll);
  Resident<Tag>::sqrt_slot(slot, poll);
}
inline void require_poll(int poll_interval) {
  if (poll_interval < 1) throw std::invalid_argument("invalid Krylov poll interval");
}
} // namespace detail

/// PCG with the same recurrence as `solvers::cg`. Dot products, the step
/// length and the recurrence residual stay in device scalars. The host copies
/// one `ScalarPoll` every `poll_interval` iterations. Interval 1 matches the
/// reference stop. A latched breakdown or estimated residual skips further
/// vector updates until that copy.
template <class Tag, class Action, class Preconditioner = IdentityPreconditioner>
KrylovResult cg(VectorOperations<Tag> &ops, Action &&action,
                const typename VectorOperations<Tag>::Vector &b,
                typename VectorOperations<Tag>::Vector &x, KrylovControls c = {},
                Preconditioner inverse = {}, int poll_interval = 1) {
  static_assert(std::is_standard_layout_v<ScalarPoll>);
  static_assert(offsetof(ScalarPoll, flag) == sizeof(double));
  pfc::solvers::detail::validate(c);
  detail::require_poll(poll_interval);
  if (x.size() != b.size())
    throw std::invalid_argument("initial guess size mismatch");
  using Vector = typename VectorOperations<Tag>::Vector;
  KrylovResult out;
  auto product = ops.make(b.size()), r = ops.make(b.size()), z = ops.make(b.size()),
       p = ops.make(b.size());
  auto apply = [&](const Vector &v, Vector &w) {
    auto t = std::chrono::steady_clock::now();
    action(v, w);
    out.operator_seconds += pfc::solvers::detail::elapsed(t);
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
    out.preconditioner_seconds += pfc::solvers::detail::elapsed(t);
    ++out.preconditioner_calls;
    if (z.size() != b.size())
      throw std::invalid_argument("preconditioner output size mismatch");
  };
  pfc::core::DataBuffer<Tag, double> slots(detail::SlotFixed);
  pfc::core::DataBuffer<Tag, ScalarPoll> poll_buf(1);
  Resident<Tag>::fill(slots.data(), slots.size(), 0);
  Resident<Tag>::clear_poll(poll_buf.data());
  Resident<Tag>::set_double(slots.data() + detail::SlotOne, 1);
  double *dot = slots.data() + detail::SlotDot;
  double *alpha = slots.data() + detail::SlotAlpha;
  double *rho = slots.data() + detail::SlotRho;
  double *curv = slots.data() + detail::SlotCurvature;
  double *next = slots.data() + detail::SlotNext;
  double *stop_slot = slots.data() + detail::SlotStop;
  double *one = slots.data() + detail::SlotOne;
  ScalarPoll *poll = poll_buf.data();
  apply(x, product);
  ops.copy(b, r);
  ops.axpy(r, -1, product);
  detail::norm_into(ops, b, dot, nullptr);
  out.rhs_norm = detail::download(ops, dot);
  detail::norm_into(ops, r, dot, nullptr);
  out.residual = detail::download(ops, dot);
  const double stop =
      std::max(c.absolute_tolerance, c.relative_tolerance * out.rhs_norm);
  Resident<Tag>::set_double(stop_slot, stop);
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
  ops.dot_store(r, z, rho, nullptr);
  auto recur = [&]() {
    ops.dot_store(r, z, next, poll);
    ops.scale_ratio_slot(p, next, rho, poll);
    ops.axpy_slot(p, one, z, 1, poll);
    Resident<Tag>::copy_double(rho, next, poll);
  };
  while (out.iterations < c.maximum_iterations) {
    int batch = std::min(poll_interval, c.maximum_iterations - out.iterations);
    const int began = out.iterations;
    for (int step = 0; step < batch; ++step) {
      apply(p, product);
      ops.dot_store(p, product, curv, poll);
      Resident<Tag>::cg_alpha(rho, curv, alpha, poll);
      ops.axpy_slot(x, alpha, p, 1, poll);
      ops.axpy_slot(r, alpha, product, -1, poll);
      Resident<Tag>::inc_iter(poll);
      ops.dot_store(r, r, dot, poll);
      Resident<Tag>::publish_sqrt(dot, poll);
      Resident<Tag>::latch_stop(poll, stop_slot);
      if (step + 1 < batch) {
        precondition();
        recur();
      }
    }
    auto viewed = detail::download_poll(ops, poll);
    out.iterations = viewed.iterations;
    if (viewed.iterations == began && viewed.flag == 0)
      throw std::runtime_error("resident Krylov made no progress");
    if (viewed.flag == FlagBreakdown) {
      out.status = LinearStatus::Breakdown;
      return out;
    }
    if (viewed.flag == FlagNonFinite || viewed.flag == FlagAbort) {
      if (viewed.flag == FlagAbort) out.residual = viewed.value;
      out.status = LinearStatus::NonFinite;
      return out;
    }
    out.residual = viewed.value;
    if (!std::isfinite(out.residual)) {
      out.status = LinearStatus::NonFinite;
      return out;
    }
    out.history.push_back({out.iterations, 0, out.residual, -1});
    if (viewed.flag == FlagEstimated || out.residual <= stop ||
        out.iterations == c.maximum_iterations) {
      apply(x, product);
      ops.copy(b, r);
      ops.axpy(r, -1, product);
      detail::norm_into(ops, r, dot, nullptr);
      out.residual = detail::download(ops, dot);
      out.history.back().true_residual = out.residual;
      if (!std::isfinite(out.residual)) {
        out.status = LinearStatus::NonFinite;
        return out;
      }
      if (out.residual <= stop) {
        out.status = LinearStatus::Converged;
        return out;
      }
      Resident<Tag>::clear_flag(poll);
      precondition();
      ops.copy(z, p);
      ops.dot_store(r, z, rho, nullptr);
      continue;
    }
    precondition();
    recur();
  }
  return out;
}

/// Restarted GMRES / FGMRES with the same recurrence as `solvers::gmres`.
/// Gram-Schmidt coefficients and Givens rotations stay on the device. The
/// host copies one convergence probe every `poll_interval` columns.
template <class Tag, class Action, class Preconditioner = IdentityPreconditioner>
KrylovResult gmres(VectorOperations<Tag> &ops, Action &&action,
                   const typename VectorOperations<Tag>::Vector &b,
                   typename VectorOperations<Tag>::Vector &x, KrylovControls c = {},
                   Preconditioner precondition = {}, int poll_interval = 1) {
  pfc::solvers::detail::validate(c);
  detail::require_poll(poll_interval);
  if (c.flexible && c.preconditioning == Preconditioning::Left)
    throw std::invalid_argument("flexible left preconditioning is unsupported");
  if (x.size() != b.size())
    throw std::invalid_argument("initial guess size mismatch");
  using Vector = typename VectorOperations<Tag>::Vector;
  KrylovResult out;
  auto product = ops.make(b.size()), residual = ops.make(b.size());
  auto apply = [&](const Vector &v, Vector &w) {
    auto t = std::chrono::steady_clock::now();
    action(v, w);
    out.operator_seconds += pfc::solvers::detail::elapsed(t);
    ++out.operator_calls;
    if (w.size() != b.size())
      throw std::invalid_argument("operator output size mismatch");
  };
  auto inverse = [&](const Vector &v, Vector &w) {
    auto t = std::chrono::steady_clock::now();
    precondition(v, w, out.iterations);
    out.preconditioner_seconds += pfc::solvers::detail::elapsed(t);
    ++out.preconditioner_calls;
    if (w.size() != b.size())
      throw std::invalid_argument("preconditioner output size mismatch");
  };
  const int m = c.restart;
  const int coeff_n = m;
  const int ng = m + 1;
  const int nh = (m + 1) * m;
  const int total = detail::SlotFixed + coeff_n + m + m + ng + m + nh;
  pfc::core::DataBuffer<Tag, double> slots(static_cast<std::size_t>(total));
  pfc::core::DataBuffer<Tag, ScalarPoll> poll_buf(1);
  Resident<Tag>::fill(slots.data(), slots.size(), 0);
  Resident<Tag>::clear_poll(poll_buf.data());
  double *base = slots.data();
  double *beta = base + detail::SlotBeta;
  double *before = base + detail::SlotBefore;
  double *dot = base + detail::SlotDot;
  double *scale = base + detail::SlotScale;
  double *stop_slot = base + detail::SlotStop;
  double *coeff = base + detail::SlotFixed;
  double *cs = coeff + coeff_n;
  double *sn = cs + m;
  double *g = sn + m;
  double *y = g + ng;
  double *H = y + m;
  ScalarPoll *poll = poll_buf.data();
  const double eps64 = 64 * std::numeric_limits<double>::epsilon();
  auto true_residual = [&]() {
    apply(x, product);
    ops.copy(b, residual);
    ops.axpy(residual, -1, product);
    detail::norm_into(ops, residual, dot, nullptr);
    out.residual = detail::download(ops, dot);
    return out.residual;
  };
  detail::norm_into(ops, b, dot, nullptr);
  out.rhs_norm = detail::download(ops, dot);
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
  std::vector<Vector> v, z;
  v.reserve(static_cast<std::size_t>(m + 1));
  z.reserve(static_cast<std::size_t>(m));
  for (int i = 0; i < m + 1; ++i) v.push_back(ops.make(b.size()));
  for (int i = 0; i < m; ++i) z.push_back(ops.make(b.size()));
  const bool left = c.preconditioning == Preconditioning::Left;
  const bool right = c.preconditioning == Preconditioning::Right;
  int stagnant = 0;
  double previous = out.residual;
  while (out.iterations < c.maximum_iterations) {
    Resident<Tag>::clear_restart(poll);
    Resident<Tag>::fill(H, static_cast<std::size_t>(nh), 0);
    Resident<Tag>::fill(cs, static_cast<std::size_t>(m), 0);
    Resident<Tag>::fill(sn, static_cast<std::size_t>(m), 0);
    Resident<Tag>::fill(g, static_cast<std::size_t>(ng), 0);
    if (left)
      inverse(residual, v[0]);
    else
      ops.copy(residual, v[0]);
    detail::norm_into(ops, v[0], beta, nullptr);
    const double beta_h = detail::download(ops, beta);
    if (!std::isfinite(beta_h)) {
      out.status = LinearStatus::NonFinite;
      return out;
    }
    if (!(beta_h > 0)) {
      out.status = LinearStatus::Breakdown;
      return out;
    }
    ops.scale_inv_slot(v[0], beta, nullptr);
    Resident<Tag>::set_double(g, beta_h);
    Resident<Tag>::set_double(stop_slot,
                              left ? c.relative_tolerance * beta_h : stop);
    int built = 0;
    int latched = 0;
    while (built < m && out.iterations < c.maximum_iterations) {
      int batch = std::min(
          poll_interval, std::min(m - built, c.maximum_iterations - out.iterations));
      const int began = out.iterations;
      for (int step = 0; step < batch; ++step) {
        const int j = built + step;
        if (right)
          inverse(v[static_cast<std::size_t>(j)], z[static_cast<std::size_t>(j)]);
        else
          ops.copy(v[static_cast<std::size_t>(j)], z[static_cast<std::size_t>(j)]);
        apply(z[static_cast<std::size_t>(j)], product);
        if (left)
          inverse(product, v[static_cast<std::size_t>(j + 1)]);
        else
          ops.copy(product, v[static_cast<std::size_t>(j + 1)]);
        detail::norm_into(ops, v[static_cast<std::size_t>(j + 1)], before, poll);
        for (int pass = 0; pass < 2; ++pass) {
          ops.dots_store(v.data(), j + 1, v[static_cast<std::size_t>(j + 1)], coeff,
                         poll);
          Resident<Tag>::add_h(H, coeff, j, m, poll);
          for (int i = 0; i <= j; ++i)
            ops.axpy_slot(v[static_cast<std::size_t>(j + 1)], coeff + i,
                          v[static_cast<std::size_t>(i)], -1, poll);
        }
        ops.dot_store(v[static_cast<std::size_t>(j + 1)],
                      v[static_cast<std::size_t>(j + 1)], dot, poll);
        Resident<Tag>::prepare_arnoldi(H, dot, before, j, m, eps64, scale, poll);
        ops.scale_mul_slot(v[static_cast<std::size_t>(j + 1)], scale, poll);
        Resident<Tag>::givens(H, cs, sn, g, j, m, stop_slot, poll);
      }
      auto viewed = detail::download_poll(ops, poll);
      const int gained = viewed.iterations - began;
      out.iterations = viewed.iterations;
      built = viewed.columns;
      latched = viewed.flag;
      if (gained > 0)
        out.history.push_back({out.iterations, out.restarts, viewed.value, -1});
      if (gained == 0 && latched == 0)
        throw std::runtime_error("resident Krylov made no progress");
      if (latched == FlagNonFinite) {
        out.status = LinearStatus::NonFinite;
        return out;
      }
      if (latched == FlagAbort) {
        out.status = LinearStatus::Breakdown;
        return out;
      }
      if (latched == FlagBreakdown || latched == FlagEstimated) break;
    }
    Resident<Tag>::backsolve(H, g, y, built, m);
    for (int i = 0; i < built; ++i)
      ops.axpy_slot(x, y + i, z[static_cast<std::size_t>(i)], 1, nullptr);
    true_residual();
    if (!out.history.empty()) out.history.back().true_residual = out.residual;
    if (!std::isfinite(out.residual)) {
      out.status = LinearStatus::NonFinite;
      return out;
    }
    if (out.residual <= stop) {
      out.status = LinearStatus::Converged;
      return out;
    }
    if (latched == FlagBreakdown) {
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
} // namespace pfc::solvers::gpu
