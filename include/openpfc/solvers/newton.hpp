// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/solvers/krylov.hpp>
#include <utility>
namespace pfc::solvers {
enum class NonlinearStatus {
  Converged,
  IterationLimit,
  LinearFailure,
  LineSearchFailure,
  NonFinite
};
struct NewtonControls {
  double relative_tolerance = 1e-9, absolute_tolerance = 0;
  int maximum_iterations = 32, maximum_backtracks = 16;
  bool adaptive_forcing = true;
  double initial_forcing = .1, minimum_forcing = 1e-9, maximum_forcing = .5;
  double forcing_gamma = .9, forcing_power = 2;
  double armijo = 1e-4, backtrack = .5;
  KrylovControls linear;
};
struct NonlinearStep {
  int iteration = 0, backtracks = 0;
  double residual_before = 0, residual_after = 0, forcing = 0, step_length = 0,
         correction_norm = 0;
  KrylovResult linear;
};
struct NonlinearResult {
  NonlinearStatus status = NonlinearStatus::IterationLimit;
  double initial_residual = 0, residual = 0;
  int residual_calls = 0, linearizations = 0, rejected_trials = 0;
  double residual_seconds = 0, linearization_seconds = 0;
  std::vector<NonlinearStep> history;
  bool converged() const { return status == NonlinearStatus::Converged; }
};
/**
 * Inexact Newton with Eisenstat-Walker choice-2 forcing and Armijo globalization.
 * Model: residual(x,r), linearize(x), jacobian(v,Jv), precondition(r,z,k),
 * begin_trial(x), accept_trial(x), reject_trial(). Trial hooks manage reversible
 * iterate caches, not permanent physical history: the caller commits material
 * history only after the whole increment is admitted. Jacobian/preconditioner
 * actions must never modify committed history. Rejected trials leave x intact.
 * A failed residual exception rejects the trial and propagates the original
 * exception; tolerances or constitutive iteration caps are never relaxed.
 */
template <class Ops, class Model>
NonlinearResult newton(Ops &ops, Model &model, typename Ops::Vector &x,
                       NewtonControls c = {}) {
  detail::validate(c.linear);
  if (!(c.relative_tolerance >= 0) || !(c.absolute_tolerance >= 0) ||
      !std::isfinite(c.relative_tolerance) || !std::isfinite(c.absolute_tolerance) ||
      (c.relative_tolerance == 0 && c.absolute_tolerance == 0) ||
      c.maximum_iterations < 0 || c.maximum_backtracks < 0 ||
      !(0 < c.minimum_forcing && c.minimum_forcing <= c.initial_forcing &&
        c.initial_forcing <= c.maximum_forcing && c.maximum_forcing < 1) ||
      !(c.forcing_gamma > 0 && c.forcing_gamma <= 1) ||
      !(c.forcing_power > 1 && c.forcing_power <= 2) ||
      !(c.armijo > 0 && c.armijo < 1) || !(c.backtrack > 0 && c.backtrack < 1))
    throw std::invalid_argument("invalid Newton controls");
  NonlinearResult out;
  auto r = ops.make(x.size()), trial_r = ops.make(x.size()),
       rhs = ops.make(x.size());
  auto s = ops.make(x.size()), trial = ops.make(x.size());
  auto residual = [&](const auto &state, auto &value) {
    auto t = std::chrono::steady_clock::now();
    model.residual(state, value);
    out.residual_seconds += detail::elapsed(t);
    ++out.residual_calls;
    if (value.size() != x.size())
      throw std::invalid_argument("residual size mismatch");
    return ops.norm(value);
  };
  out.initial_residual = out.residual = residual(x, r);
  const double stop =
      std::max(c.absolute_tolerance, c.relative_tolerance * out.initial_residual);
  double eta = c.initial_forcing;
  for (int k = 0;; ++k) {
    if (!std::isfinite(out.residual)) {
      out.status = NonlinearStatus::NonFinite;
      return out;
    }
    if (out.residual <= stop) {
      out.status = NonlinearStatus::Converged;
      return out;
    }
    if (k == c.maximum_iterations) return out;
    auto t = std::chrono::steady_clock::now();
    model.linearize(x);
    out.linearization_seconds += detail::elapsed(t);
    ++out.linearizations;
    ops.copy(r, rhs);
    ops.scale(rhs, -1);
    ops.copy(ops.make(x.size()), s);
    auto controls = c.linear;
    // The forcing criterion is relative to ||R_k||, with no absolute shortcut.
    controls.relative_tolerance = eta;
    controls.absolute_tolerance = 0;
    NonlinearStep record;
    record.iteration = k + 1;
    record.residual_before = out.residual;
    record.forcing = eta;
    record.linear = gmres(
        ops, [&](const auto &v, auto &w) { model.jacobian(v, w); }, rhs, s, controls,
        [&](const auto &v, auto &w, int i) { model.precondition(v, w, i); });
    if (!record.linear.converged() || record.linear.residual > eta * out.residual) {
      out.history.push_back(record);
      out.status = NonlinearStatus::LinearFailure;
      return out;
    }
    bool accepted = false;
    double alpha = 1, norm = 0;
    for (int bt = 0; bt <= c.maximum_backtracks; ++bt) {
      ops.copy(x, trial);
      ops.axpy(trial, alpha, s);
      model.begin_trial(trial);
      try {
        norm = residual(trial, trial_r);
      } catch (...) {
        model.reject_trial();
        ++out.rejected_trials;
        throw;
      }
      if (std::isfinite(norm) &&
          norm <= (1 - c.armijo * alpha * (1 - eta)) * out.residual) {
        model.accept_trial(trial);
        ops.copy(trial, x);
        ops.copy(trial_r, r);
        record.backtracks = bt;
        accepted = true;
        break;
      }
      model.reject_trial();
      ++out.rejected_trials;
      alpha *= c.backtrack;
    }
    record.step_length = accepted ? alpha : 0;
    record.correction_norm = accepted ? alpha * ops.norm(s) : 0;
    record.residual_after = norm;
    out.history.push_back(record);
    if (!accepted) {
      out.status = NonlinearStatus::LineSearchFailure;
      return out;
    }
    if (c.adaptive_forcing) {
      double next = c.forcing_gamma * std::pow(norm / out.residual, c.forcing_power);
      const double safeguard = c.forcing_gamma * std::pow(eta, c.forcing_power);
      if (safeguard > .1) next = std::max(next, safeguard);
      eta = std::clamp(next, c.minimum_forcing, c.maximum_forcing);
    }
    out.residual = norm;
  }
}
} // namespace pfc::solvers
