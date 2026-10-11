// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/solvers/finite_strain_fft/newton.hpp>
#include <openpfc/solvers/newton.hpp>
namespace pfc::finite_strain {
struct InexactResult {
  std::vector<Tensor2> deformation, piola;
  solvers::NonlinearResult nonlinear;
};
/** Solve projected equilibrium using generic residual-based Newton.
 * Initial deformation is the accepted field plus the prescribed macro step.
 * Compatible zero-mean corrections preserve its prescribed zero mode.
 * Unlike the published correction-based driver, convergence is established
 * by the explicitly evaluated projected residual relative to its initial norm.
 * Law evaluations always read the caller's unchanged committed material state.
 * The caller independently admits equilibrium and commits the load increment.
 */
template <LocalConstitutiveLaw Law>
InexactResult solve_nonlinear_increment(CompatibleProjector &projector,
                                        const Law &law, const Tensor2 &macro,
                                        const IncrementStart &start,
                                        solvers::NewtonControls controls = {},
                                        MPI_Comm comm = MPI_COMM_WORLD) {
  require_odd_grid(projector.domain());
  if (start.deformation.size() != projector.local_size())
    throw std::invalid_argument("increment needs one deformation per inbox point");
  struct Model {
    CompatibleProjector &projector;
    const Law &law;
    std::vector<Tensor2> state;
    void residual(const std::vector<Tensor2> &f, std::vector<Tensor2> &r) {
      r.resize(f.size());
      for (std::size_t i = 0; i < f.size(); ++i) r[i] = law.stress(i, f[i]);
      projector.apply(r, r);
    }
    void linearize(const std::vector<Tensor2> &f) { state = f; }
    void jacobian(const std::vector<Tensor2> &v, std::vector<Tensor2> &w) {
      w.resize(v.size());
      for (std::size_t i = 0; i < v.size(); ++i)
        w[i] = law.tangent_action(i, state[i], v[i]);
      projector.apply(w, w);
    }
    void precondition(const std::vector<Tensor2> &r, std::vector<Tensor2> &z, int) {
      z = r;
    }
    void begin_trial(const std::vector<Tensor2> &) {}
    void accept_trial(const std::vector<Tensor2> &) {}
    void reject_trial() {}
  } model{projector, law, {}};
  detail::TensorOperations ops(comm);
  InexactResult result;
  result.deformation = start.deformation;
  const Tensor2 step = add(macro, scaled(start.macroscopic, -1));
  for (auto &f : result.deformation) f = add(f, step);
  result.nonlinear = solvers::newton(ops, model, result.deformation, controls);
  result.piola.resize(result.deformation.size());
  for (std::size_t i = 0; i < result.piola.size(); ++i)
    result.piola[i] = law.stress(i, result.deformation[i]);
  return result;
}
} // namespace pfc::finite_strain
