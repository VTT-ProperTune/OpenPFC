// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <exception>
#include <openpfc/solvers/finite_strain_fft/newton.hpp>
#include <openpfc/solvers/newton.hpp>
namespace pfc::finite_strain {
namespace detail {
// Finish local material work on every rank before entering a projector
// collective. Propagate the original exception locally and a remote failure
// elsewhere; no rank continues into FFT while a peer unwinds its solve.
template <class Evaluation>
void collective_material_evaluation(MPI_Comm comm, Evaluation evaluate) {
  std::exception_ptr error;
  try {
    evaluate();
  } catch (...) {
    error = std::current_exception();
  }
  const int local_failed = error ? 1 : 0;
  int failed = local_failed;
  if (comm != MPI_COMM_NULL && MPI_Allreduce(&local_failed, &failed, 1, MPI_INT,
                                             MPI_MAX, comm) != MPI_SUCCESS)
    throw std::runtime_error("material admission collective failed");
  if (error) std::rethrow_exception(error);
  if (failed) throw std::runtime_error("material evaluation failed on another rank");
}
} // namespace detail
struct InexactResult {
  std::vector<Tensor2> deformation, piola;
  solvers::NonlinearResult nonlinear;
  solvers::ReductionMetrics reductions;
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
  if (start.deformation.size() != projector.local_size())
    throw std::invalid_argument("increment needs one deformation per inbox point");
  struct Model {
    CompatibleProjector &projector;
    const Law &law;
    MPI_Comm comm;
    std::vector<Tensor2> state;
    void residual(const std::vector<Tensor2> &f, std::vector<Tensor2> &r) {
      r.resize(f.size());
      detail::collective_material_evaluation(comm, [&] {
        for (std::size_t i = 0; i < f.size(); ++i) r[i] = law.stress(i, f[i]);
      });
      projector.apply(r, r);
    }
    void linearize(const std::vector<Tensor2> &f) { state = f; }
    void jacobian(const std::vector<Tensor2> &v, std::vector<Tensor2> &w) {
      w.resize(v.size());
      detail::collective_material_evaluation(comm, [&] {
        for (std::size_t i = 0; i < v.size(); ++i)
          w[i] = law.tangent_action(i, state[i], v[i]);
      });
      projector.apply(w, w);
    }
    void precondition(const std::vector<Tensor2> &r, std::vector<Tensor2> &z, int) {
      z = r;
    }
    void begin_trial(const std::vector<Tensor2> &) {}
    void accept_trial(const std::vector<Tensor2> &) {}
    void reject_trial() {}
  } model{projector, law, comm, {}};
  detail::TensorOperations ops(comm);
  InexactResult result;
  result.deformation = start.deformation;
  const Tensor2 step = add(macro, scaled(start.macroscopic, -1));
  for (auto &f : result.deformation) f = add(f, step);
  result.nonlinear = solvers::newton(ops, model, result.deformation, controls);
  result.reductions = ops.reductions;
  result.piola.resize(result.deformation.size());
  detail::collective_material_evaluation(comm, [&] {
    for (std::size_t i = 0; i < result.piola.size(); ++i)
      result.piola[i] = law.stress(i, result.deformation[i]);
  });
  return result;
}
} // namespace pfc::finite_strain
