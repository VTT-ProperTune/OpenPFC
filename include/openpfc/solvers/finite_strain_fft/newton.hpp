// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file newton.hpp
 * @brief Matrix-free Newton–CG solve of `G : P(F) = 0`.
 *
 * The loop follows Algorithm 1 of de Geus et al. for one prescribed
 * macroscopic deformation gradient. The material enters only through
 * `LocalConstitutiveLaw`: stress `P(F)` and the tangent action. The first
 * linear solve distributes `F_bar - I` with the tangent at `F = I`. Later
 * solves drive the projected stress to zero. No global matrix is assembled.
 *
 * Krylov stopping matches the published program's relative residual test:
 * `||r|| < rtol ||b||`, with a zero right-hand side returning immediately.
 * Newton stopping matches that program as well: the reference norm is
 * frozen after the macroscopic deformation is added, and the first
 * correction is not allowed to terminate the iteration.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/solvers/finite_strain_fft/constitutive.hpp>
#include <openpfc/solvers/finite_strain_fft/projector.hpp>

namespace pfc::finite_strain {

struct NewtonControls {
  /// Relative Krylov tolerance, the published program's `tol=1e-8`.
  double krylov_relative_tolerance = 1e-8;
  /// `||δF|| / ||F_ref||` threshold, the published program's `1e-5`.
  double newton_relative_tolerance = 1e-5;
  int maximum_newton_iterations = 32;
  /// `0` selects ten times the global number of tensor degrees of freedom.
  int maximum_krylov_iterations = 0;
};

struct NewtonStep {
  double correction_over_reference = 0.0;
  int krylov_iterations = 0;
  /// `||G : P||` after this update. Zero when the right-hand side was zero.
  double projected_residual = 0.0;
};

struct NewtonResult {
  std::vector<Tensor2> deformation;
  std::vector<Tensor2> piola;
  std::vector<NewtonStep> history;
  /// Frozen `||F||` after `F_bar` is added and before the first fluctuation.
  double reference_norm = 0.0;
  /// `||G : P||` at the returned field.
  double projected_residual = 0.0;
  bool converged = false;
};

namespace detail {

[[nodiscard]] inline double global_dot(const std::vector<Tensor2> &a,
                                       const std::vector<Tensor2> &b,
                                       MPI_Comm comm) {
  double local = 0.0;
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) {
    local += frobenius_dot(a[i], b[i]);
  }
  double reduced = 0.0;
  MPI_Allreduce(&local, &reduced, 1, MPI_DOUBLE, MPI_SUM, comm);
  return reduced;
}

[[nodiscard]] inline double global_norm(const std::vector<Tensor2> &a,
                                        MPI_Comm comm) {
  return std::sqrt(global_dot(a, a, comm));
}

inline void scale_add(std::vector<Tensor2> &y, double alpha,
                      const std::vector<Tensor2> &x) {
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = axpy(alpha, x[i], y[i]);
  }
}

struct KrylovOutcome {
  std::vector<Tensor2> correction;
  int iterations = 0;
  double residual = 0.0;
};

template <typename Matvec>
[[nodiscard]] KrylovOutcome conjugate_gradient(const std::vector<Tensor2> &rhs,
                                               Matvec &&matvec, double relative,
                                               int max_iterations, MPI_Comm comm) {
  KrylovOutcome out;
  const std::size_t n = rhs.size();
  out.correction.assign(n, Tensor2{});
  const double rhs_norm = global_norm(rhs, comm);
  if (rhs_norm == 0.0) {
    return out;
  }
  const double absolute = relative * rhs_norm;
  std::vector<Tensor2> residual = rhs;
  std::vector<Tensor2> direction;
  std::vector<Tensor2> product(n);
  double rho_previous = 0.0;

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const double residual_norm = global_norm(residual, comm);
    if (residual_norm < absolute) {
      out.iterations = iteration;
      out.residual = residual_norm;
      return out;
    }
    const double rho = residual_norm * residual_norm;
    if (iteration == 0) {
      direction = residual;
    } else {
      const double beta = rho / rho_previous;
      for (std::size_t i = 0; i < n; ++i) {
        direction[i] = axpy(beta, direction[i], residual[i]);
      }
    }
    matvec(direction, product);
    const double curvature = global_dot(direction, product, comm);
    if (!(std::abs(curvature) > 0.0)) {
      throw std::runtime_error(
          "finite-strain CG stopped on a zero curvature product");
    }
    const double alpha = rho / curvature;
    scale_add(out.correction, alpha, direction);
    scale_add(residual, -alpha, product);
    rho_previous = rho;
  }
  out.iterations = max_iterations;
  out.residual = global_norm(residual, comm);
  out.correction.clear();
  throw std::runtime_error("finite-strain CG reached the iteration cap");
}

} // namespace detail

/**
 * Solve `G : P(F) = 0` for one prescribed macroscopic deformation gradient.
 *
 * `law` supplies `P` and the tangent action at each inbox point. `F_bar`
 * is an arbitrary 3x3 tensor. Its mean is imposed through the zero mode.
 * The first Krylov solve still sees `F = I`, and its right-hand side is
 * the projection of the tangent acting on `F_bar - I` only.
 */
template <LocalConstitutiveLaw Law>
[[nodiscard]] inline NewtonResult
solve_equilibrium(CompatibleProjector &projector, const Law &law,
                  const Tensor2 &macroscopic, NewtonControls controls = {},
                  MPI_Comm comm = MPI_COMM_WORLD) {
  if (!(controls.krylov_relative_tolerance > 0.0) ||
      !(controls.newton_relative_tolerance > 0.0)) {
    throw std::invalid_argument("solver tolerances must be positive");
  }

  const std::size_t n = projector.local_size();
  const auto grid = domain::get_size(projector.domain());
  const long long degrees =
      9LL * static_cast<long long>(grid[0]) * grid[1] * grid[2];
  int krylov_cap = controls.maximum_krylov_iterations;
  if (krylov_cap <= 0) {
    const long long capped = degrees * 10LL;
    krylov_cap = capped > static_cast<long long>(std::numeric_limits<int>::max())
                     ? std::numeric_limits<int>::max()
                     : static_cast<int>(capped);
  }

  NewtonResult result;
  result.deformation.assign(n, identity2());
  const Tensor2 increment = add(macroscopic, scaled(identity2(), -1.0));

  auto apply_tangent = [&](const std::vector<Tensor2> &direction,
                           std::vector<Tensor2> &product) {
    if (product.size() != n) {
      product.assign(n, Tensor2{});
    }
    for (std::size_t i = 0; i < n; ++i) {
      product[i] = law.tangent_action(i, result.deformation[i], direction[i]);
    }
    projector.apply(product, product);
  };

  auto refresh_residual = [&](std::vector<Tensor2> &rhs) {
    result.piola.assign(n, Tensor2{});
    for (std::size_t i = 0; i < n; ++i) {
      result.piola[i] = law.stress(i, result.deformation[i]);
    }
    projector.apply(result.piola, rhs);
    for (Tensor2 &value : rhs) {
      value = scaled(value, -1.0);
    }
  };

  std::vector<Tensor2> macro(n, macroscopic);
  std::vector<Tensor2> load(n, increment);
  std::vector<Tensor2> rhs;
  apply_tangent(load, rhs);
  for (Tensor2 &value : rhs) {
    value = scaled(value, -1.0);
  }
  result.reference_norm = detail::global_norm(macro, comm);

  for (int newton = 0; newton < controls.maximum_newton_iterations; ++newton) {
    auto linear = detail::conjugate_gradient(
        rhs, apply_tangent, controls.krylov_relative_tolerance, krylov_cap, comm);
    if (newton == 0) {
      for (std::size_t i = 0; i < n; ++i) {
        result.deformation[i] = add(macroscopic, linear.correction[i]);
      }
    } else {
      detail::scale_add(result.deformation, 1.0, linear.correction);
    }
    refresh_residual(rhs);

    NewtonStep step;
    step.krylov_iterations = linear.iterations;
    step.correction_over_reference =
        detail::global_norm(linear.correction, comm) / result.reference_norm;
    step.projected_residual = detail::global_norm(rhs, comm);
    result.history.push_back(step);
    result.projected_residual = step.projected_residual;

    if (step.correction_over_reference < controls.newton_relative_tolerance &&
        newton > 0) {
      result.converged = true;
      return result;
    }
  }
  result.converged = false;
  return result;
}

/**
 * One load increment that continues from an already accepted deformation.
 *
 * The unknown starts at `start.deformation`. The macroscopic step
 * `F_bar - F_bar_previous` is added before the reference norm is frozen,
 * and the first Krylov solve still uses the tangent at the previous
 * deformation. Later solves use the tangent at the updated field. The law
 * is not asked to commit or reject; that decision stays with the caller.
 */
struct IncrementStart {
  std::vector<Tensor2> deformation;
  /// Macroscopic deformation that produced `deformation`. Defaults to `I`.
  Tensor2 macroscopic = identity2();
};

template <LocalConstitutiveLaw Law>
[[nodiscard]] inline NewtonResult
solve_increment(CompatibleProjector &projector, const Law &law,
                const Tensor2 &macroscopic, const IncrementStart &start,
                NewtonControls controls = {}, MPI_Comm comm = MPI_COMM_WORLD) {
  if (!(controls.krylov_relative_tolerance > 0.0) ||
      !(controls.newton_relative_tolerance > 0.0)) {
    throw std::invalid_argument("solver tolerances must be positive");
  }
  const std::size_t n = projector.local_size();
  if (start.deformation.size() != n) {
    throw std::invalid_argument(
        "incremental finite-strain solve needs one deformation per inbox point");
  }

  const auto grid = domain::get_size(projector.domain());
  const long long degrees =
      9LL * static_cast<long long>(grid[0]) * grid[1] * grid[2];
  int krylov_cap = controls.maximum_krylov_iterations;
  if (krylov_cap <= 0) {
    const long long capped = degrees * 10LL;
    krylov_cap = capped > static_cast<long long>(std::numeric_limits<int>::max())
                     ? std::numeric_limits<int>::max()
                     : static_cast<int>(capped);
  }

  const Tensor2 macro_step = add(macroscopic, scaled(start.macroscopic, -1.0));
  NewtonResult result;
  result.deformation = start.deformation;
  for (Tensor2 &value : result.deformation) {
    value = add(value, macro_step);
  }
  // The caller's field is not modified. Alias it instead of copying a
  // second 1024³ tensor into the increment.
  const std::vector<Tensor2> &tangent_state = start.deformation;
  bool tangent_from_start = true;

  auto apply_tangent = [&](const std::vector<Tensor2> &direction,
                           std::vector<Tensor2> &product) {
    if (product.size() != n) {
      product.assign(n, Tensor2{});
    }
    const std::vector<Tensor2> &state =
        tangent_from_start ? tangent_state : result.deformation;
    for (std::size_t i = 0; i < n; ++i) {
      product[i] = law.tangent_action(i, state[i], direction[i]);
    }
    projector.apply(product, product);
  };

  auto refresh_residual = [&](std::vector<Tensor2> &rhs) {
    result.piola.assign(n, Tensor2{});
    for (std::size_t i = 0; i < n; ++i) {
      result.piola[i] = law.stress(i, result.deformation[i]);
    }
    projector.apply(result.piola, rhs);
    for (Tensor2 &value : rhs) {
      value = scaled(value, -1.0);
    }
  };

  std::vector<Tensor2> rhs;
  {
    const std::vector<Tensor2> load(n, macro_step);
    apply_tangent(load, rhs);
  }
  for (Tensor2 &value : rhs) {
    value = scaled(value, -1.0);
  }
  result.reference_norm = detail::global_norm(result.deformation, comm);
  if (!(result.reference_norm > 0.0)) {
    throw std::runtime_error("incremental finite-strain reference norm vanished");
  }

  for (int newton = 0; newton < controls.maximum_newton_iterations; ++newton) {
    auto linear = detail::conjugate_gradient(
        rhs, apply_tangent, controls.krylov_relative_tolerance, krylov_cap, comm);
    detail::scale_add(result.deformation, 1.0, linear.correction);
    tangent_from_start = false;
    refresh_residual(rhs);

    NewtonStep step;
    step.krylov_iterations = linear.iterations;
    step.correction_over_reference =
        detail::global_norm(linear.correction, comm) / result.reference_norm;
    step.projected_residual = detail::global_norm(rhs, comm);
    result.history.push_back(step);
    result.projected_residual = step.projected_residual;

    if (step.correction_over_reference < controls.newton_relative_tolerance &&
        newton > 0) {
      result.converged = true;
      return result;
    }
  }
  result.converged = false;
  return result;
}

} // namespace pfc::finite_strain
