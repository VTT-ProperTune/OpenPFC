// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file explicit_rk.hpp
 * @brief Explicit Runge-Kutta steppers for single-field and multi-field systems.
 *
 * @details
 * `ExplicitRKStepper` and `MultiExplicitRKStepper` are pluggable explicit
 * Runge-Kutta time integrators that consume `ButcherTableau<T>` coefficient
 * tableaus to implement any explicit RK method (RK2, RK4, embedded, etc.).
 *
 * Both steppers follow the same pattern as `EulerStepper`: they own `dt`,
 * internal scratch buffers, and a user-supplied `Rhs` callable that knows
 * about the spatial discretization. The stepper itself is agnostic.
 *
 * **Single-field variant** (`ExplicitRKStepper<Rhs>`):
 *   - RHS signature: `rhs(double t, std::vector<double>& u, std::vector<double>& du)`
 *   - Stores one `du` buffer and one scratch buffer per RK stage (`m_k`)
 *   - Implements the explicit RK algorithm: for each stage i, compute
 *     `k_i = rhs(t + c_i*dt, u + sum_j(a_ij*k_j))`, then final accumulation
 *     `u += dt * sum_i(b_i*k_i)`
 *
 * **Multi-field variant** (`MultiExplicitRKStepper<Rhs, N>`):
 *   - RHS signature: `rhs(double t, std::tuple<std::vector<double>&, ...> u_pack, std::tuple<std::vector<double>&, ...> du_pack)`
 *   - Stores one `du` buffer per field and one scratch buffer per field per stage
 *   - Applies the same RK algorithm to each field independently using the tuple protocol
 *
 * Factory functions mirror the `euler.hpp` pattern, binding models and
 * evaluators to the canonical `for_each_interior` driver. They capture
 * `eval` and `model` by reference (must outlive the stepper).
 *
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      point-wise driver loop
 * @see openpfc/kernel/simulation/steppers/butcher_tableau.hpp for
 *      ButcherTableau coefficient infrastructure
 * @see openpfc/kernel/simulation/steppers/euler.hpp for the forward-Euler
 *      stepper pattern
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Explicit Runge-Kutta ODE stepper for single-field systems.
 *
 * Implements the standard explicit RK algorithm with stage computation and
 * final weighted accumulation. Pre-allocates scratch buffers (`m_k`) in the
 * constructor to avoid per-step allocations.
 *
 * @tparam Rhs Any callable invocable as
 *             `rhs(double t, std::vector<double>& u, std::vector<double>& du)`.
 *             It must fill `du`; the stepper accumulates the result.
 */
template <class Rhs> class ExplicitRKStepper {
public:
  /**
   * @brief Construct an explicit RK stepper.
   *
   * @param dt Time-step size.
   * @param local_size Number of cells in the rank-local field buffer.
   * @param tableau Butcher tableau defining the RK method coefficients.
   * @param rhs RHS callable.
   */
  ExplicitRKStepper(double dt, std::size_t local_size, ButcherTableau<double> tableau, Rhs rhs)
      : m_dt(dt), m_du(local_size, 0.0), m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    const unsigned int s = m_tableau.stage_count();
    m_k.resize(s);
    for (unsigned int i = 0; i < s; ++i) {
      m_k[i].assign(local_size, 0.0);
    }
  }

  /**
   * @brief Advance `u` by one explicit RK step in place; returns the new time.
   *
   * Implements the explicit RK algorithm:
   *   1. For each stage i: compute k_i = rhs(t + c_i*dt, u + dt * sum_j(a_ij * k_j))
   *   2. Final accumulation: u += dt * sum_i(b_i * k_i)
   *
   * @param t Current time.
   * @param u State vector (modified in place).
   * @return New time `t + dt`.
   */
  double step(double t, std::vector<double>& u) {
    const unsigned int s = m_tableau.stage_count();
    const std::size_t n = u.size();

    // Compute stages
    for (unsigned int i = 0; i < s; ++i) {
      // Build temp state: u_temp = u + dt * sum_j(a_ij * k_j)
      std::vector<double> u_temp(u);
      for (unsigned int j = 0; j < i; ++j) {
        const double a_ij = m_tableau.a(i, j);
        if (a_ij != 0.0) {
          const std::ptrdiff_t np = static_cast<std::ptrdiff_t>(n);
          for (std::ptrdiff_t li = 0; li < np; ++li) {
            const std::size_t idx = static_cast<std::size_t>(li);
            u_temp[idx] += m_dt * a_ij * m_k[j][idx];
          }
        }
      }

      // Compute stage i: k_i = rhs(t + c_i * dt, u_temp)
      const double stage_time = t + m_tableau.c(i) * m_dt;
      m_rhs(stage_time, u_temp, m_du);

      // Copy du to k_i
      m_k[i] = m_du;
    }

    // Final accumulation: u += dt * sum_i(b_i * k_i)
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      if (b_i != 0.0) {
        const std::ptrdiff_t np = static_cast<std::ptrdiff_t>(n);
        for (std::ptrdiff_t li = 0; li < np; ++li) {
          const std::size_t idx = static_cast<std::size_t>(li);
          u[idx] += m_dt * b_i * m_k[i][idx];
        }
      }
    }

    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  double m_dt{0.0};
  std::vector<double> m_du;
  std::vector<std::vector<double>> m_k;  // scratch buffers per stage
  ButcherTableau<double> m_tableau;
  Rhs m_rhs;
};

/**
 * @brief Multi-field explicit Runge-Kutta ODE stepper.
 *
 * Owns one `du` buffer per field and one scratch buffer per field per RK stage.
 * Applies the same RK algorithm to each field independently using the tuple protocol.
 *
 * @tparam Rhs Multi-field RHS callable invocable as
 *             `rhs(double t, std::tuple<std::vector<double>&, ...> u_pack, std::tuple<std::vector<double>&, ...> du_pack)`.
 * @tparam N Number of fields.
 */
template <class Rhs, std::size_t N> class MultiExplicitRKStepper {
public:
  /**
   * @brief Construct a multi-field explicit RK stepper.
   *
   * @param dt Time-step size.
   * @param local_sizes Array of local sizes for each field.
   * @param tableau Butcher tableau defining the RK method coefficients.
   * @param rhs Multi-field RHS callable.
   */
  MultiExplicitRKStepper(double dt, std::array<std::size_t, N> local_sizes,
                         ButcherTableau<double> tableau, Rhs rhs)
      : m_dt(dt), m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    for (std::size_t i = 0; i < N; ++i) {
      m_du[i].assign(local_sizes[i], 0.0);
      const unsigned int s = m_tableau.stage_count();
      m_k[i].resize(s);
      for (unsigned int j = 0; j < s; ++j) {
        m_k[i][j].assign(local_sizes[i], 0.0);
      }
    }
  }

  /**
   * @brief Advance every field by one explicit RK step in place.
   *
   * @tparam U Field buffer types (typically std::vector<double>).
   * @param t Current time.
   * @param u_buffers Field buffers (modified in place).
   * @return New time `t + dt`.
   */
  template <class... U>
  double step(double t, std::vector<U>&... u_buffers) {
    static_assert(sizeof...(U) == N,
                  "MultiExplicitRKStepper::step: number of u buffers must match N.");

    const unsigned int s = m_tableau.stage_count();
    auto u_pack = std::tie(u_buffers...);

    // Compute stages for each field
    for (unsigned int i = 0; i < s; ++i) {
      // Build temp states for each field
      auto u_temp_pack = make_u_temp_tuples(u_pack, std::index_sequence_for<U...>{}, i);

      // Compute stage i for all fields
      const double stage_time = t + m_tableau.c(i) * m_dt;
      auto du_pack = make_du_tuple(std::index_sequence_for<U...>{});
      m_rhs(stage_time, u_temp_pack, du_pack);

      // Copy du to k_i for each field
      copy_du_to_k(du_pack, std::index_sequence_for<U...>{}, i);
    }

    // Final accumulation for each field
    accumulate(u_pack, std::index_sequence_for<U...>{});

    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  template <class... U, std::size_t... I>
  auto make_u_temp_tuples(std::tuple<std::vector<U>&...>& u_pack, std::index_sequence<I...>, unsigned int stage_idx) {
    return std::make_tuple(make_u_temp_one<I>(std::get<I>(u_pack), stage_idx)...);
  }

  template <std::size_t FieldIdx, class U>
  std::vector<double> make_u_temp_one(std::vector<U>& u, unsigned int stage_idx) {
    std::vector<double> u_temp(u.begin(), u.end());

    // Add contributions from previous stages: u_temp += dt * sum_j(a_ij * k_j)
    for (unsigned int j = 0; j < stage_idx; ++j) {
      const double a_ij = m_tableau.a(stage_idx, j);
      if (a_ij != 0.0) {
        const std::size_t n = u.size();
        for (std::size_t li = 0; li < n; ++li) {
          u_temp[li] += m_dt * a_ij * m_k[FieldIdx][j][li];
        }
      }
    }

    return u_temp;
  }

  template <std::size_t... I>
  auto make_du_tuple(std::index_sequence<I...>) {
    return std::tie(m_du[I]...);
  }

  template <class DuPack, std::size_t... I>
  void copy_du_to_k(DuPack& du_pack, std::index_sequence<I...>, unsigned int stage_idx) {
    ((m_k[I][stage_idx] = std::get<I>(du_pack)), ...);
  }

  template <class UPack, std::size_t... I>
  void accumulate(UPack& u_pack, std::index_sequence<I...>) {
    (accumulate_one<I>(std::get<I>(u_pack)), ...);
  }

  template <std::size_t FieldIdx, class U>
  void accumulate_one(std::vector<U>& u) {
    const unsigned int s = m_tableau.stage_count();
    const std::size_t n = u.size();
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      if (b_i != 0.0) {
        for (std::size_t li = 0; li < n; ++li) {
          u[li] += m_dt * b_i * m_k[FieldIdx][i][li];
        }
      }
    }
  }

  double m_dt{0.0};
  std::array<std::vector<double>, N> m_du;
  std::array<std::vector<std::vector<double>>, N> m_k;  // scratch per field per stage
  ButcherTableau<double> m_tableau;
  Rhs m_rhs;
};

// -----------------------------------------------------------------------------
// `create` free-function factories.
//
// They build an `ExplicitRKStepper` (single-field) or `MultiExplicitRKStepper`
// (multi-field) whose RHS is the canonical point-wise loop
//
//     du[{i,j,k}] = model.rhs(t, eval(i,j,k))
//
// over the interior cells exposed by `eval`. The stepper itself remains
// agnostic of the (Eval, Model) types — the wiring lives entirely inside the
// captured lambda below.
// -----------------------------------------------------------------------------

/**
 * @brief Build an `ExplicitRKStepper` for the canonical point-wise RHS, given
 *        the local buffer size explicitly.
 *
 * Prefer the `LocalField` overload when you have one — it derives
 * `local_size` from `u.size()`.
 *
 * @param eval Per-point gradient evaluator. Captured by reference; must outlive
 *            the returned stepper.
 * @param model Physics model with a method `rhs(double t, const G&) -> double`.
 *             Captured by reference; must outlive the returned stepper.
 * @param dt Time-step size.
 * @param local_size Number of cells in the rank-local field buffer
 *                   (typically `u.size()`).
 * @param tableau Butcher tableau defining the RK method coefficients.
 */
template <class Eval, class Model>
[[nodiscard]] auto create(Eval& eval, const Model& model, double dt,
                          std::size_t local_size, const ButcherTableau<double>& tableau) {
  auto rhs = [&eval, &model](double t, const std::vector<double>& /*u*/,
                             std::vector<double>& du) {
    pfc::sim::for_each_interior(model, eval, du.data(), t);
  };
  return ExplicitRKStepper<decltype(rhs)>(dt, local_size, tableau, std::move(rhs));
}

/**
 * @brief Build an `ExplicitRKStepper` for the canonical point-wise RHS, deriving
 *        the local buffer size from the field bundle.
 *
 * Mirrors the `world::create`, `decomposition::create`, `fft::create`,
 * `field::create` family used elsewhere in OpenPFC.
 *
 * @param u Local field whose `size()` defines the internal `du` buffer
 *          (and which the application owns). Not stored by the stepper.
 * @param eval Per-point gradient evaluator. Captured by reference.
 * @param model Physics model. Captured by reference.
 * @param dt Time-step size.
 * @param tableau Butcher tableau defining the RK method coefficients.
 */
template <class T, class Eval, class Model>
[[nodiscard]] auto create(const pfc::field::LocalField<T>& u, Eval& eval,
                          const Model& model, double dt,
                          const ButcherTableau<double>& tableau) {
  return create(eval, model, dt, u.size(), tableau);
}

/**
 * @brief Multi-field overload: build a `MultiExplicitRKStepper` from a tuple of
 *        `LocalField` references, a composite evaluator, and a model whose
 *        `rhs` returns a tuple-protocol bundle of increments.
 *
 * The composite evaluator is responsible for returning a per-point bundle the
 * model can read. The model's `rhs(t, g)` must return a tuple-protocol-compatible
 * bundle (a `std::tuple` or a struct exposing `as_tuple()`); the stepper scatters
 * the elements into the per-field `du` buffers in order.
 *
 * @param fields Tuple of `LocalField` references whose `size()` defines
 *               each per-field internal `du` buffer. The fields themselves
 *               are not stored by the stepper.
 * @param eval Composite per-point evaluator. Captured by reference.
 * @param model Multi-field physics model. Captured by reference.
 * @param dt Time-step size.
 * @param tableau Butcher tableau defining the RK method coefficients.
 */
template <class... Ts, class Eval, class Model>
[[nodiscard]] auto create(std::tuple<pfc::field::LocalField<Ts>&...> fields,
                          Eval& eval, const Model& model, double dt,
                          const ButcherTableau<double>& tableau) {
  constexpr std::size_t N = sizeof...(Ts);
  std::array<std::size_t, N> sizes{};
  std::apply(
      [&](auto&... f) {
        std::size_t i = 0;
        ((sizes[i++] = f.size()), ...);
      },
      fields);

  auto rhs = [&eval, &model](double t, auto& /*u_tuple*/, auto& du_tuple) {
    auto du_ptrs = std::apply(
        [](auto&... vs) { return std::make_tuple(vs.data()...); }, du_tuple);
    pfc::sim::for_each_interior(model, eval, du_ptrs, t);
  };
  return MultiExplicitRKStepper<decltype(rhs), N>(dt, sizes, tableau, std::move(rhs));
}

} // namespace pfc::sim::steppers
