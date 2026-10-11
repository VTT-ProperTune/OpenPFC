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
 * **Single-field variant** (`ExplicitRKStepper<Rhs, Scalar>`):
 *   - RHS signature: `rhs(double t, std::vector<Scalar>& u, std::vector<Scalar>&
 * du)`
 *   - Stores one `du` buffer and one scratch buffer per RK stage (`m_k`)
 *   - Implements the explicit RK algorithm: for each stage i, compute
 *     `k_i = rhs(t + c_i*dt, u + sum_j(a_ij*k_j))`, then final accumulation
 *     `u += dt * sum_i(b_i*k_i)`
 *
 * **Multi-field variant** (`MultiExplicitRKStepper<Rhs, N, Scalar>`):
 *   - RHS signature: `rhs(double t, std::tuple<std::vector<Scalar>&, ...> u_pack,
 * std::tuple<std::vector<Scalar>&, ...> du_pack)`
 *   - Stores one `du` buffer per field and one scratch buffer per field per stage
 *   - Applies the same RK algorithm to each field independently using the tuple
 * protocol
 *
 * Factory functions mirror the `euler.hpp` pattern, binding models and
 * evaluators to the canonical `for_each_interior` driver. They capture
 * `eval` and `model` by reference (must outlive the stepper). Field factories
 * also borrow mutable Fields, all of which must outlive the stepper. Their
 * evaluator must already be bound to these same fields and layouts.
 *
 * Stage preparation publishes values into the existing field allocations so
 * bound halo callbacks, FD views and spectral FFT sources all observe the
 * current stage. Each RHS performs three full-buffer copies per field:
 * accepted-to-backup, stage-to-field, and backup-to-accepted. Backups and stage
 * buffers are allocated once at construction. Field storage must not resize
 * or change address; preparation callbacks must obey the same constraint.
 * These borrowed factories are non-reentrant. Accepted values, including
 * padding, are restored after normal evaluation or preparation exceptions.
 * Pointwise physics must not throw: exceptions escaping the canonical OpenMP
 * loop may terminate execution instead of propagating to the stage transaction.
 * Derivative caches then describe the last evaluated stage: direct evaluator
 * use after a step requires prepare() again for the accepted state.
 *
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      point-wise driver loop
 * @see openpfc/kernel/simulation/steppers/butcher_tableau.hpp for
 *      ButcherTableau coefficient infrastructure
 * @see openpfc/kernel/simulation/steppers/euler.hpp for the forward-Euler
 *      stepper pattern
 * @see openpfc/runtime/gpu/explicit_rk_gpu.hpp for the device-resident
 *      two-field factory on `Field<double, CUDASpace>` / `HIPSpace`
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <openpfc/kernel/simulation/state_concepts.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Explicit Runge-Kutta ODE stepper for single-field systems.
 *
 * Implements the standard explicit RK algorithm with stage computation and
 * final weighted accumulation. Pre-allocates scratch buffers (`m_k`) in the
 * constructor to avoid per-step allocations.
 *
 * @tparam Rhs    Any callable invocable as
 *                `rhs(double t, std::vector<Scalar>& u, std::vector<Scalar>& du)`.
 *                It must fill `du`; the stepper accumulates the result.
 * @tparam Scalar Field element type (`double` or `std::complex<double>`).
 */
template <class Rhs, class Scalar = double>
  requires StageFunctionFor<Rhs, Scalar>
class ExplicitRKStepper {
public:
  using scalar_type = Scalar;
  using Attempt = StepAttempt<Scalar>;

  /**
   * @brief Construct an explicit RK stepper.
   *
   * @param dt Time-step size.
   * @param local_size Number of cells in the rank-local field buffer.
   * @param tableau Butcher tableau defining the RK method coefficients.
   * @param rhs RHS callable.
   */
  ExplicitRKStepper(double dt, std::size_t local_size,
                    ButcherTableau<double> tableau, Rhs rhs)
      : m_dt(dt), m_local_size(local_size), m_du(local_size, Scalar{}),
        m_u_temp(local_size, Scalar{}), m_candidate(local_size, Scalar{}),
        m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    const unsigned int s = m_tableau.stage_count();
    m_k.resize(s);
    for (unsigned int i = 0; i < s; ++i) {
      m_k[i].assign(local_size, Scalar{});
    }
  }

  /**
   * @brief Isolate `u + dt * sum_i(b_i * k_i)` without writing accepted `u`.
   *
   * Implements the explicit RK algorithm:
   *   1. For each stage i: k_i = rhs(t + c_i*dt, u + dt * sum_j(a_ij * k_j))
   *   2. Candidate: u + dt * sum_i(b_i * k_i)
   */
  [[nodiscard]] Attempt attempt(double t, const std::vector<Scalar> &u) {
    const unsigned int s = m_tableau.stage_count();
    const std::size_t n = u.size();
    if (n != m_local_size || m_du.size() != m_local_size ||
        m_u_temp.size() != m_local_size) {
      throw std::invalid_argument("ExplicitRKStepper: input size changed");
    }

    for (unsigned int i = 0; i < s; ++i) {
      m_u_temp = u;
      for (unsigned int j = 0; j < i; ++j) {
        const double a_ij = m_tableau.a(i, j);
        if (a_ij != 0.0) {
          const Scalar scale = Scalar(m_dt * a_ij);
          for (std::size_t idx = 0; idx < n; ++idx) {
            m_u_temp[idx] += scale * m_k[j][idx];
          }
        }
      }

      const double stage_time = t + m_tableau.c(i) * m_dt;
      m_rhs(stage_time, m_u_temp, m_du);
      if (m_du.size() != n || m_u_temp.size() != n) {
        throw std::invalid_argument(
            "ExplicitRKStepper: RHS stage/output size changed");
      }
      m_k[i] = m_du;
    }

    m_candidate = u;
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      if (b_i != 0.0) {
        const Scalar scale = Scalar(m_dt * b_i);
        for (std::size_t idx = 0; idx < n; ++idx) {
          m_candidate[idx] += scale * m_k[i][idx];
        }
      }
    }
    return Attempt(t, m_dt, t + m_dt, /*success=*/true, m_candidate);
  }

  /** Advance `u` by one explicit RK step; commit of `attempt`. */
  double step(double t, std::vector<Scalar> &u) {
    const Attempt r = attempt(t, u);
    commit_step_attempt(u, r);
    return r.t1;
  }

  double dt() const noexcept { return m_dt; }

  /** Isolate a candidate from host field state (via `vec()`). */
  template <pfc::field::HostFieldState<Scalar> F>
  [[nodiscard]] Attempt attempt(double t, const F &u) {
    return attempt(t, u.vec());
  }

  /** Advance host field state by one explicit RK step. */
  template <pfc::field::HostFieldState<Scalar> F> double step(double t, F &u) {
    return step(t, u.vec());
  }

private:
  double m_dt{0.0};
  std::size_t m_local_size;
  std::vector<Scalar> m_du;
  std::vector<Scalar> m_u_temp;
  std::vector<Scalar> m_candidate;
  std::vector<std::vector<Scalar>> m_k; // scratch buffers per stage
  ButcherTableau<double> m_tableau;
  Rhs m_rhs;
};

/**
 * @brief Multi-field explicit Runge-Kutta ODE stepper.
 *
 * Owns one `du` buffer per field and one scratch buffer per field per RK stage.
 * Applies the same RK algorithm to each field independently using the tuple
 * protocol.
 *
 * @tparam Rhs    Multi-field RHS callable invocable as
 *                `rhs(double t, std::tuple<std::vector<Scalar>&, ...> u_pack,
 * std::tuple<std::vector<Scalar>&, ...> du_pack)`.
 * @tparam N      Number of fields.
 * @tparam Scalar Field element type (`double` or `std::complex<double>`).
 */
template <class Rhs, std::size_t N, class Scalar = double>
class MultiExplicitRKStepper {
public:
  using scalar_type = Scalar;
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
      : m_dt(dt), m_local_sizes(local_sizes), m_tableau(std::move(tableau)),
        m_rhs(std::move(rhs)) {
    for (std::size_t i = 0; i < N; ++i) {
      m_du[i].assign(local_sizes[i], Scalar{});
      m_u_temp[i].assign(local_sizes[i], Scalar{});
      const unsigned int s = m_tableau.stage_count();
      m_k[i].resize(s);
      for (unsigned int j = 0; j < s; ++j) {
        m_k[i][j].assign(local_sizes[i], Scalar{});
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
  template <class... U> double step(double t, std::vector<U> &...u_buffers) {
    static_assert(sizeof...(U) == N,
                  "MultiExplicitRKStepper::step: number of u buffers must match N.");
    static_assert((std::is_same_v<U, Scalar> && ...),
                  "MultiExplicitRKStepper requires std::vector<Scalar>");

    const unsigned int s = m_tableau.stage_count();
    auto u_pack = std::tie(u_buffers...);
    validate_sizes(u_pack, std::index_sequence_for<U...>{});

    // Compute stages for each field
    for (unsigned int i = 0; i < s; ++i) {
      // Build temp states for each field
      auto u_temp_pack =
          make_u_temp_tuples(u_pack, std::index_sequence_for<U...>{}, i);

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

  /** Advance every host field by one explicit RK step (via `vec()`). */
  template <pfc::field::HostFieldState<Scalar>... Fs>
    requires(sizeof...(Fs) == N)
  double step(double t, Fs &...u_buffers) {
    return step(t, u_buffers.vec()...);
  }

  double dt() const noexcept { return m_dt; }

private:
  template <class... U, std::size_t... I>
  auto make_u_temp_tuples(std::tuple<std::vector<U> &...> &u_pack,
                          std::index_sequence<I...>, unsigned int stage_idx) {
    return std::tie(make_u_temp_one<I>(std::get<I>(u_pack), stage_idx)...);
  }

  template <std::size_t FieldIdx, class U>
  std::vector<Scalar> &make_u_temp_one(std::vector<U> &u, unsigned int stage_idx) {
    auto &u_temp = m_u_temp[FieldIdx];
    std::copy(u.begin(), u.end(), u_temp.begin());

    // Add contributions from previous stages: u_temp += dt * sum_j(a_ij * k_j)
    for (unsigned int j = 0; j < stage_idx; ++j) {
      const double a_ij = m_tableau.a(stage_idx, j);
      if (a_ij != 0.0) {
        const Scalar scale = Scalar(m_dt * a_ij);
        const std::size_t n = u.size();
        for (std::size_t li = 0; li < n; ++li) {
          u_temp[li] += scale * m_k[FieldIdx][j][li];
        }
      }
    }

    return u_temp;
  }

  template <std::size_t... I> auto make_du_tuple(std::index_sequence<I...>) {
    return std::tie(m_du[I]...);
  }

  template <class DuPack, std::size_t... I>
  void copy_du_to_k(DuPack &du_pack, std::index_sequence<I...>,
                    unsigned int stage_idx) {
    if ((((std::get<I>(du_pack).size() != m_local_sizes[I]) ||
          (m_u_temp[I].size() != m_local_sizes[I])) ||
         ...)) {
      throw std::invalid_argument("MultiExplicitRKStepper: RHS output size changed");
    }
    ((m_k[I][stage_idx] = std::get<I>(du_pack)), ...);
  }

  template <class UPack, std::size_t... I>
  void validate_sizes(const UPack &u_pack, std::index_sequence<I...>) const {
    if ((((std::get<I>(u_pack).size() != m_local_sizes[I]) ||
          (m_du[I].size() != m_local_sizes[I]) ||
          (m_u_temp[I].size() != m_local_sizes[I])) ||
         ...)) {
      throw std::invalid_argument("MultiExplicitRKStepper: input size changed");
    }
  }

  template <class UPack, std::size_t... I>
  void accumulate(UPack &u_pack, std::index_sequence<I...>) {
    (accumulate_one<I>(std::get<I>(u_pack)), ...);
  }

  template <std::size_t FieldIdx, class U> void accumulate_one(std::vector<U> &u) {
    const unsigned int s = m_tableau.stage_count();
    const std::size_t n = u.size();
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      if (b_i != 0.0) {
        const Scalar scale = Scalar(m_dt * b_i);
        for (std::size_t li = 0; li < n; ++li) {
          u[li] += scale * m_k[FieldIdx][i][li];
        }
      }
    }
  }

  double m_dt{0.0};
  std::array<std::size_t, N> m_local_sizes;
  std::array<std::vector<Scalar>, N> m_du;
  std::array<std::vector<Scalar>, N> m_u_temp;
  std::array<std::vector<std::vector<Scalar>>, N> m_k; // scratch per field per stage
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

namespace detail {

// Keep evaluator views and already-bound halo callbacks valid by preserving
// the Field allocations. All fields contain stage values during preparation
// and evaluation; accepted contents are restored on normal or exceptional exit.
template <std::size_t N> class FieldStageTransaction {
public:
  using Buffers = std::array<std::vector<double> *, N>;
  using Sources = std::array<const std::vector<double> *, N>;
  explicit FieldStageTransaction(Buffers fields) : m_fields(fields) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        if (fields[i] == fields[j]) {
          throw std::invalid_argument(
              "RK stage binding: duplicate field references");
        }
      }
      m_sizes[i] = fields[i]->size();
      m_addresses[i] = fields[i]->data();
      m_backup[i].resize(m_sizes[i]);
    }
  }

  void validate() const {
    for (std::size_t i = 0; i < N; ++i) {
      if (m_fields[i]->size() != m_sizes[i] ||
          m_fields[i]->data() != m_addresses[i]) {
        throw std::invalid_argument("RK stage binding: bound field storage changed; "
                                    "recreate evaluator and stepper");
      }
    }
  }

  template <class Fn> void evaluate(const Sources &stage, Fn &&fn) {
    // Validate every buffer before touching any bound accepted storage.
    validate();
    for (std::size_t i = 0; i < N; ++i) {
      if (stage[i]->size() != m_sizes[i]) {
        throw std::invalid_argument("RK stage binding: field storage size changed");
      }
    }
    for (std::size_t i = 0; i < N; ++i)
      std::copy(m_fields[i]->begin(), m_fields[i]->end(), m_backup[i].begin());
    struct Restore {
      FieldStageTransaction &binding;
      ~Restore() { binding.restore(); }
    } restore{*this};
    for (std::size_t i = 0; i < N; ++i)
      std::copy(stage[i]->begin(), stage[i]->end(), m_fields[i]->begin());
    std::forward<Fn>(fn)();
    validate();
  }

private:
  void restore() noexcept {
    for (std::size_t i = 0; i < N; ++i) {
      if (m_fields[i]->size() != m_sizes[i] ||
          m_fields[i]->data() != m_addresses[i]) {
        // A malformed preparation callback broke the fixed-storage contract.
        // Restore accepted contents without an out-of-range copy/allocation;
        // validate() fails on subsequent use until the caller rebuilds bindings.
        m_fields[i]->swap(m_backup[i]);
      } else {
        std::copy(m_backup[i].begin(), m_backup[i].end(), m_fields[i]->begin());
      }
    }
  }
  Buffers m_fields;
  std::array<std::vector<double>, N> m_backup;
  std::array<std::size_t, N> m_sizes;
  std::array<double *, N> m_addresses;
};

// Validate preparation before any stencil reads. This is only a checked
// forwarding adapter for the existing per-point evaluator/prepare protocol.
template <class Eval, class Binding> struct CheckedStageEvaluator {
  Eval &eval;
  Binding &binding;
  void prepare() {
    eval.prepare();
    binding.validate();
  }
  auto operator()(int i, int j, int k) const { return eval(i, j, k); }
  auto idx(int i, int j, int k) const { return eval.idx(i, j, k); }
  int imin() const { return eval.imin(); }
  int imax() const { return eval.imax(); }
  int jmin() const { return eval.jmin(); }
  int jmax() const { return eval.jmax(); }
  int kmin() const { return eval.kmin(); }
  int kmax() const { return eval.kmax(); }
};
} // namespace detail

/**
 * @brief Build an `ExplicitRKStepper` for the canonical point-wise RHS, given
 *        the local buffer size explicitly.
 *
 * Prefer the `Field` overload when you have one — it derives
 * `local_size` from `u.size()`. This raw-size overload requires an explicit
 * `eval.bind_stage(std::vector<double>&)` operation before prepare(). It may
 * leave eval bound to stepper-owned scratch; rebind before independent use or
 * destruction of the stepper. Evaluators without this contract fail explicitly.
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
[[nodiscard]] auto create(Eval &eval, const Model &model, double dt,
                          std::size_t local_size,
                          const ButcherTableau<double> &tableau) {
  if constexpr (!requires(std::vector<double> &stage) { eval.bind_stage(stage); }) {
    throw std::invalid_argument(
        "RK create(eval,...): evaluator must explicitly bind_stage(vector); "
        "use create(field,eval,...) for existing bound field evaluators");
  }
  auto rhs = [&eval, &model](double t, std::vector<double> &u,
                             std::vector<double> &du) {
    if constexpr (requires { eval.bind_stage(u); }) {
      eval.bind_stage(u);
      std::fill(du.begin(), du.end(), 0.0);
      pfc::sim::for_each_interior(model, eval, du.data(), t);
    }
  };
  return ExplicitRKStepper<decltype(rhs)>(dt, local_size, tableau, std::move(rhs));
}

/**
 * @brief Build an `ExplicitRKStepper` for the canonical point-wise RHS, deriving
 *        the local buffer size from the field bundle.
 *
 * Mirrors the `domain::create` and `pfc::data::field_from_subdomain`
 * pattern used elsewhere in OpenPFC.
 *
 * @param u Local field whose `size()` defines the internal `du` buffer
 *          (and which the application owns). Borrowed by the stepper; its
 *          fixed storage and the evaluator/model must outlive the stepper.
 * @param eval Per-point gradient evaluator. Captured by reference.
 * @param model Physics model. Captured by reference.
 * @param dt Time-step size.
 * @param tableau Butcher tableau defining the RK method coefficients.
 */
template <class Eval, class Model>
[[nodiscard]] auto create(pfc::data::Field<double> &u, Eval &eval,
                          const Model &model, double dt,
                          const ButcherTableau<double> &tableau) {
  auto rhs = [&eval, &model, offset = u.idx(0, 0, 0),
              binding = detail::FieldStageTransaction<1>({&u.vec()})](
                 double t, const std::vector<double> &stage,
                 std::vector<double> &du) mutable {
    std::fill(du.begin(), du.end(), 0.0);
    binding.evaluate({&stage}, [&] {
      detail::CheckedStageEvaluator checked{eval, binding};
      pfc::sim::for_each_interior(model, checked, du.data() + offset, t);
    });
  };
  return ExplicitRKStepper<decltype(rhs)>(dt, u.size(), tableau, std::move(rhs));
}

/**
 * @brief Multi-field overload: build a `MultiExplicitRKStepper` from a tuple of
 *        `Field` references, a composite evaluator, and a model whose
 *        `rhs` returns a tuple-protocol bundle of increments.
 *
 * The composite evaluator is responsible for returning a per-point bundle the
 * model can read. The model's `rhs(t, g)` must return a tuple-protocol-compatible
 * bundle (a `std::tuple` or a struct exposing `as_tuple()`); the stepper scatters
 * the elements into the per-field `du` buffers in order.
 *
 * @param fields Tuple of `Field` references whose `size()` defines
 *               each per-field internal `du` buffer. These mutable fields
 *               are borrowed and must outlive the stepper with fixed storage.
 *               Aliased fields and differing layouts are rejected.
 * @param eval Composite per-point evaluator. Captured by reference.
 * @param model Multi-field physics model. Captured by reference.
 * @param dt Time-step size.
 * @param tableau Butcher tableau defining the RK method coefficients.
 */
template <class... Ts, class Eval, class Model>
[[nodiscard]] auto create(std::tuple<pfc::data::Field<Ts> &...> fields, Eval &eval,
                          const Model &model, double dt,
                          const ButcherTableau<double> &tableau) {
  constexpr std::size_t N = sizeof...(Ts);
  static_assert(N >= 1, "RK multi-field factory needs at least one field");
  static_assert((std::is_same_v<Ts, double> && ...),
                "RK pointwise field factory requires real double fields");
  std::array<std::size_t, N> sizes{};
  std::array<std::size_t, N> offsets{};
  typename detail::FieldStageTransaction<N>::Buffers buffers{};
  const auto &first = std::get<0>(fields);
  std::apply(
      [&](auto &...f) {
        std::size_t i = 0;
        auto bind = [&](auto &field) {
          if (field.domain() != first.domain() ||
              field.box().low != first.box().low ||
              field.local_size() != first.local_size() ||
              field.storage_halo() != first.storage_halo()) {
            throw std::invalid_argument("RK multi-field factory: layouts differ");
          }
          sizes[i] = field.size();
          offsets[i] = field.idx(0, 0, 0);
          buffers[i++] = &field.vec();
        };
        (bind(f), ...);
      },
      fields);

  auto rhs = [&eval, &model, offsets,
              binding = detail::FieldStageTransaction<N>(buffers)](
                 double t, auto &u_tuple, auto &du_tuple) mutable {
    std::apply([](auto &...du) { (std::fill(du.begin(), du.end(), 0.0), ...); },
               du_tuple);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto du_ptrs = std::make_tuple(std::get<I>(du_tuple).data() + offsets[I]...);
      binding.evaluate({&std::get<I>(u_tuple)...}, [&] {
        detail::CheckedStageEvaluator checked{eval, binding};
        pfc::sim::for_each_interior(model, checked, du_ptrs, t);
      });
    }(std::make_index_sequence<N>{});
  };
  return MultiExplicitRKStepper<decltype(rhs), N>(dt, sizes, tableau,
                                                  std::move(rhs));
}

} // namespace pfc::sim::steppers
