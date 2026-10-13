// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <span>

#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>

namespace pfc::sim::steppers {

/**
 * Runtime homogeneous CPU field group, using the existing explicit RK engine.
 *
 * Fields are borrowed with fixed object addresses, geometry and storage. Rates
 * are compact owned-cell arrays in x-fastest order, one array per bound field.
 * The callback observes all published stage Fields and receives the actual
 * tableau time. It owns physical preparation and rates. Accepted padding is
 * preserved, not refreshed: the callback must prepare stage ghosts every time.
 *
 * Construction, attempt, commit and step are collective on comm (SELF by
 * default); candidate access and queries are local. Callbacks must
 * finish all their own communication and collectives before throwing; divergence
 * inside a callback cannot be repaired here. Non-reentrant; the communicator and
 * Fields outlive the stepper. Identical field count, global geometry and method are
 * required across ranks; local boxes may differ. Differing layouts within a rank
 * fail. CPU double only; device, adaptive and spectral methods are not provided.
 */
class RuntimeFieldRKStepper {
  struct State;

public:
  using Field = pfc::data::Field<double, pfc::HostSpace>;
  using Fields = std::span<Field *const>;
  using Rates = std::span<const std::span<double>>;
  using StageRhs = std::function<void(double, Rates)>;
  struct Attempt : StepAttempt<double> {
  private:
    const State *owner;
    std::size_t generation;
    Attempt(const StepAttempt<double> &result, const State *state,
            std::size_t serial)
        : StepAttempt<double>(result), owner(state), generation(serial) {}
    friend class RuntimeFieldRKStepper;
  };

private:
  struct Agreement {
    MPI_Comm comm;
    template <class Fn> void run(Fn &&fn) const {
      std::exception_ptr error;
      try {
        std::forward<Fn>(fn)();
      } catch (...) {
        error = std::current_exception();
      }
      int ok = error ? 0 : 1, all = ok;
      if (comm != MPI_COMM_SELF)
        pfc::mpi::throw_on_mpi_error(
            MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, comm),
            "runtime RK admission");
      if (!all) {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("runtime RK rejected on another rank");
      }
    }
    void same_count(std::size_t value) const {
      if (comm == MPI_COMM_SELF) return;
      const auto count = static_cast<unsigned long long>(value);
      unsigned long long low = 0, high = 0;
      pfc::mpi::throw_on_mpi_error(
          MPI_Allreduce(&count, &low, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm),
          "runtime RK count minimum");
      pfc::mpi::throw_on_mpi_error(
          MPI_Allreduce(&count, &high, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm),
          "runtime RK count maximum");
      if (low != high)
        throw std::invalid_argument("runtime RK rank field/stage counts differ");
    }
    void same(double value) const {
      if (comm == MPI_COMM_SELF) return;
      double low = 0, high = 0;
      pfc::mpi::throw_on_mpi_error(
          MPI_Allreduce(&value, &low, 1, MPI_DOUBLE, MPI_MIN, comm),
          "runtime RK metadata minimum");
      pfc::mpi::throw_on_mpi_error(
          MPI_Allreduce(&value, &high, 1, MPI_DOUBLE, MPI_MAX, comm),
          "runtime RK metadata maximum");
      if (low != high)
        throw std::invalid_argument("runtime RK rank metadata differ");
    }
  };

  struct State {
    std::vector<Field *> fields;
    pfc::Domain domain;
    pfc::Int3 low, local;
    int halo;
    StageRhs rhs;
    Agreement agreement;
    std::unique_ptr<detail::RuntimeFieldStageTransaction> binding;
    std::vector<std::size_t> indices;
    std::vector<std::vector<double>> stages, baseline;
    std::vector<const std::vector<double> *> sources;
    std::vector<std::span<double>> rates;
    std::vector<double> accepted;
    bool busy = false;

    State(Fields group, StageRhs callback, MPI_Comm comm)
        : fields(group.begin(), group.end()), domain(fields.front()->domain()),
          low(fields.front()->box().low), local(fields.front()->local_size()),
          halo(fields.front()->storage_halo()), rhs(std::move(callback)),
          agreement{comm}, stages(fields.size()), baseline(fields.size()),
          sources(fields.size()), rates(fields.size()) {
      std::vector<std::vector<double> *> buffers;
      for (auto *field : fields) buffers.push_back(&field->vec());
      binding = std::make_unique<detail::RuntimeFieldStageTransaction>(buffers);
      for (int k = 0; k < local[2]; ++k)
        for (int j = 0; j < local[1]; ++j)
          for (int i = 0; i < local[0]; ++i)
            indices.push_back(fields[0]->idx(i, j, k));
      if (indices.empty() ||
          fields.size() > std::numeric_limits<std::size_t>::max() / indices.size())
        throw std::length_error("runtime RK owned size is invalid");
      accepted.resize(fields.size() * indices.size());
      for (std::size_t f = 0; f < fields.size(); ++f) {
        stages[f].resize(fields[f]->size());
        baseline[f].resize(fields[f]->size());
        sources[f] = &stages[f];
      }
    }

    void validate() const {
      binding->validate();
      for (auto *field : fields)
        if (field->domain() != domain || field->box().low != low ||
            field->local_size() != local || field->storage_halo() != halo)
          throw std::invalid_argument("runtime RK bound geometry changed");
    }
    void pack() {
      validate();
      const auto n = indices.size();
      for (std::size_t f = 0; f < fields.size(); ++f) {
        std::copy(fields[f]->vec().begin(), fields[f]->vec().end(),
                  baseline[f].begin());
        for (std::size_t i = 0; i < n; ++i) {
          const double value = fields[f]->vec()[indices[i]];
          if (!std::isfinite(value))
            throw std::invalid_argument("runtime RK accepted state is not finite");
          accepted[f * n + i] = value;
        }
      }
    }
    void evaluate(double t, const std::vector<double> &stage,
                  std::vector<double> &du) {
      // Gate all rank-local failures before any next-stage halo callback.
      agreement.run([&] {
        validate();
        if (!std::isfinite(t) || stage.size() != accepted.size())
          throw std::invalid_argument("runtime RK stage is invalid");
        for (double value : stage)
          if (!std::isfinite(value))
            throw std::invalid_argument("runtime RK stage is not finite");
        const auto n = indices.size();
        std::fill(du.begin(), du.end(), 0.);
        for (std::size_t f = 0; f < fields.size(); ++f) {
          std::copy(fields[f]->vec().begin(), fields[f]->vec().end(),
                    stages[f].begin());
          for (std::size_t i = 0; i < n; ++i)
            stages[f][indices[i]] = stage[f * n + i];
          rates[f] = std::span<double>(du).subspan(f * n, n);
        }
      });
      agreement.run([&] {
        binding->evaluate(sources, [&] {
          rhs(t, rates);
          validate();
          for (double value : du)
            if (!std::isfinite(value))
              throw std::runtime_error("runtime RK rate is not finite");
        });
      });
    }
  };
  using EngineRhs = std::function<void(double, const std::vector<double> &,
                                       std::vector<double> &)>;
  using Engine = ExplicitRKStepper<EngineRhs>;
  std::shared_ptr<State> m_state;
  std::unique_ptr<Engine> m_engine;
  const std::vector<double> *m_candidate = nullptr;
  double m_time = 0;
  std::size_t m_generation = 0;

public:
  RuntimeFieldRKStepper(Fields fields, double dt,
                        const ButcherTableau<double> &tableau, StageRhs rhs,
                        MPI_Comm comm = MPI_COMM_SELF) {
    if (comm == MPI_COMM_NULL)
      throw std::invalid_argument("runtime RK communicator is null");
    int initialized = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Initialized(&initialized),
                                 "runtime RK MPI state");
    if (comm != MPI_COMM_SELF && !initialized)
      throw std::invalid_argument("runtime RK communicator requires MPI_Init");
    Agreement agreement{comm};
    agreement.run([&] {
      if (!std::isfinite(dt) || dt <= 0 || !rhs || fields.empty())
        throw std::invalid_argument(
            "runtime RK requires fields, callback and positive finite dt");
      for (std::size_t f = 0; f < fields.size(); ++f) {
        if (!fields[f]) throw std::invalid_argument("runtime RK null field");
        for (std::size_t j = 0; j < f; ++j)
          if (fields[f] == fields[j])
            throw std::invalid_argument("runtime RK duplicate field");
        if (fields[f]->domain() != fields[0]->domain() ||
            fields[f]->box().low != fields[0]->box().low ||
            fields[f]->local_size() != fields[0]->local_size() ||
            fields[f]->storage_halo() != fields[0]->storage_halo())
          throw std::invalid_argument("runtime RK layouts differ");
        std::size_t padded = 1;
        for (int axis = 0; axis < 3; ++axis) {
          const auto &d = fields[f]->domain();
          if (!std::isfinite(d.origin[axis]) || !std::isfinite(d.spacing[axis]) ||
              d.spacing[axis] <= 0 || fields[f]->local_size()[axis] <= 0)
            throw std::invalid_argument("runtime RK geometry is invalid");
          const auto extent =
              static_cast<std::size_t>(fields[f]->padded_extent(axis));
          if (extent == 0 ||
              padded > std::numeric_limits<std::size_t>::max() / extent)
            throw std::length_error("runtime RK padded size overflow");
          padded *= extent;
        }
        if (padded != fields[f]->size())
          throw std::invalid_argument(
              "runtime RK field storage does not match geometry");
      }
    });
    agreement.same_count(fields.size());
    agreement.same(dt);
    agreement.same_count(tableau.stage_count());
    for (unsigned int i = 0; i < tableau.stage_count(); ++i) {
      agreement.same(tableau.b(i));
      agreement.same(tableau.c(i));
      for (unsigned int j = 0; j < i; ++j) agreement.same(tableau.a(i, j));
    }
    agreement.same(fields[0]->storage_halo());
    const auto &d = fields[0]->domain();
    for (int axis = 0; axis < 3; ++axis) {
      agreement.same(d.size[axis]);
      agreement.same(d.origin[axis]);
      agreement.same(d.spacing[axis]);
      agreement.same(d.periodic[axis] ? 1. : 0.);
    }
    agreement.run([&] {
      m_state = std::make_shared<State>(fields, std::move(rhs), comm);
      EngineRhs evaluate = [state = m_state](double t, const auto &u, auto &du) {
        state->evaluate(t, u, du);
      };
      m_engine = std::make_unique<Engine>(dt, m_state->accepted.size(), tableau,
                                          std::move(evaluate));
    });
  }

  [[nodiscard]] Attempt attempt(double t) {
    // Reentrant calls from a single-rank callback fail before another collective.
    if (m_state->busy) throw std::logic_error("runtime RK is non-reentrant");
    m_candidate = nullptr;
    m_state->agreement.run([&] {
      if (m_generation == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("runtime RK attempt generation overflow");
      if (!std::isfinite(t) || !std::isfinite(t + dt()))
        throw std::invalid_argument(
            "runtime RK time or reentrant attempt is invalid");
      m_state->pack();
    });
    m_state->agreement.same(t);
    ++m_generation;
    m_state->busy = true;
    struct Release {
      State &state;
      ~Release() { state.busy = false; }
    } release{*m_state};
    const auto result = m_engine->attempt(t, m_state->accepted);
    m_state->agreement.run([&] {
      for (double value : result.candidate)
        if (!std::isfinite(value))
          throw std::runtime_error("runtime RK candidate is not finite");
    });
    m_candidate = &result.candidate;
    m_time = t;
    return Attempt(result, m_state.get(), m_generation);
  }

  [[nodiscard]] std::span<const double> candidate_field(const Attempt &a,
                                                        std::size_t field) const {
    if (a.owner != m_state.get() || a.generation != m_generation ||
        &a.candidate != m_candidate || field >= m_state->fields.size())
      throw std::invalid_argument(
          "runtime RK candidate does not belong to this attempt");
    return std::span<const double>(a.candidate)
        .subspan(field * m_state->indices.size(), m_state->indices.size());
  }

  void commit(const Attempt &a) {
    if (m_state->busy) throw std::logic_error("runtime RK is non-reentrant");
    try {
      m_state->agreement.run([&] {
        m_state->validate();
        if (!a.success || a.owner != m_state.get() || a.generation != m_generation ||
            &a.candidate != m_candidate || a.t0 != m_time || a.dt != dt() ||
            a.t1 != m_time + dt() || a.candidate.size() != m_state->accepted.size())
          throw std::invalid_argument(
              "runtime RK candidate is failed, stale or already committed");
        const auto n = m_state->indices.size();
        for (std::size_t f = 0; f < m_state->fields.size(); ++f) {
          if (std::memcmp(m_state->fields[f]->data(), m_state->baseline[f].data(),
                          m_state->baseline[f].size() * sizeof(double)) != 0)
            throw std::invalid_argument(
                "runtime RK accepted buffer changed before commit");
          for (std::size_t i = 0; i < n; ++i)
            if (m_state->fields[f]->vec()[m_state->indices[i]] !=
                    m_state->accepted[f * n + i] ||
                !std::isfinite(a.candidate[f * n + i]))
              throw std::invalid_argument(
                  "runtime RK accepted state changed before commit");
        }
      });
    } catch (...) {
      m_candidate = nullptr;
      throw;
    }
    const auto n = m_state->indices.size();
    // All ranks and all fields have passed admission; writes cannot allocate.
    for (std::size_t f = 0; f < m_state->fields.size(); ++f)
      for (std::size_t i = 0; i < n; ++i)
        m_state->fields[f]->vec()[m_state->indices[i]] = a.candidate[f * n + i];
    m_candidate = nullptr;
  }

  double step(double t) {
    const auto a = attempt(t);
    commit(a);
    return a.t1;
  }
  [[nodiscard]] double dt() const noexcept { return m_engine->dt(); }
  /// Scalar scratch capacity only: excludes Fields, coefficients, callback
  /// captures and index/pointer metadata. This is not process memory usage.
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    std::size_t bytes =
        m_engine->workspace_bytes() + m_state->binding->workspace_bytes();
    bytes += m_state->accepted.capacity() * sizeof(double);
    for (const auto &buffer : m_state->stages)
      bytes += buffer.capacity() * sizeof(double);
    for (const auto &buffer : m_state->baseline)
      bytes += buffer.capacity() * sizeof(double);
    return bytes;
  }
  [[nodiscard]] std::size_t field_count() const noexcept {
    return m_state->fields.size();
  }
};

[[nodiscard]] inline auto create_runtime(RuntimeFieldRKStepper::Fields fields,
                                         double dt,
                                         const ButcherTableau<double> &tableau,
                                         RuntimeFieldRKStepper::StageRhs rhs,
                                         MPI_Comm comm = MPI_COMM_SELF) {
  return RuntimeFieldRKStepper(fields, dt, tableau, std::move(rhs), comm);
}

} // namespace pfc::sim::steppers
