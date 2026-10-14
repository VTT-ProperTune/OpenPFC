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
 * One padded snapshot per field is the rollback copy. Owned stage values are
 * published into that field and the snapshot is restored after the callback,
 * including when the callback throws.
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
    std::vector<std::size_t> sizes;
    std::vector<double *> addresses;
    std::vector<std::vector<double>> baseline;
    std::vector<std::span<double>> rates;
    std::vector<double> accepted;
    bool busy = false;

    State(Fields group, StageRhs callback, MPI_Comm comm)
        : fields(group.begin(), group.end()), domain(fields.front()->domain()),
          low(fields.front()->box().low), local(fields.front()->local_size()),
          halo(fields.front()->storage_halo()), rhs(std::move(callback)),
          agreement{comm}, sizes(fields.size()), addresses(fields.size()),
          baseline(fields.size()), rates(fields.size()) {
      const auto n = owned_count();
      if (n == 0 || fields.size() > std::numeric_limits<std::size_t>::max() / n)
        throw std::length_error("runtime RK owned size is invalid");
      accepted.resize(fields.size() * n);
      for (std::size_t f = 0; f < fields.size(); ++f) {
        sizes[f] = fields[f]->size();
        addresses[f] = fields[f]->data();
        baseline[f].resize(sizes[f]);
      }
    }

    [[nodiscard]] std::size_t owned_count() const noexcept {
      return static_cast<std::size_t>(local[0]) *
             static_cast<std::size_t>(local[1]) * static_cast<std::size_t>(local[2]);
    }

    template <class Fn> void for_owned_rows(Fn &&fn) const {
      const int nx = local[0];
      const int ny = local[1];
      const int nz = local[2];
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          const auto src = fields[0]->idx(0, j, k);
          const auto dst =
              static_cast<std::size_t>((static_cast<std::size_t>(k) * ny + j) * nx);
          fn(src, dst, nx);
        }
      }
    }

    void validate() const {
      for (std::size_t f = 0; f < fields.size(); ++f) {
        if (fields[f]->size() != sizes[f] || fields[f]->data() != addresses[f])
          throw std::invalid_argument(
              "runtime RK bound field storage changed; recreate the stepper");
        if (fields[f]->domain() != domain || fields[f]->box().low != low ||
            fields[f]->local_size() != local || fields[f]->storage_halo() != halo)
          throw std::invalid_argument("runtime RK bound geometry changed");
      }
    }
    void restore_accepted() noexcept {
      for (std::size_t f = 0; f < fields.size(); ++f) {
        auto &buffer = fields[f]->vec();
        if (buffer.size() != baseline[f].size() || buffer.data() != addresses[f]) {
          // A callback broke the fixed-storage contract. Put the accepted
          // bytes back without allocating; validate() then fails closed.
          buffer.swap(baseline[f]);
        } else {
          std::memcpy(buffer.data(), baseline[f].data(),
                      baseline[f].size() * sizeof(double));
        }
      }
    }
    void publish(const std::vector<double> &stage) {
      const auto n = owned_count();
      for (std::size_t f = 0; f < fields.size(); ++f) {
        for_owned_rows([&](std::size_t src, std::size_t dst, int nx) {
          std::memcpy(fields[f]->data() + src, stage.data() + f * n + dst,
                      static_cast<std::size_t>(nx) * sizeof(double));
        });
      }
    }
    void pack() {
      validate();
      const auto n = owned_count();
      for (std::size_t f = 0; f < fields.size(); ++f) {
        std::memcpy(baseline[f].data(), fields[f]->data(),
                    sizes[f] * sizeof(double));
        for_owned_rows([&](std::size_t src, std::size_t dst, int nx) {
          const double *row = fields[f]->data() + src;
          double *out = accepted.data() + f * n + dst;
          for (int i = 0; i < nx; ++i) {
            if (!std::isfinite(row[i]))
              throw std::invalid_argument("runtime RK accepted state is not finite");
            out[i] = row[i];
          }
        });
      }
    }
    void evaluate(double t, const std::vector<double> &stage,
                  std::vector<double> &du) {
      // Gate rank-local stage metadata before any field is published.
      agreement.run([&] {
        validate();
        if (!std::isfinite(t) || stage.size() != accepted.size())
          throw std::invalid_argument("runtime RK stage is invalid");
        for (double value : stage)
          if (!std::isfinite(value))
            throw std::invalid_argument("runtime RK stage is not finite");
      });
      agreement.run([&] {
        validate();
        struct Restore {
          State &self;
          ~Restore() { self.restore_accepted(); }
        } restore{*this};
        publish(stage);
        std::fill(du.begin(), du.end(), 0.);
        const auto n = owned_count();
        for (std::size_t f = 0; f < fields.size(); ++f)
          rates[f] = std::span<double>(du).subspan(f * n, n);
        rhs(t, rates);
        validate();
        for (double value : du)
          if (!std::isfinite(value))
            throw std::runtime_error("runtime RK rate is not finite");
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
        .subspan(field * m_state->owned_count(), m_state->owned_count());
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
        const auto n = m_state->owned_count();
        for (std::size_t f = 0; f < m_state->fields.size(); ++f) {
          if (std::memcmp(m_state->fields[f]->data(), m_state->baseline[f].data(),
                          m_state->baseline[f].size() * sizeof(double)) != 0)
            throw std::invalid_argument(
                "runtime RK accepted buffer changed before commit");
          m_state->for_owned_rows([&](std::size_t src, std::size_t dst, int nx) {
            const double *row = m_state->fields[f]->data() + src;
            for (int i = 0; i < nx; ++i)
              if (row[i] !=
                      m_state->accepted[f * n + dst + static_cast<std::size_t>(i)] ||
                  !std::isfinite(
                      a.candidate[f * n + dst + static_cast<std::size_t>(i)]))
                throw std::invalid_argument(
                    "runtime RK accepted state changed before commit");
          });
        }
      });
    } catch (...) {
      m_candidate = nullptr;
      throw;
    }
    const auto n = m_state->owned_count();
    // All ranks and all fields have passed admission; writes cannot allocate.
    for (std::size_t f = 0; f < m_state->fields.size(); ++f)
      m_state->for_owned_rows([&](std::size_t src, std::size_t dst, int nx) {
        std::memcpy(m_state->fields[f]->data() + src,
                    a.candidate.data() + f * n + dst,
                    static_cast<std::size_t>(nx) * sizeof(double));
      });
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
    std::size_t bytes = m_engine->workspace_bytes();
    bytes += m_state->accepted.capacity() * sizeof(double);
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
