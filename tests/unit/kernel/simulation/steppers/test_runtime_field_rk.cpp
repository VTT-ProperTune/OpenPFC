// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/steppers/runtime_field_rk.hpp>

namespace {
using namespace pfc::sim::steppers;
using Field = RuntimeFieldRKStepper::Field;
using Rates = RuntimeFieldRKStepper::Rates;

int world_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}
int world_size() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}
struct Group {
  int rank = world_rank(), ranks = world_size();
  pfc::Domain domain = pfc::domain::create({32, 2, 2});
  pfc::decomposition::Decomposition decomposition{domain, pfc::Int3{ranks, 1, 1}};
  std::vector<Field> fields;
  std::vector<Field *> refs;
  std::vector<std::unique_ptr<pfc::comm::HaloExchange<pfc::HostSpace, double>>>
      halos;
  explicit Group(std::size_t count, int width = 1) {
    fields.reserve(count);
    for (std::size_t f = 0; f < count; ++f) {
      fields.push_back(
          pfc::data::field_from_subdomain<double>(decomposition, rank, width));
      std::fill(fields.back().vec().begin(), fields.back().vec().end(), -12345.);
    }
    for (auto &field : fields) {
      refs.push_back(&field);
      halos.push_back(
          std::make_unique<pfc::comm::HaloExchange<pfc::HostSpace, double>>(
              field, decomposition, rank, MPI_COMM_WORLD));
    }
  }
  void owned(double value) {
    const auto n = fields.front().local_size();
    for (auto &f : fields)
      for (int k = 0; k < n[2]; ++k)
        for (int j = 0; j < n[1]; ++j)
          for (int i = 0; i < n[0]; ++i) f(i, j, k) = value;
  }
  auto snapshot() const {
    std::vector<std::vector<double>> out;
    for (const auto &field : fields) out.push_back(field.vec());
    return out;
  }
  void same(const std::vector<std::vector<double>> &state) const {
    for (std::size_t f = 0; f < fields.size(); ++f) {
      REQUIRE(fields[f].size() == state[f].size());
      REQUIRE(std::memcmp(fields[f].data(), state[f].data(),
                          state[f].size() * sizeof(double)) == 0);
    }
  }
  void exchange() {
    for (auto &h : halos) h->exchange();
  }
};

// Exact solution of the fixed FD2 semi-discrete system, not a refined trajectory.
// w'=-gamma*w+a*sin(t), w(0)=d. All field coefficients sum to zero for N>1.
double heat_error(std::size_t count, const ButcherTableau<double> &method,
                  double dt) {
  Group group(count);
  const auto n = group.fields[0].local_size();
  const double wave = 2 * std::numbers::pi / 32;
  const double lambda = 2 * std::cos(wave) - 2;
  const double gamma = (count == 1 ? 0. : 1.) - lambda;
  auto coefficient = [&](std::size_t f) {
    return count == 1 ? 1. : double(f) - double(count - 1) / 2;
  };
  for (std::size_t f = 0; f < count; ++f)
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          group.fields[f](i, j, k) =
              3. + 0.2 * coefficient(f) / (count + 1) *
                       std::cos(wave * group.fields[f].global(i, j, k)[0]);
  auto rhs = [&](double time, Rates rates) {
    group.exchange();
    std::size_t cell = 0;
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i, ++cell) {
          double mean = 0;
          for (auto &field : group.fields) mean += field(i, j, k) / count;
          const double forcing =
              std::sin(time) * std::cos(wave * group.fields[0].global(i, j, k)[0]);
          for (std::size_t f = 0; f < count; ++f) {
            const auto &u = group.fields[f];
            rates[f][cell] = u(i - 1, j, k) - 2 * u(i, j, k) + u(i + 1, j, k) +
                             (count == 1 ? 0. : mean - u(i, j, k)) +
                             coefficient(f) * forcing;
          }
        }
  };
  auto rk = create_runtime(group.refs, dt, method, rhs, MPI_COMM_WORLD);
  REQUIRE(rk.field_count() == count);
  const int steps = static_cast<int>(std::lround(1. / dt));
  for (int step = 0; step < steps; ++step) rk.step(step * dt);
  double error = 0, inventory = 0;
  for (std::size_t f = 0; f < count; ++f) {
    const double a = coefficient(f), d = 0.2 * a / (count + 1);
    const double amplitude =
        std::exp(-gamma) * d +
        a * (gamma * std::sin(1.) - std::cos(1.) + std::exp(-gamma)) /
            (gamma * gamma + 1);
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const double exact =
              3. + amplitude * std::cos(wave * group.fields[f].global(i, j, k)[0]);
          error = std::max(error, std::abs(group.fields[f](i, j, k) - exact));
          inventory += group.fields[f](i, j, k);
        }
  }
  double global_error = 0, total = 0;
  MPI_Allreduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&inventory, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::abs(total - 3. * count * 32 * 2 * 2) < 1e-10);
  return global_error;
}
} // namespace

TEST_CASE("runtime field RK preserves coupled invariants and analytic orders",
          "[runtime_field_rk][MPI]") {
  for (std::size_t count : {1u, 2u, 3u, 8u, 9u, 16u, 17u}) {
    for (int order : {1, 2, 4}) {
      const auto method = order == 1   ? ButcherTableau<double>(1, {0.}, {1.}, {0.})
                          : order == 2 ? make_rk2_midpoint<double>()
                                       : make_rk4_classical<double>();
      const double coarse = heat_error(count, method, 0.2),
                   fine = heat_error(count, method, 0.1);
      INFO("fields=" << count << " order=" << order << " coarse=" << coarse
                     << " fine=" << fine);
      const double ratio = coarse / fine;
      REQUIRE(ratio > std::pow(2., order) * 0.8);
      REQUIRE(ratio < std::pow(2., order) * 1.3);
    }
  }
}

TEST_CASE("runtime field RK publishes actual stage time and preserves padding",
          "[runtime_field_rk][MPI]") {
  Group group(17);
  group.owned(2.);
  // NaN padding remains bitwise accepted state; stage preparation refreshes it.
  group.fields[0](-1, 0, 0) = std::numeric_limits<double>::quiet_NaN();
  const auto before = group.snapshot();
  std::vector<double> times;
  auto rhs = [&](double t, Rates rates) {
    times.push_back(t);
    group.exchange();
    for (auto rate : rates) std::fill(rate.begin(), rate.end(), t * t * t);
  };
  auto rk = create_runtime(group.refs, 0.2, make_rk4_classical<double>(), rhs,
                           MPI_COMM_WORLD);
  auto moved = std::move(rk);
  const auto attempt = moved.attempt(0.7);
  group.same(before);
  REQUIRE(times.size() == 4);
  const double expected_times[] = {0.7, 0.8, 0.8, 0.9};
  for (int i = 0; i < 4; ++i)
    REQUIRE(std::abs(times[i] - expected_times[i]) < 2e-15);
  const double exact = 2. + (std::pow(0.9, 4) - std::pow(0.7, 4)) / 4;
  for (std::size_t f = 0; f < 17; ++f)
    for (double v : moved.candidate_field(attempt, f))
      REQUIRE(std::abs(v - exact) < 3e-15);
  moved.commit(attempt);
  for (std::size_t f = 0; f < 17; ++f) {
    auto accepted = group.fields[f].vec();
    const auto n = group.fields[f].local_size();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          accepted[group.fields[f].idx(i, j, k)] =
              before[f][group.fields[f].idx(i, j, k)];
    REQUIRE(std::memcmp(accepted.data(), before[f].data(),
                        accepted.size() * sizeof(double)) == 0);
  }
  const auto committed = group.snapshot();
  REQUIRE_THROWS(moved.commit(attempt));
  group.same(committed);
}

TEST_CASE("runtime field RK rejects rank-local failures collectively and retries",
          "[runtime_field_rk][MPI]") {
  Group group(17);
  group.owned(2.);
  const auto before = group.snapshot();
  bool fail = true, nan = false;
  int calls = 0;
  auto rhs = [&](double, Rates rates) {
    group.exchange();
    ++calls;
    for (auto rate : rates) std::fill(rate.begin(), rate.end(), 1.);
    if (group.rank == 0 && fail && calls == 2) {
      group.fields[5](0, 0, 0) = 9.;
      if (nan)
        rates[3][0] = std::numeric_limits<double>::quiet_NaN();
      else
        throw std::runtime_error("manufactured rank-zero stage failure");
    }
  };
  auto rk = create_runtime(group.refs, 0.1, make_rk4_classical<double>(), rhs,
                           MPI_COMM_WORLD);
  REQUIRE_THROWS(rk.step(0.));
  group.same(before);
  calls = 0;
  nan = true;
  REQUIRE_THROWS(rk.step(0.));
  group.same(before);
  calls = 0;
  fail = false;
  const auto good = rk.attempt(0.);
  group.same(before);
  rk.commit(good);
  for (auto &f : group.fields) REQUIRE(std::abs(f(0, 0, 0) - 2.1) < 1e-14);
}

TEST_CASE("runtime field RK rejects stale and edited attempt descriptors",
          "[runtime_field_rk][MPI]") {
  Group group(2);
  group.owned(2.);
  auto rhs = [](double, Rates rates) {
    for (auto r : rates) std::fill(r.begin(), r.end(), 1.);
  };
  auto rk = create_runtime(group.refs, 0.1, make_rk4_classical<double>(), rhs,
                           MPI_COMM_WORLD);
  const auto before = group.snapshot();
  for (int descriptor = 0; descriptor < 4; ++descriptor) {
    auto a = rk.attempt(0.);
    if (group.rank == 0) {
      if (descriptor == 0) a.t0 += 1.;
      if (descriptor == 1) a.dt += 1.;
      if (descriptor == 2) a.t1 += 1.;
      if (descriptor == 3) a.success = false;
    }
    REQUIRE_THROWS(rk.commit(a));
    group.same(before);
  }
  auto old = rk.attempt(0.);
  auto latest = rk.attempt(0.);
  REQUIRE_THROWS(rk.commit(old));
  group.same(before);
  REQUIRE_THROWS(rk.commit(latest));
  group.same(before);
  auto a = rk.attempt(0.);
  if (group.rank == 0) group.fields[1](0, 0, 0) = 4.;
  const auto edited = group.snapshot();
  REQUIRE_THROWS(rk.commit(a));
  group.same(edited);
  group.owned(2.);
  const auto retry = rk.attempt(0.);
  rk.commit(retry);
}

TEST_CASE("runtime field RK validates rank-local input before publication",
          "[runtime_field_rk][MPI]") {
  Group group(2);
  group.owned(2.);
  const auto before = group.snapshot();
  int calls = 0;
  RuntimeFieldRKStepper::StageRhs rhs = [&](double, Rates rates) {
    ++calls;
    for (auto r : rates) std::fill(r.begin(), r.end(), 1.);
  };
  const auto method = make_rk4_classical<double>();
  for (int invalid = 0; invalid < 6; ++invalid) {
    if (invalid == 3 && group.ranks == 1) continue;
    auto refs = group.refs;
    double dt = 0.1;
    auto callback = rhs;
    if (group.rank == 0) {
      if (invalid == 0) refs.clear();
      if (invalid == 1) refs[0] = nullptr;
      if (invalid == 2) refs[1] = refs[0];
      if (invalid == 3) refs.pop_back();
      if (invalid == 4) dt = std::numeric_limits<double>::quiet_NaN();
      if (invalid == 5) callback = {};
    }
    REQUIRE_THROWS(create_runtime(refs, dt, method, callback, MPI_COMM_WORLD));
    group.same(before);
    REQUIRE(calls == 0);
  }
  for (int invalid = 0; invalid < 4; ++invalid) {
    auto domain = group.domain;
    if (invalid == 0) domain.origin[0] = 1.;
    if (invalid == 1) domain.spacing[0] = 0.5;
    if (invalid == 2) domain.periodic[0] = false;
    const int halo = invalid == 3 ? 2 : 1;
    Field other(domain, group.fields[0].box(), halo);
    auto refs = group.refs;
    if (group.rank == 0) refs[1] = &other;
    REQUIRE_THROWS(create_runtime(refs, 0.1, method, rhs, MPI_COMM_WORLD));
    group.same(before);
    REQUIRE(calls == 0);
  }
  auto rk = create_runtime(group.refs, 0.1, method, rhs, MPI_COMM_WORLD);
  REQUIRE_THROWS(
      rk.attempt(group.rank == 0 ? std::numeric_limits<double>::infinity() : 0.));
  group.same(before);
  REQUIRE(calls == 0);
  if (group.ranks > 1) {
    REQUIRE_THROWS(rk.attempt(group.rank == 0 ? 0.1 : 0.));
    group.same(before);
    REQUIRE(calls == 0);
  }
  if (group.rank == 0)
    group.fields[1](0, 0, 0) = std::numeric_limits<double>::quiet_NaN();
  const auto invalid_state = group.snapshot();
  REQUIRE_THROWS(rk.attempt(0.));
  group.same(invalid_state);
  REQUIRE(calls == 0);
}

TEST_CASE("runtime field RK preserves malformed-callback restoration contract",
          "[runtime_field_rk][MPI]") {
  Group group(2);
  group.owned(2.);
  const auto before = group.snapshot();
  auto rhs = [&](double, Rates rates) {
    group.exchange();
    for (auto rate : rates) std::fill(rate.begin(), rate.end(), 1.);
    if (group.rank == 0) group.fields[1].vec().resize(group.fields[1].size() + 7);
  };
  auto rk = create_runtime(group.refs, 0.1, make_rk2_midpoint<double>(), rhs,
                           MPI_COMM_WORLD);
  REQUIRE_THROWS(rk.step(0.));
  group.same(before);
  REQUIRE_THROWS(rk.step(0.));
  group.same(before);
  // Changed storage invalidates old bindings; recreating a complete binding works.
  auto restored = create_runtime(
      group.refs, 0.1, make_rk2_midpoint<double>(),
      [](double, Rates rates) {
        for (auto r : rates) std::fill(r.begin(), r.end(), 1.);
      },
      MPI_COMM_WORLD);
  restored.step(0.);
  for (auto &field : group.fields) REQUIRE(std::abs(field(0, 0, 0) - 2.1) < 1e-14);
}

TEST_CASE("runtime field RK reports scalar scratch capacity without guessing",
          "[runtime_field_rk][MPI]") {
  Group group(17);
  group.owned(2.);
  auto rk = create_runtime(
      group.refs, 0.1, make_rk4_classical<double>(),
      [](double, Rates rates) {
        for (auto r : rates) std::fill(r.begin(), r.end(), 1.);
      },
      MPI_COMM_WORLD);
  const auto n = group.fields[0].local_size();
  const std::size_t owned = static_cast<std::size_t>(n[0]) * n[1] * n[2];
  // Compact accepted state, RK4 rates, stage input and candidate, plus one
  // accepted padded snapshot per field. No second stage image or backup.
  const std::size_t expected =
      17 * (7 * owned + group.fields[0].size()) * sizeof(double);
  REQUIRE(rk.workspace_bytes() == expected);
  rk.step(0.);
  REQUIRE(rk.workspace_bytes() == expected);
}

TEST_CASE("runtime field RK rejects finite-rate overflow and invalidates attempts",
          "[runtime_field_rk][MPI]") {
  Group group(2);
  group.owned(2.);
  const auto before = group.snapshot();
  bool overflow = false, fail = false;
  auto rhs = [&](double, Rates rates) {
    for (auto r : rates)
      std::fill(r.begin(), r.end(),
                overflow ? std::numeric_limits<double>::max() : 1.);
    if (group.rank == 0 && fail)
      throw std::runtime_error("post-collective rejection");
  };
  auto rk =
      create_runtime(group.refs, 2., ButcherTableau<double>(1, {0.}, {1.}, {0.}),
                     rhs, MPI_COMM_WORLD);
  const auto old = rk.attempt(0.);
  overflow = true;
  REQUIRE_THROWS(rk.attempt(0.));
  group.same(before);
  REQUIRE_THROWS(rk.commit(old));
  group.same(before);
  overflow = false;
  fail = true;
  REQUIRE_THROWS(rk.attempt(0.));
  group.same(before);
  fail = false;
  rk.step(0.);
  for (auto &field : group.fields) REQUIRE(field(0, 0, 0) == 4.);
}

TEST_CASE("runtime field RK rejects a rank-local reentrant callback safely",
          "[runtime_field_rk][MPI]") {
  Group group(2);
  group.owned(2.);
  const auto before = group.snapshot();
  RuntimeFieldRKStepper *active = nullptr;
  bool recurse = true;
  auto rhs = [&](double t, Rates rates) {
    group.exchange();
    for (auto r : rates) std::fill(r.begin(), r.end(), 1.);
    if (group.rank == 0 && recurse) (void)active->attempt(t);
  };
  auto rk = create_runtime(group.refs, 0.1, make_rk4_classical<double>(), rhs,
                           MPI_COMM_WORLD);
  active = &rk;
  REQUIRE_THROWS(rk.step(0.));
  group.same(before);
  recurse = false;
  rk.step(0.);
  for (auto &field : group.fields) REQUIRE(std::abs(field(0, 0, 0) - 2.1) < 1e-14);
}

TEST_CASE("runtime field RK rejects cross-rank method and global geometry mismatch",
          "[runtime_field_rk][MPI]") {
  Group group(2);
  group.owned(2.);
  const auto before = group.snapshot();
  if (group.ranks == 1) return;
  auto rhs = [](double, Rates rates) {
    for (auto r : rates) std::fill(r.begin(), r.end(), 1.);
  };
  auto method =
      group.rank == 0 ? make_rk2_midpoint<double>() : make_rk4_classical<double>();
  REQUIRE_THROWS(create_runtime(group.refs, 0.1, method, rhs, MPI_COMM_WORLD));
  auto domain = group.domain;
  if (group.rank == 0) domain.origin[0] = 1.;
  Field u(domain, group.fields[0].box(), 1), v(domain, group.fields[0].box(), 1);
  std::vector<Field *> foreign{&u, &v};
  REQUIRE_THROWS(create_runtime(foreign, 0.1, make_rk4_classical<double>(), rhs,
                                MPI_COMM_WORLD));
  REQUIRE_THROWS(create_runtime(group.refs, group.rank == 0 ? 0.2 : 0.1,
                                make_rk4_classical<double>(), rhs, MPI_COMM_WORLD));
  group.same(before);
}

TEST_CASE("runtime field RK rejects a unilateral callback commit without nesting "
          "collectives",
          "[runtime_field_rk][MPI]") {
  Group group(2);
  group.owned(2.);
  const auto before = group.snapshot();
  RuntimeFieldRKStepper *stepper = nullptr;
  const RuntimeFieldRKStepper::Attempt *previous = nullptr;
  bool nested = false;
  auto rhs = [&](double, Rates rates) {
    group.exchange();
    if (nested && group.rank == 0) stepper->commit(*previous);
    for (auto rate : rates) std::fill(rate.begin(), rate.end(), 1.);
  };
  auto rk = create_runtime(group.refs, 0.1, make_rk4_classical<double>(), rhs,
                           MPI_COMM_WORLD);
  stepper = &rk;
  const auto old = rk.attempt(0.);
  previous = &old;
  nested = true;
  REQUIRE_THROWS(rk.attempt(0.1));
  group.same(before);
  nested = false;
  const auto retry = rk.attempt(0.1);
  rk.commit(retry);
  REQUIRE(std::abs(group.fields.back()(0, 0, 0) - 2.1) < 2e-14);
}
