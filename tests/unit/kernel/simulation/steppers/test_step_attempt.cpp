// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <complex>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/state_concepts.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

using pfc::sim::steppers::commit_step_attempt;
using pfc::sim::steppers::EulerStepper;
using pfc::sim::steppers::MultiEulerStepper;
using pfc::sim::steppers::PackedEulerStepper;
using pfc::sim::steppers::PackedStageFunction;
using pfc::sim::steppers::StepAttemptResult;

namespace {

struct DecayRhs {
  void operator()(double /*t*/, std::vector<double> & /*u*/,
                  std::vector<double> &du) const {
    for (double &v : du) {
      v = 1.0;
    }
  }
};

struct TwoFieldRhs {
  void operator()(double /*t*/,
                  std::tuple<std::vector<double> &, std::vector<double> &> /*u*/,
                  std::tuple<std::vector<double> &, std::vector<double> &> du)
      const {
    auto &d0 = std::get<0>(du);
    auto &d1 = std::get<1>(du);
    for (double &v : d0) {
      v = 1.0;
    }
    for (double &v : d1) {
      v = 2.0;
    }
  }
};

struct ThreeFieldRhs {
  void operator()(
      double /*t*/,
      std::tuple<std::vector<double> &, std::vector<double> &,
                 std::vector<double> &> /*u*/,
      std::tuple<std::vector<double> &, std::vector<double> &,
                 std::vector<double> &>
          du) const {
    std::get<0>(du)[0] = 1.0;
    std::get<1>(du)[0] = 2.0;
    std::get<2>(du)[0] = 3.0;
  }
};

} // namespace

TEST_CASE("success_isolates_accepted_until_commit", "[step_attempt][unit]") {
  DecayRhs rhs{};
  EulerStepper stepper(0.25, 3, rhs);

  std::vector<double> accepted{1.0, 2.0, 3.0};
  const std::vector<double> fingerprint = accepted;
  const auto result = stepper.attempt(0.5, accepted);

  REQUIRE(result.success);
  REQUIRE(result.t0 == Catch::Approx(0.5));
  REQUIRE(result.dt == Catch::Approx(0.25));
  REQUIRE(result.t1 == Catch::Approx(0.75));
  REQUIRE(accepted == fingerprint);
  for (std::size_t i = 0; i < accepted.size(); ++i) {
    REQUIRE(result.candidate[i] ==
            Catch::Approx(fingerprint[i] + 0.25 * 1.0));
  }

  commit_step_attempt(accepted, result);
  REQUIRE(accepted == result.candidate);
  REQUIRE(accepted != fingerprint);
}

TEST_CASE("failed_result_cannot_be_committed", "[step_attempt][unit]") {
  std::vector<double> dummy{0.0, 0.0};
  const StepAttemptResult fail(1.0, 0.1, 1.0, /*success=*/false, dummy);
  std::vector<double> accepted{4.0, -1.0};
  const auto fingerprint = accepted;
  REQUIRE_THROWS_AS(commit_step_attempt(accepted, fail), std::invalid_argument);
  REQUIRE(accepted == fingerprint);
}

TEST_CASE("multi_field_N2_isolation", "[step_attempt][unit]") {
  TwoFieldRhs rhs{};
  MultiEulerStepper<TwoFieldRhs, 2> stepper(0.5, {2, 3}, rhs);

  std::vector<double> u0{1.0, 2.0};
  std::vector<double> u1{3.0, 4.0, 5.0};
  const auto fp0 = u0;
  const auto fp1 = u1;
  const auto result = stepper.attempt(0.0, u0, u1);

  REQUIRE(result.success);
  REQUIRE(result.t1 == Catch::Approx(0.5));
  REQUIRE(u0 == fp0);
  REQUIRE(u1 == fp1);
  for (std::size_t i = 0; i < fp0.size(); ++i) {
    REQUIRE(result.candidate(0)[i] == Catch::Approx(fp0[i] + 0.5 * 1.0));
  }
  for (std::size_t i = 0; i < fp1.size(); ++i) {
    REQUIRE(result.candidate(1)[i] == Catch::Approx(fp1[i] + 0.5 * 2.0));
  }

  commit_step_attempt(u0, u1, result);
  REQUIRE(u0 == result.candidate(0));
  REQUIRE(u1 == result.candidate(1));
}

TEST_CASE("multi_field_N3_isolation", "[step_attempt][unit]") {
  ThreeFieldRhs rhs{};
  MultiEulerStepper<ThreeFieldRhs, 3> stepper(0.1, {1, 1, 1}, rhs);
  std::vector<double> u0{1.0}, u1{2.0}, u2{3.0};
  const auto fp0 = u0;
  const auto fp1 = u1;
  const auto fp2 = u2;
  const auto result = stepper.attempt(0.0, u0, u1, u2);
  REQUIRE(result.success);
  REQUIRE(u0 == fp0);
  REQUIRE(u1 == fp1);
  REQUIRE(u2 == fp2);
  REQUIRE(result.candidate(0)[0] == Catch::Approx(1.1));
  REQUIRE(result.candidate(1)[0] == Catch::Approx(2.2));
  REQUIRE(result.candidate(2)[0] == Catch::Approx(3.3));
  (void)stepper.step(0.0, u0, u1, u2);
  REQUIRE(u0[0] == Catch::Approx(1.1));
  REQUIRE(u1[0] == Catch::Approx(2.2));
  REQUIRE(u2[0] == Catch::Approx(3.3));
}

TEST_CASE("multi_field_host_field_pack_isolation", "[step_attempt][unit][field]") {
  using pfc::data::Field;
  static_assert(pfc::field::HostFieldPack<double, Field<double>, Field<double>>);

  TwoFieldRhs rhs{};
  MultiEulerStepper<TwoFieldRhs, 2> stepper(0.5, {2, 2}, rhs);

  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 0, 0});
  Field<double> a(domain, box, 0);
  Field<double> b(domain, box, 0);
  a.vec() = {1.0, 2.0};
  b.vec() = {3.0, 4.0};
  const auto fp0 = a.vec();
  const auto fp1 = b.vec();

  const auto result = stepper.attempt(0.0, a, b);
  REQUIRE(result.success);
  REQUIRE(a.vec() == fp0);
  REQUIRE(b.vec() == fp1);
  REQUIRE(result.candidate(0)[0] == Catch::Approx(1.0 + 0.5 * 1.0));
  REQUIRE(result.candidate(1)[0] == Catch::Approx(3.0 + 0.5 * 2.0));

  (void)stepper.step(0.0, a, b);
  REQUIRE(a.vec()[0] == Catch::Approx(1.5));
  REQUIRE(b.vec()[0] == Catch::Approx(4.0));
}

using Complex = std::complex<double>;

struct MixedScalarRhs {
  void operator()(
      double /*t*/,
      std::tuple<std::vector<double> &, std::vector<Complex> &> /*u*/,
      std::tuple<std::vector<double> &, std::vector<Complex> &> du) const {
    std::get<0>(du)[0] = 1.0;
    std::get<1>(du)[0] = Complex{0.0, 2.0};
  }
};

TEST_CASE("packed_euler_mixed_scalar_isolation",
          "[step_attempt][unit][packed]") {
  static_assert(PackedStageFunction<MixedScalarRhs, double, Complex>);
  MixedScalarRhs rhs{};
  PackedEulerStepper<MixedScalarRhs, double, Complex> stepper(0.5, {1, 1}, rhs);

  std::vector<double> u0{2.0};
  std::vector<Complex> u1{Complex{1.0, -1.0}};
  const auto fp0 = u0;
  const auto fp1 = u1;
  const auto result = stepper.attempt(0.0, u0, u1);
  REQUIRE(result.success);
  REQUIRE(u0 == fp0);
  REQUIRE(u1 == fp1);
  REQUIRE(result.candidate<0>()[0] == Catch::Approx(2.0 + 0.5 * 1.0));
  REQUIRE(result.candidate<1>()[0].real() ==
          Catch::Approx(1.0).margin(1e-12));
  REQUIRE(result.candidate<1>()[0].imag() ==
          Catch::Approx(-1.0 + 0.5 * 2.0).margin(1e-12));

  commit_step_attempt(u0, u1, result);
  REQUIRE(u0[0] == Catch::Approx(2.5));
  REQUIRE(u1[0].imag() == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("packed_euler_mixed_host_fields",
          "[step_attempt][unit][packed][field]") {
  using pfc::data::Field;
  MixedScalarRhs rhs{};
  PackedEulerStepper<MixedScalarRhs, double, Complex> stepper(0.25, {1, 1},
                                                              rhs);
  const auto domain = pfc::domain::create({1, 1, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {0, 0, 0});
  Field<double> a(domain, box, 0);
  Field<Complex> b(domain, box, 0);
  a.vec() = {4.0};
  b.vec() = {Complex{0.0, 1.0}};
  const auto result = stepper.attempt(0.0, a, b);
  REQUIRE(result.success);
  REQUIRE(a.vec()[0] == Catch::Approx(4.0));
  REQUIRE(b.vec()[0].imag() == Catch::Approx(1.0).margin(1e-12));
  (void)stepper.step(0.0, a, b);
  REQUIRE(a.vec()[0] == Catch::Approx(4.25));
  REQUIRE(b.vec()[0].imag() == Catch::Approx(1.0 + 0.25 * 2.0).margin(1e-12));
}
