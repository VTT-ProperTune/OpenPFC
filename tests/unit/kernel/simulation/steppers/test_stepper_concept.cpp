// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <openpfc/kernel/simulation/steppers/rk2_heun.hpp>
#include <openpfc/kernel/simulation/steppers/stepper_concept.hpp>

#include <tuple>
#include <vector>

using namespace pfc::sim::steppers;

namespace {

// Single-field RHS matching EulerStepper/RK2HeunStepper/ExplicitRKStepper's
// required signature: rhs(t, u, du), filling du.
struct ConstantRHS {
  void operator()(double /*t*/, const std::vector<double> & /*u*/,
                  std::vector<double> &du) const {
    for (auto &v : du) v = 1.0;
  }
};

// Two-field RHS matching MultiEulerStepper<Rhs, 2>/MultiExplicitRKStepper<Rhs,
// 2>'s required signature: rhs(t, u_pack, du_pack), filling every du in
// du_pack.
struct CompositeRHS {
  void
  operator()(double /*t*/,
             const std::tuple<std::vector<double> &, std::vector<double> &> /*u*/,
             std::tuple<std::vector<double> &, std::vector<double> &> du) const {
    std::get<0>(du).assign(std::get<0>(du).size(), 1.0);
    std::get<1>(du).assign(std::get<1>(du).size(), 1.0);
  }
};

// Three-field RHS matching MultiEulerStepper<Rhs, 3>'s required signature.
struct CompositeRHS3 {
  void operator()(
      double /*t*/,
      const std::tuple<std::vector<double> &, std::vector<double> &,
                       std::vector<double> &> /*u*/,
      std::tuple<std::vector<double> &, std::vector<double> &, std::vector<double> &>
          du) const {
    std::get<0>(du).assign(std::get<0>(du).size(), 1.0);
    std::get<1>(du).assign(std::get<1>(du).size(), 1.0);
    std::get<2>(du).assign(std::get<2>(du).size(), 1.0);
  }
};

} // namespace

// -- SingleFieldStepper --------------------------------------------------

static_assert(SingleFieldStepper<EulerStepper<ConstantRHS>>);
static_assert(SingleFieldStepper<RK2HeunStepper<ConstantRHS>>);
static_assert(SingleFieldStepper<ExplicitRKStepper<ConstantRHS>>);

static_assert(!SingleFieldStepper<int>);
static_assert(!SingleFieldStepper<ConstantRHS>);

// -- MultiFieldStepper ----------------------------------------------------

// MultiEulerStepper has a `static constexpr std::size_t field_count` member
// and satisfies MultiFieldStepper, for both a 2-field and a 3-field
// instantiation.
static_assert(MultiFieldStepper<MultiEulerStepper<CompositeRHS, 2>>);
static_assert(MultiEulerStepper<CompositeRHS, 2>::field_count == 2);

static_assert(MultiEulerStepper<CompositeRHS3, 3>::field_count == 3);
// step() is a variadic template (template<class... U> step(double,
// std::vector<U>&...)); a concept's requires-expression only checks that
// the call is well-formed at the declaration level (deduction + return
// type), never instantiating the body -- so calling it with exactly two
// buffers deduces U...={double,double} and satisfies the concept even for
// N=3, without ever reaching step()'s own internal
// static_assert(sizeof...(U) == N) (which only fires if the 2-argument
// overload is actually called for real, elsewhere). MultiFieldStepper is
// therefore satisfied here too; field_count itself is verified above.
static_assert(MultiFieldStepper<MultiEulerStepper<CompositeRHS3, 3>>);

// MultiExplicitRKStepper does NOT currently expose a `field_count` static
// member (verified directly in explicit_rk.hpp -- grep finds no match), so
// it does not satisfy MultiFieldStepper as specified. This is a real,
// verified property of the current codebase, not an oversight in the
// concept: MultiFieldStepper's field_count requirement is deliberate (the
// same requirement multi-field callers already rely on via
// MultiEulerStepper), and this static_assert documents that
// MultiExplicitRKStepper would need a field_count member added before it
// can satisfy the same contract.
static_assert(!MultiFieldStepper<MultiExplicitRKStepper<CompositeRHS, 2>>);

static_assert(!MultiFieldStepper<int>);
// A single-field stepper does not satisfy MultiFieldStepper (no field_count,
// and step() only accepts one buffer).
static_assert(!MultiFieldStepper<EulerStepper<ConstantRHS>>);

TEST_CASE("SingleFieldStepper concept accepts real single-field steppers") {
  REQUIRE(SingleFieldStepper<EulerStepper<ConstantRHS>>);
  REQUIRE(SingleFieldStepper<RK2HeunStepper<ConstantRHS>>);
  REQUIRE(SingleFieldStepper<ExplicitRKStepper<ConstantRHS>>);
}

TEST_CASE("MultiFieldStepper concept accepts MultiEulerStepper with 2 fields") {
  REQUIRE(MultiFieldStepper<MultiEulerStepper<CompositeRHS, 2>>);
}

TEST_CASE("MultiEulerStepper with 3 fields reports field_count == 3") {
  REQUIRE(MultiEulerStepper<CompositeRHS3, 3>::field_count == 3);
}

TEST_CASE("MultiFieldStepper concept rejects types lacking field_count") {
  REQUIRE_FALSE(MultiFieldStepper<int>);
  REQUIRE_FALSE(MultiFieldStepper<EulerStepper<ConstantRHS>>);
  // MultiExplicitRKStepper has no field_count member today (verified: grep
  // finds no match in explicit_rk.hpp), so it does not satisfy this concept.
  REQUIRE_FALSE(MultiFieldStepper<MultiExplicitRKStepper<CompositeRHS, 2>>);
}
