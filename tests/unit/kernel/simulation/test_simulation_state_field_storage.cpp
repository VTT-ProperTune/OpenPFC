// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// M2 unit tests for pfc::SimulationState: owning the canonical
// pfc::data::Field<T, MemorySpace> by name and by typed FieldHandle<T>,
// including heterogeneous element types coexisting in one state.

#include <complex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

using namespace pfc;

namespace {

Box3i whole_box(int nx, int ny, int nz) {
  return Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
}

// A host field whose owned cell (0,0,0) is seeded with `seed` so we can tell
// two fields apart after they have been moved into the state.
data::Field<double> make_field(double seed, int nx = 4, int ny = 3, int nz = 2) {
  data::Field<double> f(domain::create({nx, ny, nz}), whole_box(nx, ny, nz), 0);
  f(0, 0, 0) = seed;
  return f;
}

} // namespace

TEST_CASE("SimulationState: add then get by name round-trips",
          "[simulation_state][unit]") {
  SimulationState state;
  REQUIRE(state.num_fields() == 0);
  REQUIRE_FALSE(state.has_field("u"));

  state.add_field<double>("u", make_field(1.5));

  REQUIRE(state.num_fields() == 1);
  REQUIRE(state.has_field("u"));

  data::Field<double> &u = state.get_field<double>("u");
  REQUIRE(u(0, 0, 0) == 1.5);

  // A mutation through the reference is observed on the owned field.
  u(0, 0, 0) = 9.0;
  REQUIRE(state.get_field<double>("u")(0, 0, 0) == 9.0);
}

TEST_CASE("SimulationState: handle from name fetches the same field",
          "[simulation_state][unit]") {
  SimulationState state;
  state.add_field<double>("u", make_field(2.0));
  state.add_field<double>("v", make_field(3.0));

  FieldHandle<double> hu = state.get_field_handle<double>("u");
  FieldHandle<double> hv = state.get_field_handle<double>("v");

  REQUIRE(hu.is_valid());
  REQUIRE(hv.is_valid());
  REQUIRE(hu != hv);

  REQUIRE(state.get_field_by_handle<double>(hu)(0, 0, 0) == 2.0);
  REQUIRE(state.get_field_by_handle<double>(hv)(0, 0, 0) == 3.0);

  // Handle access and name access alias the same owned field.
  state.get_field_by_handle<double>(hu)(0, 0, 0) = 42.0;
  REQUIRE(state.get_field<double>("u")(0, 0, 0) == 42.0);
}

TEST_CASE("SimulationState: heterogeneous element types coexist",
          "[simulation_state][unit]") {
  using cdouble = std::complex<double>;
  SimulationState state;

  state.add_field<double>("psi", make_field(1.0));

  data::Field<cdouble> spec(domain::create({4, 3, 2}), whole_box(4, 3, 2), 0);
  spec(0, 0, 0) = cdouble(4.0, -1.0);
  state.add_field<cdouble>("psi_hat", std::move(spec));

  REQUIRE(state.num_fields() == 2);
  REQUIRE(state.get_field<double>("psi")(0, 0, 0) == 1.0);
  REQUIRE(state.get_field<cdouble>("psi_hat")(0, 0, 0) == cdouble(4.0, -1.0));

  // The real and complex handle spaces are independent types; a
  // FieldHandle<cdouble> resolves only against the complex store.
  FieldHandle<cdouble> hc = state.get_field_handle<cdouble>("psi_hat");
  REQUIRE(state.get_field_by_handle<cdouble>(hc)(0, 0, 0) == cdouble(4.0, -1.0));
}

TEST_CASE("SimulationState: error paths throw", "[simulation_state][unit]") {
  using cdouble = std::complex<double>;
  SimulationState state;
  state.add_field<double>("u", make_field(1.0));

  SECTION("duplicate name is rejected") {
    REQUIRE_THROWS_AS(state.add_field<double>("u", make_field(2.0)),
                      std::runtime_error);
  }
  SECTION("unknown name throws") {
    REQUIRE_THROWS_AS(state.get_field<double>("nope"), std::runtime_error);
    REQUIRE_THROWS_AS(state.get_field_handle<double>("nope"), std::runtime_error);
  }
  SECTION("wrong element type throws") {
    REQUIRE_THROWS_AS(state.get_field<cdouble>("u"), std::runtime_error);
    REQUIRE_THROWS_AS(state.get_field_handle<cdouble>("u"), std::runtime_error);
  }
  SECTION("invalid / default handle throws") {
    FieldHandle<double> bad; // default-constructed, invalid
    REQUIRE_FALSE(bad.is_valid());
    REQUIRE_THROWS_AS(state.get_field_by_handle<double>(bad), std::runtime_error);
  }
}

TEST_CASE("SimulationState: const access is read-only and consistent",
          "[simulation_state][unit]") {
  SimulationState state;
  state.add_field<double>("u", make_field(7.0));
  const SimulationState &cs = state;

  REQUIRE(cs.has_field("u"));
  REQUIRE(cs.num_fields() == 1);
  REQUIRE(cs.get_field<double>("u")(0, 0, 0) == 7.0);
  FieldHandle<double> h = cs.get_field_handle<double>("u");
  REQUIRE(cs.get_field_by_handle<double>(h)(0, 0, 0) == 7.0);
}

TEST_CASE("FieldHandle: value semantics and std::hash", "[simulation_state][unit]") {
  FieldHandle<double> a; // invalid
  FieldHandle<double> b(7);
  FieldHandle<double> c(7);

  REQUIRE_FALSE(a.is_valid());
  REQUIRE(b.is_valid());
  REQUIRE(b == c);
  REQUIRE(a != b);

  // Usable as an unordered-container key via the std::hash specialization.
  std::unordered_set<FieldHandle<double>> seen;
  seen.insert(b);
  REQUIRE(seen.count(c) == 1);
  REQUIRE(seen.count(a) == 0);

  std::unordered_map<FieldHandle<double>, int> labels;
  labels[b] = 5;
  REQUIRE(labels.at(c) == 5);
}

#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>

// Compile-only coverage for Device memory space (CPU SIF cannot execute these).
// Satisfies the "multiple T / Device combinations" acceptance criterion when
// the CUDA toolchain builds this TU.
TEST_CASE("SimulationState: CudaSpace fields register by name (compile check)",
          "[simulation_state][unit][cuda]") {
  using namespace pfc;
  SimulationState state;

  data::Field<double, CudaSpace> u(domain::create({4, 2, 1}),
                                   whole_box(4, 2, 1), 0);
  data::Field<std::complex<double>, CudaSpace> uh(
      domain::create({4, 2, 1}), whole_box(4, 2, 1), 0);

  state.add_field<double, CudaSpace>("u_dev", std::move(u));
  state.add_field<std::complex<double>, CudaSpace>("uh_dev", std::move(uh));

  REQUIRE(state.has_field("u_dev"));
  REQUIRE(state.has_field("uh_dev"));
  REQUIRE(state.num_fields() == 2);

  // Touch residency APIs so the device Field path is not dead-stripped.
  auto &uref = state.get_field<double, CudaSpace>("u_dev");
  uref.sync_to_device();
  uref.note_device_write();
  (void)uref.residency();
}
#endif // OpenPFC_ENABLE_CUDA
