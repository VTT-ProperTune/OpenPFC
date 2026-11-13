// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_float.cpp
 * @brief Tungsten in single precision (ADR 0006, #854): the `model.precision`
 *        switch, the float session against the double reference, the pinned
 *        32^3 sine checksum within tolerance, restart equivalence and the
 *        moving boundary on float fields. The double goldens are untouched.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numbers>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/precision_json.hpp>
#include <tungsten/tungsten_physics.hpp>
#include <tungsten/tungsten_session.hpp>

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
using nlohmann::json;
using pfc::ui::Precision;
using pfc::ui::precision_from_json;

namespace {

json float_params_json() {
  return {{"n0", -0.10},        {"n_sol", -0.047},
          {"n_vap", -0.464},    {"T", 3300.0},
          {"T0", 156000.0},     {"Bx", 0.8582},
          {"alpha", 0.50},      {"alpha_farTol", 0.001},
          {"alpha_highOrd", 4}, {"lambda", 0.22},
          {"stabP", 0.2},       {"shift_u", 0.3341},
          {"shift_s", 0.1898},  {"p2", 1.0},
          {"p3", -0.5},         {"p4", 0.333333333},
          {"q20", -0.0037},     {"q21", 1.0},
          {"q30", -12.4567},    {"q31", 20.0},
          {"q40", 45.0}};
}

json float_settings(int n, double t1, double dt) {
  return {{"model", {{"name", "tungsten"}, {"params", float_params_json()}}},
          {"domain",
           {{"size", {n, n, n}}, {"lower", {0.0, 0.0, 0.0}}, {"upper", {n, n, n}}}},
          {"timestepping", {{"t0", 0.0}, {"t1", t1}, {"dt", dt}, {"saveat", dt}}},
          {"initial_conditions",
           {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.10}}}}};
}

double sine_ic(std::size_t i, std::size_t n) {
  return -0.4 + 0.1 * std::sin(2.0 * std::numbers::pi * static_cast<double>(i) /
                               static_cast<double>(n));
}

template <class Real> void fill_sine(std::vector<Real> &v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<Real>(sine_ic(i, v.size()));
  }
}

double max_abs_diff(const std::vector<double> &a, const std::vector<float> &b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(std::isfinite(b[i]));
    m = std::max(m, std::abs(a[i] - static_cast<double>(b[i])));
  }
  return m;
}

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char *tag) {
    path = std::filesystem::temp_directory_path() /
           (std::string("openpfc_tungsten_float_") + tag + "_" +
            std::to_string(static_cast<long>(::getpid())));
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

} // namespace

static_assert(std::is_same_v<tungsten::TungstenFloatSession::Real, float>);
static_assert(std::is_same_v<tungsten::TungstenSession::Real, double>);
static_assert(std::is_same_v<tungsten::TungstenPhysics<float>::real_type, float>);

TEST_CASE("model.precision parses double, float and nothing else",
          "[tungsten][float][precision]") {
  json s = float_settings(8, 0.01, 0.01);
  REQUIRE(precision_from_json(s) == Precision::Double);
  s["model"]["precision"] = "double";
  REQUIRE(precision_from_json(s) == Precision::Double);
  s["model"]["precision"] = "float";
  REQUIRE(precision_from_json(s) == Precision::Float);
  s["model"]["precision"] = "half";
  REQUIRE_THROWS_WITH(precision_from_json(s), ContainsSubstring("half"));
  s["model"]["precision"] = 32;
  REQUIRE_THROWS_AS(precision_from_json(s), std::invalid_argument);
  REQUIRE(precision_from_json(json::object()) == Precision::Double);
  REQUIRE(std::string(pfc::ui::to_string(Precision::Float)) == "float");
  REQUIRE(pfc::ui::precision_of<float>() == Precision::Float);
  REQUIRE(pfc::ui::precision_of<double>() == Precision::Double);
}

TEST_CASE("a session rejects model.precision naming the other scalar",
          "[tungsten][float][precision]") {
  json s = float_settings(8, 0.01, 0.01);
  s["model"]["precision"] = "float";
  REQUIRE_THROWS_WITH((tungsten::TungstenSession(s, 0, 1, MPI_COMM_WORLD)),
                      ContainsSubstring("model.precision"));
  s["model"]["precision"] = "double";
  REQUIRE_THROWS_WITH((tungsten::TungstenFloatSession(s, 0, 1, MPI_COMM_WORLD)),
                      ContainsSubstring("model.precision"));
  // Absent key: either session accepts the input.
  s["model"].erase("precision");
  REQUIRE_NOTHROW(tungsten::TungstenFloatSession(s, 0, 1, MPI_COMM_WORLD));
}

TEST_CASE("TungstenFloatSession declares float fields and runs",
          "[tungsten][float][session]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  json s = float_settings(8, 0.05, 0.01);
  s["model"]["precision"] = "float";
  tungsten::TungstenFloatSession session(s, 0, 1, MPI_COMM_WORLD);
  REQUIRE(session.state().has_typed_field<float>("psi"));
  REQUIRE(session.state().has_typed_field<std::complex<float>>("psi_hat"));
  REQUIRE(session.state().has_typed_field<float>("psi_mf"));
  REQUIRE_FALSE(session.state().has_typed_field<double>("psi"));
  session.run();
  const auto cs = session.field_checksum();
  REQUIRE(std::isfinite(cs.sum));
  REQUIRE(cs.sumsq > 0.0);
}

// Tolerance baseline (tests/baselines/BASELINES.md): the float session tracks
// the double session from the same sine IC within 1e-5 after ten steps.
TEST_CASE("TungstenFloatSession tracks TungstenSession within 1e-5 (8^3 and "
          "32^3 sine IC, 10 steps)",
          "[tungsten][float][session][baseline]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  for (int n : {8, 32}) {
    INFO("n = " << n);
    json s = float_settings(n, 0.1, 0.01);
    s["model"]["params"]["n0"] = -0.4;
    s["model"]["params"]["T"] = 0.5;
    s["initial_conditions"] =
        json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.4}}});

    tungsten::TungstenSession ref(s, 0, 1, MPI_COMM_WORLD);
    fill_sine(ref.psi().vec());
    ref.run();

    s["model"]["precision"] = "float";
    tungsten::TungstenFloatSession flt(s, 0, 1, MPI_COMM_WORLD);
    fill_sine(flt.psi().vec());
    flt.run();

    REQUIRE(max_abs_diff(ref.psi().vec(), flt.psi().vec()) < 1e-5);
    const auto cr = ref.field_checksum();
    const auto cf = flt.field_checksum();
    REQUIRE_THAT(cf.sumsq, WithinRel(cr.sumsq, 1e-5));
  }
}

// The pinned double golden of tungsten-cpu-golden, reproduced in float within
// 1e-5 relative. The double pin itself is unchanged (test_tungsten_physics).
TEST_CASE("TungstenFloatSession 32^3/10-step sine IC checksum within 1e-5 of "
          "the double pin",
          "[tungsten][float][baseline]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  json s = float_settings(32, 0.1, 0.01);
  s["model"]["params"]["n0"] = -0.4;
  s["model"]["params"]["T"] = 0.5;
  s["model"]["precision"] = "float";
  s["timestepping"]["saveat"] = 0.05;
  s["initial_conditions"] =
      json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.4}}});
  tungsten::TungstenFloatSession session(s, 0, 1, MPI_COMM_WORLD);
  fill_sine(session.psi().vec());
  session.run();
  double sum = 0.0;
  double sumsq = 0.0;
  for (float x : session.psi().vec()) {
    sum += x;
    sumsq += static_cast<double>(x) * static_cast<double>(x);
  }
  REQUIRE(session.psi().vec().size() == 32768);
  REQUIRE_THAT(sum, WithinRel(-13107.200000000043, 1e-5));
  REQUIRE_THAT(sumsq, WithinRel(5406.3450894885682, 1e-5));
}

TEST_CASE("TungstenFloatSession checkpoint restart matches the continuous run",
          "[tungsten][float][checkpoint]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  TempDir tmp("restart");
  json s = float_settings(8, 0.04, 0.01);
  s["model"]["precision"] = "float";

  tungsten::TungstenFloatSession full(s, 0, 1, MPI_COMM_WORLD);
  fill_sine(full.psi().vec());
  full.run();

  // Run two steps with checkpoints to publish step_2, then restart from it.
  json head = s;
  head["timestepping"]["t1"] = 0.02;
  head["checkpoint"] = {{"every", 2}, {"directory", (tmp.path / "ckpt").string()}};
  tungsten::TungstenFloatSession first(head, 0, 1, MPI_COMM_WORLD);
  fill_sine(first.psi().vec());
  first.run();
  REQUIRE(std::filesystem::exists(tmp.path / "ckpt" / "step_2" / "metadata.json"));
  json tail = s;
  tail["restart_from"] = (tmp.path / "ckpt" / "step_2").string();
  tungsten::TungstenFloatSession resumed(tail, 0, 1, MPI_COMM_WORLD);
  resumed.run();

  const auto &a = full.psi().vec();
  const auto &b = resumed.psi().vec();
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i] == b[i]); // float64 brick of float values restores exactly
  }
}

TEST_CASE("TungstenFloatSession runs the moving boundary on float fields",
          "[tungsten][float][moving_bc]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  tungsten::register_catalog();
  json s = float_settings(8, 0.02, 0.01);
  s["model"]["precision"] = "float";
  s["boundary_conditions"] = json::array({{{"target", "psi"},
                                           {"type", "moving"},
                                           {"rho_low", -0.464},
                                           {"rho_high", -0.10},
                                           {"width", 2.0},
                                           {"alpha", 1.0},
                                           {"disp", 1.0},
                                           {"xpos", 4.0}}});
  tungsten::TungstenFloatSession session(s, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(session.psi().vec().size() == 512);
  for (float v : session.psi().vec()) {
    REQUIRE(std::isfinite(v));
  }
}
