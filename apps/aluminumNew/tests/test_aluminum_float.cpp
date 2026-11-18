// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_aluminum_float.cpp
 * @brief AluminumNew in single precision (ADR 0006, #870): the
 *        `model.precision` switch, the float session against the double
 *        reference, the pinned seed_grid_fcc 32^3 checksum within tolerance,
 *        restart equivalence and the directional preset with its moving
 *        boundary on float fields. The double goldens are untouched.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <aluminum/aluminum_physics.hpp>
#include <aluminum/aluminum_session.hpp>
#include <openpfc/frontend/ui/precision_json.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
using nlohmann::json;
using pfc::ui::Precision;
using pfc::ui::precision_from_json;

namespace {

/// Every field the schema declares, and nothing it does not.
json params_json() {
  return {{"alpha", 0.20},
          {"T_const", 980.0},
          {"T_min", 780.0},
          {"T_max", 1280.0},
          {"T0", 89285.0},
          {"Bx", 0.817900686921996},
          {"G_grid", 0.0},
          {"V_grid", 0.0},
          {"x_initial", 130.0},
          {"lambda", 0.22},
          {"stabP", 0.0},
          {"p2_bar", 0.8286531831},
          {"p3_bar", -0.04204863},
          {"p4_bar", 0.007533},
          {"q20_bar", 0.016531729105214},
          {"q21_bar", 5.467},
          {"q30_bar", 1.7152418049986},
          {"q31_bar", 0.45},
          {"q40_bar", 0.787482}};
}

json seed_ic(double x0, int ny, int nz, double radius) {
  return {{"target", "psi"},  {"type", "seed_grid_fcc"},
          {"X0", x0},         {"Ny", ny},
          {"Nz", nz},         {"radius", radius},
          {"amplitude", 0.4}, {"rho", -0.036},
          {"rseed", 42}};
}

json settings(int n, double t1, double dt) {
  return {
      {"model", {{"name", "aluminum"}, {"params", params_json()}}},
      {"domain",
       {{"size", {n, n, n}}, {"lower", {0.0, 0.0, 0.0}}, {"upper", {n, n, n}}}},
      {"timestepping", {{"t0", 0.0}, {"t1", t1}, {"dt", dt}, {"saveat", dt}}},
      {"initial_conditions",
       json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.0060}}})}};
}

/// The 32^3 / 5-step case of `aluminum-etd-cpu-golden` (aluminumTest.cpp).
json golden_seed_settings() {
  json s = settings(32, 0.05, 0.01);
  s["timestepping"]["saveat"] = 0.05;
  s["initial_conditions"].push_back(seed_ic(8.0, 2, 2, 4.0));
  return s;
}

template <class Real>
double max_abs_diff(const std::vector<double> &a, const std::vector<Real> &b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(std::isfinite(b[i]));
    m = std::max(m, std::abs(a[i] - static_cast<double>(b[i])));
  }
  return m;
}

template <class Session> int threshold_column(Session &session) {
  auto &psi = session.psi();
  const int nx = psi.local_size()[0];
  std::vector<double> column(static_cast<std::size_t>(nx),
                             -std::numeric_limits<double>::infinity());
  psi.for_each_owned([&](int i, int j, int k) {
    column[static_cast<std::size_t>(i)] = std::max(
        column[static_cast<std::size_t>(i)], static_cast<double>(psi(i, j, k)));
  });
  int last = -1;
  for (int i = 0; i < nx; ++i) {
    if (column[static_cast<std::size_t>(i)] > 0.1) last = i;
  }
  return last;
}

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char *tag) {
    path = std::filesystem::temp_directory_path() /
           (std::string("openpfc_aluminum_float_") + tag + "_" +
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

bool single_rank() {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  return nproc == 1;
}

} // namespace

static_assert(std::is_same_v<aluminum::AluminumFloatSession::Real, float>);
static_assert(std::is_same_v<aluminum::AluminumSession::Real, double>);
static_assert(std::is_same_v<aluminum::AluminumPhysics<float>::real_type, float>);
static_assert(pfc::sim::SpectralETDPhysics<aluminum::AluminumPhysics<float>>);

TEST_CASE("a session rejects model.precision naming the other scalar",
          "[aluminum][float][precision]") {
  json s = settings(8, 0.01, 0.01);
  REQUIRE(precision_from_json(s) == Precision::Double);
  s["model"]["precision"] = "float";
  REQUIRE(precision_from_json(s) == Precision::Float);
  REQUIRE_THROWS_WITH((aluminum::AluminumSession(s, 0, 1, MPI_COMM_WORLD)),
                      ContainsSubstring("model.precision"));
  s["model"]["precision"] = "double";
  REQUIRE_THROWS_WITH((aluminum::AluminumFloatSession(s, 0, 1, MPI_COMM_WORLD)),
                      ContainsSubstring("model.precision"));
  // Absent key: either session accepts the input.
  s["model"].erase("precision");
  REQUIRE_NOTHROW(aluminum::AluminumFloatSession(s, 0, 1, MPI_COMM_WORLD));
}

TEST_CASE("AluminumFloatSession declares float fields and runs",
          "[aluminum][float][session]") {
  if (!single_rank()) SKIP("one MPI rank");
  json s = settings(8, 0.02, 0.01);
  s["model"]["precision"] = "float";
  aluminum::AluminumFloatSession session(s, 0, 1, MPI_COMM_WORLD);
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
// the double session on the seed_grid_fcc 32^3 / 5-step golden case within
// 1e-5, and reproduces the pinned double checksum within 1e-5 relative. The
// double pin itself is unchanged (aluminumTest.cpp).
TEST_CASE("AluminumFloatSession seed_grid_fcc 32^3/5-step checksum within 1e-5 "
          "of the double golden",
          "[aluminum][float][baseline]") {
  if (!single_rank()) SKIP("one MPI rank");
  aluminum::register_catalog();
  json s = golden_seed_settings();
  aluminum::AluminumSession ref(s, 0, 1, MPI_COMM_WORLD);
  ref.run();

  s["model"]["precision"] = "float";
  aluminum::AluminumFloatSession flt(s, 0, 1, MPI_COMM_WORLD);
  flt.run();

  double sum = 0.0;
  double sumsq = 0.0;
  for (float x : flt.psi().vec()) {
    sum += x;
    sumsq += static_cast<double>(x) * static_cast<double>(x);
  }
  const double gap = max_abs_diff(ref.psi().vec(), flt.psi().vec());
  std::cout << std::setprecision(17)
            << "FLOAT_BASELINE aluminum_etd n=" << flt.psi().vec().size()
            << " sum=" << sum << " sumsq=" << sumsq << " max_abs_diff=" << gap
            << '\n';
  REQUIRE(flt.psi().vec().size() == 32768);
  REQUIRE(gap < 1e-5);
  REQUIRE_THAT(sum, WithinRel(-263.63079658808601, 1e-5));
  REQUIRE_THAT(sumsq, WithinRel(1111.6016268617182, 1e-5));
}

// The temperature profile and its clamp run in double inside the float
// functor; the float trajectory with G_grid != 0 stays within 1e-5 of double.
TEST_CASE("AluminumFloatSession with the temperature profile tracks the double "
          "session within 1e-5",
          "[aluminum][float][session][temperature]") {
  if (!single_rank()) SKIP("one MPI rank");
  aluminum::register_catalog();
  json s = settings(16, 0.1, 0.01);
  s["model"]["params"]["G_grid"] = 2.0;
  s["model"]["params"]["V_grid"] = 0.1;
  s["model"]["params"]["x_initial"] = 8.0;
  s["initial_conditions"].push_back(seed_ic(4.0, 1, 1, 4.0));

  aluminum::AluminumSession ref(s, 0, 1, MPI_COMM_WORLD);
  ref.run();
  s["model"]["precision"] = "float";
  aluminum::AluminumFloatSession flt(s, 0, 1, MPI_COMM_WORLD);
  flt.run();
  REQUIRE(max_abs_diff(ref.psi().vec(), flt.psi().vec()) < 1e-5);
  REQUIRE_THAT(flt.field_checksum().sumsq,
               WithinRel(ref.field_checksum().sumsq, 1e-5));
}

TEST_CASE("AluminumFloatSession checkpoint restart matches the continuous run",
          "[aluminum][float][checkpoint]") {
  if (!single_rank()) SKIP("one MPI rank");
  aluminum::register_catalog();
  TempDir tmp("restart");
  json s = settings(8, 0.04, 0.01);
  s["initial_conditions"].push_back(seed_ic(4.0, 1, 1, 3.0));
  s["model"]["precision"] = "float";

  aluminum::AluminumFloatSession full(s, 0, 1, MPI_COMM_WORLD);
  full.run();

  json head = s;
  head["timestepping"]["t1"] = 0.02;
  head["checkpoint"] = {{"every", 2}, {"directory", (tmp.path / "ckpt").string()}};
  aluminum::AluminumFloatSession first(head, 0, 1, MPI_COMM_WORLD);
  first.run();
  REQUIRE(std::filesystem::exists(tmp.path / "ckpt" / "step_2" / "metadata.json"));

  json tail = s;
  tail["restart_from"] = (tmp.path / "ckpt" / "step_2").string();
  tail.erase("initial_conditions");
  aluminum::AluminumFloatSession resumed(tail, 0, 1, MPI_COMM_WORLD);
  resumed.run();

  const auto &a = full.psi().vec();
  const auto &b = resumed.psi().vec();
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i] == b[i]); // float64 brick of float values restores exactly
  }
}

// The shipped directional preset (pulled front, moving boundary in `cutoff`
// mode) runs on float fields; the boundary and the front tracking go through
// the double scratch of `apply_field_modifier`. The front advances as in the
// double run; the field is compared with a looser tolerance than the
// isothermal cases (1e-4 instead of 1e-5), since the cutoff detection is a
// discrete decision that float rounding could move by one column.
TEST_CASE("AluminumFloatSession runs the directional preset with the moving "
          "boundary",
          "[aluminum][float][directional]") {
  if (!single_rank()) SKIP("one MPI rank");
  aluminum::register_catalog();
  const std::filesystem::path path =
      std::filesystem::path{ALUMINUM_INPUTS_JSON_DIR} / "aluminum_directional.json";
  json s = pfc::ui::load_settings_file(path);
  REQUIRE(s.at("boundary_conditions").at(0).at("type") == "moving");

  aluminum::AluminumSession ref(s, 0, 1, MPI_COMM_WORLD);
  const int hot0 = threshold_column(ref);
  ref.run();

  s["model"]["precision"] = "float";
  aluminum::AluminumFloatSession flt(s, 0, 1, MPI_COMM_WORLD);
  REQUIRE(threshold_column(flt) == hot0);
  flt.run();

  for (float v : flt.psi().vec()) {
    REQUIRE(std::isfinite(v));
  }
  const int hot_ref = threshold_column(ref);
  const int hot_flt = threshold_column(flt);
  const double gap = max_abs_diff(ref.psi().vec(), flt.psi().vec());
  std::cout << std::setprecision(17)
            << "FLOAT_DIRECTIONAL aluminum_etd hot0=" << hot0
            << " hot_double=" << hot_ref << " hot_float=" << hot_flt
            << " sumsq_double=" << ref.field_checksum().sumsq
            << " sumsq_float=" << flt.field_checksum().sumsq
            << " max_abs_diff=" << gap << '\n';
  REQUIRE(hot_flt > hot0);
  REQUIRE(std::abs(hot_flt - hot_ref) <= 1);
  REQUIRE(gap < 1e-4);
  REQUIRE_THAT(flt.field_checksum().sumsq,
               WithinRel(ref.field_checksum().sumsq, 1e-4));
}

TEST_CASE("the shipped float preset loads and names float",
          "[aluminum][float][inputs]") {
  const std::filesystem::path path =
      std::filesystem::path{ALUMINUM_INPUTS_JSON_DIR} / "fcc_seed_float.json";
  const json s = pfc::ui::load_settings_file(path);
  REQUIRE(pfc::ui::has_precision_key(s));
  REQUIRE(precision_from_json(s) == Precision::Float);
  REQUIRE_NOTHROW(
      aluminum::make_aluminum_schema().parse(s.at("model").at("params")));
}
