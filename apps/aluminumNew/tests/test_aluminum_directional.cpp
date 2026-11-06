// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// The directional preset is a host AluminumSession run. Temperature is the
// prescribed profile. The numbers below are that run's response, not a
// casting prediction and not a device result.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <aluminum/aluminum_session.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/kernel/field/crystal_observables.hpp>
#include <openpfc/kernel/simulation/checkpoint_service.hpp>

using nlohmann::json;

namespace {

constexpr double kLevel = 0.5;
constexpr double kThreshold = 0.1;

// Remove checkpoint data even if a Catch2 assertion interrupts the case.
struct TemporaryDirectory {
  std::filesystem::path root;
  explicit TemporaryDirectory(const std::filesystem::path &path) : root(path) {
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
  }
  ~TemporaryDirectory() noexcept {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
};

pfc::observables::FrontSample planar_front(aluminum::AluminumSession &session,
                                           bool rising) {
  auto &psi = session.psi();
  const auto n = psi.local_size();
  std::vector<double> values(static_cast<std::size_t>(n[0]) *
                             static_cast<std::size_t>(n[1]) *
                             static_cast<std::size_t>(n[2]));
  psi.for_each_owned([&](int i, int j, int k) {
    const std::size_t local =
        static_cast<std::size_t>(i) +
        static_cast<std::size_t>(n[0]) *
            (static_cast<std::size_t>(j) +
             static_cast<std::size_t>(n[1]) * static_cast<std::size_t>(k));
    values[local] = psi(i, j, k);
  });
  const auto origin = psi.origin();
  const auto spacing = psi.spacing();
  const auto global = psi.global_size();
  const pfc::observables::OwnedDensity density{
      .values = values,
      .global_size = {global[0], global[1], global[2]},
      .spacing = {spacing[0], spacing[1], spacing[2]},
      .origin = {origin[0], origin[1], origin[2]},
      .box = psi.box()};
  return pfc::observables::planar_front_position(
      density, MPI_COMM_WORLD, {.axis = 0, .level = kLevel, .rising = rising});
}

int threshold_column(aluminum::AluminumSession &session) {
  auto &psi = session.psi();
  const int nx = psi.local_size()[0];
  std::vector<double> column(static_cast<std::size_t>(nx),
                             -std::numeric_limits<double>::infinity());
  psi.for_each_owned([&](int i, int j, int k) {
    (void)j;
    (void)k;
    column[static_cast<std::size_t>(i)] =
        std::max(column[static_cast<std::size_t>(i)], psi(i, j, k));
  });
  int last = -1;
  for (int i = 0; i < nx; ++i) {
    if (column[static_cast<std::size_t>(i)] > kThreshold) last = i;
  }
  return last;
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  REQUIRE(a.size() == b.size());
  double gap = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    gap = std::max(gap, std::abs(a[i] - b[i]));
  }
  return gap;
}

json reservoir_state(const std::filesystem::path &dir, int step) {
  pfc::sim::CheckpointService reader({}, MPI_COMM_WORLD);
  const auto meta = reader.read_metadata(dir / ("step_" + std::to_string(step)));
  return meta.boundary_conditions.at(0).at("state");
}

} // namespace

TEST_CASE("aluminum directional preset runs a pulled front on the host session",
          "[aluminum][directional][preset]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) SKIP("aluminum directional preset requires one MPI rank");

  aluminum::register_catalog();
  const std::filesystem::path path =
      std::filesystem::path{ALUMINUM_INPUTS_JSON_DIR} / "aluminum_directional.json";
  json settings = pfc::ui::load_settings_file(path);
  const json &params = settings.at("model").at("params");
  REQUIRE(params.at("G_grid").get<double>() == Catch::Approx(2.0));
  REQUIRE(params.at("V_grid").get<double>() == Catch::Approx(0.1));
  REQUIRE(params.at("x_initial").get<double>() == Catch::Approx(60.0));
  REQUIRE(settings.at("boundary_conditions").at(0).at("type") == "moving");

  const auto root = std::filesystem::temp_directory_path() /
                    ("openpfc_al_directional_" + std::to_string(getpid()));
  const TemporaryDirectory scratch(root);
  std::filesystem::create_directories(root / "drive");

  json drive = settings;
  drive["checkpoint"] = {{"every", 2}, {"directory", (root / "drive").string()}};
  aluminum::AluminumSession session(drive, 0, 1, MPI_COMM_WORLD);
  const auto initial = planar_front(session, false);
  const int hot0 = threshold_column(session);
  REQUIRE(initial.validity == pfc::observables::FrontValidity::valid);
  REQUIRE(initial.crossings == 1);
  REQUIRE(initial.position > 35.0);
  REQUIRE(initial.position < 50.0);
  REQUIRE(hot0 > 0);

  const auto phys = aluminum::AluminumPhysics<>::from_json(params, session.domain(),
                                                           session.psi().box());
  REQUIRE(phys.temperature_variation(params.at("x_initial").get<double>(), 0.0) ==
          Catch::Approx(0.0).margin(1e-12));
  const double g = params.at("G_grid").get<double>();
  const double v = params.at("V_grid").get<double>();
  const double x0 = params.at("x_initial").get<double>();
  const double at_front = phys.temperature_variation(initial.position, 0.0);
  REQUIRE(at_front == Catch::Approx(g * (initial.position - x0)).margin(1e-9));
  REQUIRE(at_front < -1.0);
  const double horizon = settings.at("timestepping").at("t1").get<double>();
  REQUIRE(phys.temperature_variation(initial.position, horizon) ==
          Catch::Approx(at_front - g * v * horizon).margin(1e-9));
  double tvar_lo = 1e300;
  double tvar_hi = -1e300;
  const auto &domain = session.domain();
  for (const double t : {0.0, horizon}) {
    for (int i = 0; i < domain.size[0]; ++i) {
      const double tv = phys.temperature_variation(
          domain.origin[0] + static_cast<double>(i) * domain.spacing[0], t);
      tvar_lo = std::min(tvar_lo, tv);
      tvar_hi = std::max(tvar_hi, tv);
    }
  }
  REQUIRE(tvar_lo > phys.params.T_min - phys.params.T_const);
  REQUIRE(tvar_hi < phys.params.T_max - phys.params.T_const);

  session.run();
  const auto evolved = planar_front(session, false);
  const int hot_drive = threshold_column(session);
  const double sumsq_drive = session.field_checksum().sumsq;
  REQUIRE(evolved.validity == pfc::observables::FrontValidity::not_single_front);
  REQUIRE(hot_drive > hot0);
  REQUIRE(std::isfinite(sumsq_drive));
  REQUIRE(sumsq_drive > 0.0);

  const json early = reservoir_state(root / "drive", 2);
  const json mid = reservoir_state(root / "drive", 4);
  const json late = reservoir_state(root / "drive", 40);
  REQUIRE(early.at("first") == false);
  REQUIRE(mid.at("xpos").get<double>() > early.at("xpos").get<double>());
  REQUIRE(late.at("xpos").get<double>() + 1e-12 >= mid.at("xpos").get<double>());
  REQUIRE(late.at("xpos").get<double>() > initial.position);
  REQUIRE(late.at("idx").get<int>() >= hot_drive);
  REQUIRE(late.at("idx").get<int>() - hot_drive <= 6);

  json nopull = settings;
  nopull["model"]["params"]["V_grid"] = 0.0;
  aluminum::AluminumSession still(nopull, 0, 1, MPI_COMM_WORLD);
  still.run();
  json isothermal = nopull;
  isothermal["model"]["params"]["G_grid"] = 0.0;
  aluminum::AluminumSession flat(isothermal, 0, 1, MPI_COMM_WORLD);
  flat.run();
  const double sumsq_still = still.field_checksum().sumsq;
  const double sumsq_flat = flat.field_checksum().sumsq;
  const int hot_flat = threshold_column(flat);
  REQUIRE(std::abs(sumsq_drive - sumsq_still) > 1.0);
  REQUIRE(std::abs(sumsq_drive - sumsq_flat) > 20.0);
  REQUIRE(hot_drive > hot_flat);

  std::filesystem::create_directories(root / "head");
  json head = drive;
  head["timestepping"]["t1"] = 8.0;
  head["checkpoint"]["directory"] = (root / "head").string();
  aluminum::AluminumSession prefix(head, 0, 1, MPI_COMM_WORLD);
  prefix.run();
  REQUIRE(prefix.time().get_increment() == 8);

  std::filesystem::create_directories(root / "tail");
  json resumed = drive;
  resumed["restart_from"] = (root / "head" / "step_8").string();
  resumed["checkpoint"]["directory"] = (root / "tail").string();
  resumed.erase("initial_conditions");
  aluminum::AluminumSession tail(resumed, 0, 1, MPI_COMM_WORLD);
  REQUIRE(tail.time().get_increment() == 8);
  tail.run();
  REQUIRE(tail.time().get_increment() == 40);
  REQUIRE(max_abs_diff(session.psi().vec(), tail.psi().vec()) < 1e-12);
  const json resumed_state = reservoir_state(root / "tail", 40);
  REQUIRE(resumed_state == late);
}
