// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_inverse_convergence.cpp
 * @brief Drive the shipped inverse-homogenization stopping protocol.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <inverse_homogenization/inverse_checkpoint.hpp>
#include <inverse_homogenization/inverse_convergence.hpp>
#include <inverse_homogenization/simp_penalty.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::apps::inverse::apply_tracker;
using pfc::apps::inverse::capture_problem;
using pfc::apps::inverse::capture_tracker;
using pfc::apps::inverse::checkpoint_generation_name;
using pfc::apps::inverse::checkpoint_is_restartable;
using pfc::apps::inverse::checkpoint_matches_problem;
using pfc::apps::inverse::checkpoint_staging_dir;
using pfc::apps::inverse::continuation_fraction;
using pfc::apps::inverse::ConvergenceConfig;
using pfc::apps::inverse::ConvergenceMetrics;
using pfc::apps::inverse::ConvergenceTracker;
using pfc::apps::inverse::copy_design_buffer;
using pfc::apps::inverse::fill_voigt6;
using pfc::apps::inverse::format_checkpoint_text;
using pfc::apps::inverse::InverseCheckpoint;
using pfc::apps::inverse::is_terminal;
using pfc::apps::inverse::make_metrics;
using pfc::apps::inverse::params_frozen;
using pfc::apps::inverse::prepare_checkpoint_staging;
using pfc::apps::inverse::publish_checkpoint_generation;
using pfc::apps::inverse::read_checkpoint_text;
using pfc::apps::inverse::read_current_pointer;
using pfc::apps::inverse::read_dump_steps;
using pfc::apps::inverse::relative_norm_change;
using pfc::apps::inverse::relative_objective_change;
using pfc::apps::inverse::resolve_checkpoint_bundle;
using pfc::apps::inverse::store_voigt6;
using pfc::apps::inverse::termination_name;
using pfc::apps::inverse::TerminationReason;
using pfc::apps::inverse::write_dump_steps;

TEST_CASE("continuation fraction freezes at the last continuation step",
          "[inverse-conv][59]") {
  REQUIRE_THAT(continuation_fraction(0, 300), WithinAbs(0.0, 0.0));
  REQUIRE_THAT(continuation_fraction(299, 300), WithinAbs(1.0, 0.0));
  REQUIRE_THAT(continuation_fraction(500, 300), WithinAbs(1.0, 0.0));
  REQUIRE_THAT(continuation_fraction(0, 0), WithinAbs(1.0, 0.0));
  REQUIRE(!params_frozen(299, 300));
  REQUIRE(params_frozen(300, 300));
  REQUIRE(params_frozen(0, 0));
}

TEST_CASE("convergence cannot trigger during continuation", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 5;
  tr.cfg.max_steps = 100;
  tr.cfg.conv_window = 2;
  tr.cfg.verify_steps = 0;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  for (int s = 0; s < 5; ++s) {
    const auto r = tr.after_step(s, true, quiet);
    REQUIRE(r == TerminationReason::Running);
    REQUIRE_FALSE(tr.candidate);
  }
}

TEST_CASE("one quiet frozen step is not enough for window 20",
          "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 100;
  tr.cfg.conv_window = 20;
  tr.cfg.verify_steps = 0;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE_FALSE(tr.candidate);
  REQUIRE(tr.quiet_count == 1);
}

TEST_CASE("quiet counter resets when a frozen criterion fails",
          "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 100;
  tr.cfg.conv_window = 3;
  tr.cfg.verify_steps = 0;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  ConvergenceMetrics loud{};
  loud.quiet = false;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.quiet_count == 2);
  REQUIRE(tr.after_step(2, true, loud) == TerminationReason::Running);
  REQUIRE(tr.quiet_count == 0);
  REQUIRE_FALSE(tr.candidate);
}

TEST_CASE("objective relative change uses max(1, |J_prev|)", "[inverse-conv][59]") {
  REQUIRE_THAT(relative_objective_change(0.0, 1e-9), WithinAbs(1e-9, 1e-20));
  REQUIRE_THAT(relative_objective_change(2.0, 2.002), WithinAbs(0.001, 1e-15));
  REQUIRE(relative_objective_change(-0.5, -0.5) == 0.0);
}

TEST_CASE("tensor relative change floors the previous norm", "[inverse-conv][59]") {
  REQUIRE_THAT(relative_norm_change(1e-8, 0.0, 1e-30), WithinRel(1e22, 1e-6));
  REQUIRE_THAT(relative_norm_change(1e-6, 2.0, 1e-30), WithinRel(5e-7, 1e-12));
}

TEST_CASE("make_metrics uses post-projection design RMS not step_rms",
          "[inverse-conv][59]") {
  ConvergenceConfig cfg;
  cfg.tol_design = 1e-4;
  cfg.tol_objective = 1e-6;
  cfg.tol_tensor = 1e-4;
  const auto quiet = make_metrics(1e-5, 1.0, 1.0, 1e-8, 1.0, 0.0, cfg);
  REQUIRE(quiet.quiet);
  REQUIRE_THAT(quiet.design_rms, WithinAbs(1e-5, 0.0));
  const auto loud_design = make_metrics(0.018, 1.0, 1.0, 1e-8, 1.0, 0.0, cfg);
  REQUIRE_FALSE(loud_design.quiet);
}

TEST_CASE("max-step termination is distinct from convergence",
          "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 3;
  tr.cfg.conv_window = 20;
  tr.cfg.verify_steps = 100;
  ConvergenceMetrics loud{};
  loud.quiet = false;
  REQUIRE(tr.after_step(0, true, loud) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, loud) == TerminationReason::Running);
  REQUIRE(tr.after_step(2, true, loud) == TerminationReason::MaxSteps);
  REQUIRE(std::string(termination_name(TerminationReason::MaxSteps)) == "MAX_STEPS");
  REQUIRE(std::string(termination_name(TerminationReason::Converged)) ==
          "CONVERGED");
}

TEST_CASE("elasticity failure stops immediately", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 5000;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  REQUIRE(tr.after_step(0, false, quiet) == TerminationReason::ElasticityFailure);
}

TEST_CASE("verification hold rejects a false candidate", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 5000;
  tr.cfg.conv_window = 2;
  tr.cfg.verify_steps = 2;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  ConvergenceMetrics loud{};
  loud.quiet = false;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.candidate);
  REQUIRE(tr.after_step(2, true, loud) == TerminationReason::Running);
  REQUIRE_FALSE(tr.candidate);
  REQUIRE(tr.rejected_hold);
  REQUIRE(tr.quiet_count == 0);
}

TEST_CASE("verified hold after window reports CONVERGED", "[inverse-conv][59]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 5000;
  tr.cfg.conv_window = 2;
  tr.cfg.verify_steps = 2;
  ConvergenceMetrics quiet{};
  quiet.quiet = true;
  REQUIRE(tr.after_step(0, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(1, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(2, true, quiet) == TerminationReason::Running);
  REQUIRE(tr.after_step(3, true, quiet) == TerminationReason::Converged);
  REQUIRE(tr.verified);
}

TEST_CASE("criteria_hold requires all three tolerances", "[inverse-conv][59]") {
  ConvergenceConfig cfg;
  REQUIRE(criteria_hold(1e-5, 1e-7, 1e-5, cfg));
  REQUIRE_FALSE(criteria_hold(1e-3, 1e-7, 1e-5, cfg));
  REQUIRE_FALSE(criteria_hold(1e-5, 1e-4, 1e-5, cfg));
  REQUIRE_FALSE(criteria_hold(1e-5, 1e-7, 1e-3, cfg));
}

TEST_CASE("CSV iterate rows match the documented header width",
          "[inverse-conv][63]") {
  using pfc::apps::inverse::csv_field_count;
  using pfc::apps::inverse::csv_is_comment_line;
  using pfc::apps::inverse::format_inverse_csv_row;
  using pfc::apps::inverse::InverseCsvRow;
  using pfc::apps::inverse::kInverseCsvHeader;
  using pfc::apps::inverse::validate_inverse_csv;
  const auto ncol = csv_field_count(kInverseCsvHeader);
  REQUIRE(ncol == 26);
  InverseCsvRow row;
  row.termination = "CONVERGED";
  const auto line = format_inverse_csv_row(row);
  REQUIRE(csv_field_count(line) == ncol);
  const std::string truncated =
      "12,0.1,0.01,0.0,0.09,0.25,0.4,0.04,-0.008,0.3,0.0,1";
  REQUIRE(csv_field_count(truncated) != ncol);
  REQUIRE_FALSE(csv_is_comment_line(truncated));
  REQUIRE(csv_is_comment_line(
      "# FINAL_RECOMPUTE unpenalized C_H of certified accepted h"));
  REQUIRE(csv_is_comment_line("# CERTIFIED_STEP 12 termination CONVERGED"));
  std::istringstream ok(std::string(kInverseCsvHeader) + "\n" + line + "\n" +
                        "# FINAL_RECOMPUTE J_tensor=0.1 C11=0.04\n");
  REQUIRE(validate_inverse_csv(ok).empty());
  std::istringstream bad(std::string(kInverseCsvHeader) + "\n" + truncated + "\n");
  REQUIRE_FALSE(validate_inverse_csv(bad).empty());
}

TEST_CASE("first accepted state cannot seed a quiet window", "[inverse-conv][63]") {
  ConvergenceConfig cfg;
  auto m = make_metrics(0.0, 1.0, 1.0, 0.0, 1.0, 0.0, cfg);
  REQUIRE(m.quiet);
  m.quiet = false;
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 0;
  tr.cfg.max_steps = 100;
  tr.cfg.conv_window = 2;
  tr.cfg.verify_steps = 0;
  REQUIRE(tr.after_step(0, true, m) == TerminationReason::Running);
  REQUIRE(tr.quiet_count == 0);
  REQUIRE_FALSE(tr.candidate);
}

TEST_CASE("SIMP p=1 is identity and skips 0^0", "[inverse-conv][63]") {
  using pfc::apps::inverse::simp_chain;
  using pfc::apps::inverse::simp_density;
  REQUIRE(simp_density(0.0, 1.0) == 0.0);
  REQUIRE(simp_density(0.7, 1.0) == 0.7);
  REQUIRE(simp_density(1.0, 1.0) == 1.0);
  REQUIRE(simp_chain(0.0, 1.0) == 1.0);
  REQUIRE(simp_chain(0.4, 1.0) == 1.0);
  REQUIRE_THAT(simp_density(0.5, 3.0), WithinAbs(0.125, 1e-15));
  REQUIRE_THAT(simp_chain(0.5, 3.0), WithinAbs(0.75, 1e-15));
  REQUIRE(simp_density(0.0, 3.0) == 0.0);
  REQUIRE(simp_chain(0.0, 3.0) == 0.0);
  REQUIRE(simp_density(1.0, 3.0) == 1.0);
  REQUIRE(simp_chain(1.0, 3.0) == 3.0);
  REQUIRE(std::isfinite(simp_density(0.0, 3.0)));
  REQUIRE(std::isfinite(simp_chain(0.0, 3.0)));
  REQUIRE(std::isfinite(simp_chain(1e-300, 3.0)));
  REQUIRE(simp_density(-0.2, 3.0) == 0.0);
  REQUIRE(simp_chain(1.2, 3.0) == 3.0);
}

TEST_CASE("terminal reasons are distinct from RUNNING", "[inverse-conv][63]") {
  REQUIRE_FALSE(is_terminal(TerminationReason::Running));
  REQUIRE(is_terminal(TerminationReason::Converged));
  REQUIRE(is_terminal(TerminationReason::MaxSteps));
  REQUIRE(is_terminal(TerminationReason::ElasticityFailure));
}

TEST_CASE("copy_design_buffer restores a trailing in-place update",
          "[inverse-conv][63]") {
  const double certified[4] = {0.1, 0.2, 0.8, 0.9};
  double h[4] = {0.0, 0.0, 0.0, 0.0};
  copy_design_buffer(certified, h, 4);
  REQUIRE(h[0] == 0.1);
  REQUIRE(h[3] == 0.9);
  h[0] = 0.5;
  copy_design_buffer(certified, h, 4);
  REQUIRE(h[0] == 0.1);
}

struct DummyInvCfg {
  int nx{64}, ny{64}, nz{121};
  int continuation_steps{300};
  int conv_window{20};
  int verify_steps{100};
  int normalize{1};
  int project_volume{1};
  int n_el_iter{400};
  double dx{1.0};
  double E_solid{1.0}, nu_solid{0.3}, E_void{0.002}, nu_void{0.3};
  double volume{0.2576};
  double lambda_volume{1.0};
  double lambda_reg{0.05};
  double lambda_reg_end{0.2};
  double simp{1.0};
  double simp_end{2.0};
  double epsilon{2.0};
  double dt{0.04};
  double max_delta{0.04};
  double tol_design{1e-4};
  double tol_objective{1e-6};
  double tol_tensor{1e-4};
};

struct Tiny6 {
  double a[6][6]{};
  double &operator()(int i, int j) { return a[i][j]; }
  double operator()(int i, int j) const { return a[i][j]; }
};

TEST_CASE("checkpoint text round-trips tracker, C_H and problem id",
          "[inverse-conv][72]") {
  DummyInvCfg cfg;
  Tiny6 Ct{};
  Tiny6 W{};
  Ct(0, 0) = 0.04342;
  Ct(0, 1) = -0.00854;
  W(0, 0) = 1.0;
  InverseCheckpoint ck;
  capture_problem(ck, cfg, Ct, W);
  ck.next_step = 440;
  ck.max_steps = 5000;
  ck.quiet_count = 108;
  ck.verify_left = 12;
  ck.candidate = 1;
  ck.have_prev = 1;
  ck.n_snap = 5;
  ck.last_dumped = 39;
  ck.J_prev = 0.123456789;
  Tiny6 C{};
  C(0, 0) = 0.04;
  C(0, 1) = -0.008;
  store_voigt6(ck.C_prev, C);
  std::istringstream in(format_checkpoint_text(ck));
  InverseCheckpoint got;
  REQUIRE(read_checkpoint_text(in, got));
  REQUIRE(got.next_step == 440);
  REQUIRE(got.quiet_count == 108);
  REQUIRE(got.verify_left == 12);
  REQUIRE(got.candidate == 1);
  REQUIRE(got.n_snap == 5);
  REQUIRE(got.last_dumped == 39);
  REQUIRE_THAT(got.J_prev, WithinAbs(0.123456789, 1e-15));
  Tiny6 C2{};
  fill_voigt6(C2, got.C_prev);
  REQUIRE_THAT(C2(0, 0), WithinAbs(0.04, 1e-15));
  REQUIRE_THAT(C2(0, 1), WithinAbs(-0.008, 1e-15));
  Tiny6 Ct2{};
  fill_voigt6(Ct2, got.C_target);
  REQUIRE_THAT(Ct2(0, 0), WithinAbs(0.04342, 1e-15));
  ConvergenceTracker tr;
  apply_tracker(got, tr);
  REQUIRE(tr.candidate);
  REQUIRE(tr.quiet_count == 108);
  REQUIRE_THAT(tr.cfg.tol_design, WithinAbs(1e-4, 1e-18));
  REQUIRE(checkpoint_matches_problem(got, cfg, Ct, W));
  DummyInvCfg other = cfg;
  other.nx = 32;
  REQUIRE_FALSE(checkpoint_matches_problem(got, other, Ct, W));
}

TEST_CASE("checkpoint rejects a changed frozen problem", "[inverse-conv][72]") {
  DummyInvCfg cfg;
  Tiny6 Ct{};
  Tiny6 W{};
  Ct(0, 0) = 0.04;
  W(0, 0) = 1.0;
  InverseCheckpoint ck;
  capture_problem(ck, cfg, Ct, W);
  ck.next_step = 10;
  ck.max_steps = 100;
  ck.have_prev = 1;
  std::istringstream in(format_checkpoint_text(ck));
  InverseCheckpoint got;
  REQUIRE(read_checkpoint_text(in, got));
  REQUIRE(checkpoint_matches_problem(got, cfg, Ct, W));
  got.max_steps = 9999;
  REQUIRE(checkpoint_matches_problem(got, cfg, Ct, W));
  Tiny6 Ct2 = Ct;
  Ct2(0, 0) = 0.05;
  REQUIRE_FALSE(checkpoint_matches_problem(got, cfg, Ct2, W));
  DummyInvCfg nrm = cfg;
  nrm.normalize = 0;
  REQUIRE_FALSE(checkpoint_matches_problem(got, nrm, Ct, W));
  DummyInvCfg tol = cfg;
  tol.tol_design = 1e-3;
  REQUIRE_FALSE(checkpoint_matches_problem(got, tol, Ct, W));
  DummyInvCfg lr = cfg;
  lr.lambda_reg_end = 0.5;
  REQUIRE_FALSE(checkpoint_matches_problem(got, lr, Ct, W));
  DummyInvCfg sm = cfg;
  sm.simp_end = 3.0;
  REQUIRE_FALSE(checkpoint_matches_problem(got, sm, Ct, W));
}

TEST_CASE("checkpoint text rejects schema 1 and a bad magic line",
          "[inverse-conv][72]") {
  InverseCheckpoint ck;
  std::istringstream old_schema("OPENPFC_INVERSE_CHECKPOINT 1\nnx 8\n");
  REQUIRE_FALSE(read_checkpoint_text(old_schema, ck));
  std::istringstream schema2("OPENPFC_INVERSE_CHECKPOINT 2\nnx 8\n");
  REQUIRE_FALSE(read_checkpoint_text(schema2, ck));
  std::istringstream in("NOT_A_CHECKPOINT 3\n");
  REQUIRE_FALSE(read_checkpoint_text(in, ck));
}

TEST_CASE("a terminal checkpoint is not restartable as continuation",
          "[inverse-conv][72]") {
  DummyInvCfg cfg;
  Tiny6 Ct{};
  Tiny6 W{};
  InverseCheckpoint ck;
  capture_problem(ck, cfg, Ct, W);
  ck.next_step = 420;
  ck.max_steps = 5000;
  ck.have_prev = 1;
  ck.quiet_count = 120;
  ck.candidate = ck.verified = 1;
  ck.termination = static_cast<int>(TerminationReason::Converged);
  REQUIRE_FALSE(checkpoint_is_restartable(ck, 5000));
  std::istringstream in(format_checkpoint_text(ck));
  InverseCheckpoint got;
  REQUIRE(read_checkpoint_text(in, got));
  REQUIRE(got.termination == static_cast<int>(TerminationReason::Converged));
  REQUIRE_FALSE(checkpoint_is_restartable(got, 5000));
  got.termination = static_cast<int>(TerminationReason::Running);
  REQUIRE(checkpoint_is_restartable(got, 5000));
}

TEST_CASE("restart requires an iteration remaining in the requested budget",
          "[inverse-conv][94]") {
  InverseCheckpoint ck;
  ck.next_step = 4;
  ck.max_steps = 9;
  ck.termination = static_cast<int>(TerminationReason::Running);
  for (int budget : {0, 3, 4}) REQUIRE_FALSE(checkpoint_is_restartable(ck, budget));
  for (int budget : {5, 9, 100}) REQUIRE(checkpoint_is_restartable(ck, budget));
  for (auto reason : {TerminationReason::Converged, TerminationReason::MaxSteps,
                      TerminationReason::ElasticityFailure}) {
    ck.termination = static_cast<int>(reason);
    REQUIRE_FALSE(checkpoint_is_restartable(ck, 100));
  }
}

TEST_CASE("capture/apply tracker preserves window and hold", "[inverse-conv][72]") {
  ConvergenceTracker tr;
  tr.cfg.continuation_steps = 300;
  tr.cfg.max_steps = 5000;
  tr.cfg.conv_window = 20;
  tr.cfg.verify_steps = 100;
  tr.quiet_count = 73;
  tr.candidate = true;
  tr.verify_left = 47;
  tr.verified = false;
  InverseCheckpoint ck;
  capture_tracker(ck, tr, 412);
  ConvergenceTracker got;
  apply_tracker(ck, got);
  REQUIRE(ck.next_step == 412);
  REQUIRE(got.quiet_count == 73);
  REQUIRE(got.candidate);
  REQUIRE(got.verify_left == 47);
  REQUIRE_FALSE(got.verified);
  REQUIRE(got.cfg.continuation_steps == 300);
  REQUIRE(got.cfg.max_steps == 5000);
}

namespace {

struct ScratchDir {
  std::filesystem::path path;
  ScratchDir() {
    path = std::filesystem::temp_directory_path() /
           ("openpfc_inv_ckpt_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }
  ~ScratchDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_dummy_bundle(const std::filesystem::path &dir) {
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "h.bin") << "h";
  std::ofstream(dir / "h_prev.bin") << "p";
  std::ofstream(dir / "state.txt") << "s";
}

} // namespace

TEST_CASE("incomplete staging does not replace the published generation",
          "[inverse-conv][72]") {
  ScratchDir tmp;
  const auto gen1 = checkpoint_generation_name(4);
  const auto gen2 = checkpoint_generation_name(8);
  const auto gen3 = checkpoint_generation_name(12);
  REQUIRE(prepare_checkpoint_staging(tmp.path));
  write_dummy_bundle(checkpoint_staging_dir(tmp.path));
  REQUIRE(publish_checkpoint_generation(tmp.path, gen1));
  REQUIRE(resolve_checkpoint_bundle(tmp.path) == tmp.path / gen1);

  REQUIRE(prepare_checkpoint_staging(tmp.path));
  std::ofstream(checkpoint_staging_dir(tmp.path) / "h.bin") << "partial";
  REQUIRE(resolve_checkpoint_bundle(tmp.path) == tmp.path / gen1);
  REQUIRE_FALSE(publish_checkpoint_generation(tmp.path, gen2));
  REQUIRE(read_current_pointer(tmp.path) == gen1);

  write_dummy_bundle(checkpoint_staging_dir(tmp.path));
  REQUIRE(publish_checkpoint_generation(tmp.path, gen2));
  REQUIRE(resolve_checkpoint_bundle(tmp.path) == tmp.path / gen2);
  REQUIRE(std::filesystem::is_directory(tmp.path / gen1));

  REQUIRE(prepare_checkpoint_staging(tmp.path));
  write_dummy_bundle(checkpoint_staging_dir(tmp.path));
  REQUIRE(publish_checkpoint_generation(tmp.path, gen3));
  REQUIRE(resolve_checkpoint_bundle(tmp.path) == tmp.path / gen3);
  REQUIRE(std::filesystem::is_directory(tmp.path / gen2));
  REQUIRE_FALSE(std::filesystem::exists(tmp.path / gen1));
}

TEST_CASE("unpublished generation is ignored until CURRENT is retargeted",
          "[inverse-conv][72]") {
  ScratchDir tmp;
  const auto gen1 = checkpoint_generation_name(1);
  const auto gen2 = checkpoint_generation_name(2);
  REQUIRE(prepare_checkpoint_staging(tmp.path));
  write_dummy_bundle(checkpoint_staging_dir(tmp.path));
  REQUIRE(publish_checkpoint_generation(tmp.path, gen1));
  REQUIRE(prepare_checkpoint_staging(tmp.path));
  write_dummy_bundle(checkpoint_staging_dir(tmp.path));
  std::filesystem::rename(checkpoint_staging_dir(tmp.path), tmp.path / gen2);
  REQUIRE(resolve_checkpoint_bundle(tmp.path) == tmp.path / gen1);
  REQUIRE(prepare_checkpoint_staging(tmp.path));
  write_dummy_bundle(checkpoint_staging_dir(tmp.path));
  REQUIRE(publish_checkpoint_generation(tmp.path, gen2));
  REQUIRE(resolve_checkpoint_bundle(tmp.path) == tmp.path / gen2);
}

TEST_CASE("dump_steps text round-trips the snapshot index map",
          "[inverse-conv][72]") {
  ScratchDir tmp;
  const auto path = tmp.path / "dump_steps.txt";
  const std::vector<int> want{0, 20, 40, 60};
  REQUIRE(write_dump_steps(path, want));
  std::vector<int> got;
  REQUIRE(read_dump_steps(path, got, 4, 60));
  REQUIRE(got == want);
}

TEST_CASE("explicit generation directory loads without a CURRENT pointer",
          "[inverse-conv][72]") {
  ScratchDir tmp;
  const auto gen = tmp.path / checkpoint_generation_name(9);
  write_dummy_bundle(gen);
  REQUIRE(resolve_checkpoint_bundle(gen) == gen);
  REQUIRE(resolve_checkpoint_bundle(tmp.path).empty());
}

TEST_CASE("checkpoint accepts reachable tracker histories and resumes holds",
          "[inverse-conv][91]") {
  using namespace pfc::apps::inverse;
  // Enumerate all quiet/nonquiet histories through a small finite horizon,
  // including no-hold convergence, interrupted holds and continuation freeze.
  for (int continuation : {0, 2})
    for (int window : {1, 3})
      for (int hold : {0, 2, 3})
        for (unsigned mask = 0; mask < 256; ++mask) {
          ConvergenceTracker tr;
          tr.cfg = {continuation, 8, window, hold};
          for (int step = 0; step < 8; ++step) {
            auto failed = tr;
            ConvergenceMetrics m;
            m.quiet = step > 0 && ((mask >> step) & 1);
            const auto failure = failed.after_step(step, false, m);
            const auto reason = tr.after_step(step, true, m);
            auto roundtrip = [&](const ConvergenceTracker &state,
                                 TerminationReason why) {
              InverseCheckpoint ck;
              Tiny6 tensor;
              capture_problem(ck, DummyInvCfg{}, tensor, tensor);
              capture_tracker(ck, state, step + 1);
              ck.have_prev = 1;
              ck.termination = static_cast<int>(why);
              REQUIRE(checkpoint_state_valid(ck));
              std::istringstream input(format_checkpoint_text(ck));
              InverseCheckpoint restored;
              REQUIRE(read_checkpoint_text(input, restored));
              ConvergenceTracker resumed;
              apply_tracker(restored, resumed);
              REQUIRE(resumed.quiet_count == state.quiet_count);
              REQUIRE(resumed.verify_left == state.verify_left);
              if (why == TerminationReason::Running) {
                auto reference = state;
                for (int next = step + 1; next < 8; ++next) {
                  ConvergenceMetrics subsequent;
                  subsequent.quiet = ((mask >> next) & 1) != 0;
                  const auto a = reference.after_step(next, true, subsequent);
                  const auto b = resumed.after_step(next, true, subsequent);
                  REQUIRE(a == b);
                  REQUIRE(reference.quiet_count == resumed.quiet_count);
                  REQUIRE(reference.verify_left == resumed.verify_left);
                  REQUIRE(reference.verified == resumed.verified);
                  if (is_terminal(a)) break;
                }
              }
            };
            roundtrip(failed, failure);
            roundtrip(tr, reason);
            if (is_terminal(reason)) break;
          }
        }
}

TEST_CASE("checkpoint rejects impossible counters and accepted state",
          "[inverse-conv][91]") {
  using namespace pfc::apps::inverse;
  InverseCheckpoint valid;
  Tiny6 tensor;
  capture_problem(valid, DummyInvCfg{}, tensor, tensor);
  valid.next_step = 340;
  valid.max_steps = 5000;
  valid.have_prev = 1;
  valid.quiet_count = 25;
  valid.candidate = 1;
  valid.verify_left = 95;
  REQUIRE(checkpoint_state_valid(valid));
  const std::vector<std::function<void(InverseCheckpoint &)>> corruptions = {
      [](auto &c) { c.verify_left = -1; },
      [](auto &c) { c.verify_left = 94; },
      [](auto &c) { c.quiet_count = 19; },
      [](auto &c) { c.quiet_count = -1; },
      [](auto &c) { c.next_step = 310; },
      [](auto &c) { c.candidate = 2; },
      [](auto &c) { c.candidate = 0; },
      [](auto &c) { c.verified = -1; },
      [](auto &c) { c.verified = 1; },
      [](auto &c) { c.have_prev = 0; },
      [](auto &c) { c.normalize = 2; },
      [](auto &c) { c.project_volume = -1; },
      [](auto &c) { c.conv_window = 0; },
      [](auto &c) { c.verify_steps = -1; },
      [](auto &c) { c.continuation_steps = -1; },
      [](auto &c) { c.max_steps = 339; },
      [](auto &c) { c.max_steps = 340; },
      [](auto &c) { c.termination = 1; },
      [](auto &c) { c.termination = 2; },
      [](auto &c) { c.dx = 0; },
      [](auto &c) { c.tol_design = -1; },
      [](auto &c) { c.J_prev = std::numeric_limits<double>::infinity(); },
      [](auto &c) { c.C_prev[35] = std::numeric_limits<double>::quiet_NaN(); },
      [](auto &c) { c.W[2] = std::numeric_limits<double>::infinity(); }};
  for (const auto &corrupt : corruptions) {
    auto ck = valid;
    corrupt(ck);
    REQUIRE_FALSE(checkpoint_state_valid(ck));
    std::istringstream input(format_checkpoint_text(ck));
    InverseCheckpoint restored;
    REQUIRE_FALSE(read_checkpoint_text(input, restored));
  }
}

TEST_CASE("checkpoint metadata rejects delayed writes and preserves CURRENT",
          "[inverse-conv][96]") {
  using namespace pfc::apps::inverse;
  ScratchDir tmp;
  REQUIRE_FALSE(write_checkpoint_file(tmp.path, InverseCheckpoint{}));
  REQUIRE_FALSE(write_dump_steps(tmp.path, {0, 1}));
  if (std::filesystem::exists("/dev/full")) {
    REQUIRE_FALSE(write_checkpoint_file("/dev/full", InverseCheckpoint{}));
    REQUIRE_FALSE(write_dump_steps("/dev/full", {0, 1}));
    REQUIRE(prepare_checkpoint_staging(tmp.path));
    write_dummy_bundle(checkpoint_staging_dir(tmp.path));
    REQUIRE(publish_checkpoint_generation(tmp.path, "gen_1"));
    REQUIRE(prepare_checkpoint_staging(tmp.path));
    write_dummy_bundle(checkpoint_staging_dir(tmp.path));
    std::filesystem::create_symlink("/dev/full", tmp.path / "CURRENT.tmp");
    REQUIRE_FALSE(publish_checkpoint_generation(tmp.path, "gen_2"));
    REQUIRE(read_current_pointer(tmp.path) == "gen_1");
    REQUIRE(resolve_checkpoint_bundle(tmp.path) == tmp.path / "gen_1");
    std::filesystem::remove(tmp.path / "CURRENT.tmp");
    REQUIRE(write_current_pointer(tmp.path, "gen_2"));
  }
}

TEST_CASE("snapshot ledger rejects corrupt or inconsistent restoration",
          "[inverse-conv][96]") {
  using namespace pfc::apps::inverse;
  ScratchDir tmp;
  const auto path = tmp.path / "dump_steps.txt";
  std::vector<int> got{99};
  REQUIRE_FALSE(read_dump_steps(path, got, 3, 2));
  REQUIRE(got.empty());
  for (const std::string text :
       {"0 1", "0 1 2 3", "0 1 junk", "0 1 99999999999999999999", "0 0 2", "-1 1 2",
        "1 0 2", "0 1 3"}) {
    std::ofstream(path) << text;
    REQUIRE_FALSE(read_dump_steps(path, got, 3, 2));
    REQUIRE(got.empty());
  }
  std::ofstream(path) << "0 1 2\n";
  REQUIRE(read_dump_steps(path, got, 3, 2));
  REQUIRE(got == std::vector<int>{0, 1, 2});
  std::ofstream(path) << "";
  REQUIRE(read_dump_steps(path, got, 0, -1));
  REQUIRE_FALSE(read_dump_steps(path, got, 0, 0));
}

TEST_CASE("checkpoint field sizes reject incomplete publication",
          "[inverse-conv][96]") {
  using namespace pfc::apps::inverse;
  ScratchDir tmp;
  InverseCheckpoint ck;
  ck.nx = ck.ny = ck.nz = 1;
  std::ofstream(tmp.path / "h.bin", std::ios::binary) << std::string(8, '\0');
  std::ofstream(tmp.path / "h_prev.bin", std::ios::binary) << std::string(7, '\0');
  REQUIRE_FALSE(checkpoint_field_sizes_match(tmp.path, ck));
  std::ofstream(tmp.path / "h_prev.bin", std::ios::binary) << std::string(8, '\0');
  REQUIRE(checkpoint_field_sizes_match(tmp.path, ck));
  ck.nx = ck.ny = ck.nz = std::numeric_limits<int>::max();
  REQUIRE_FALSE(checkpoint_field_sizes_match(tmp.path, ck));
}
