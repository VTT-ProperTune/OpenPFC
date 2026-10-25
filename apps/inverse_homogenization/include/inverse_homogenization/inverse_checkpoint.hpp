// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file inverse_checkpoint.hpp
 * @brief Durable restart bundle for frozen inverse homogenization.
 *
 * A restart continues the same mathematical optimization: incoming design
 * `h`, previous accepted `h_prev`, absolute iterate index, continuation
 * freeze, tracker window/hold, previous `J` / `C_H`, and an explicit
 * problem fingerprint (target tensor, moduli, SIMP/regularization
 * endpoints, step/projection, tolerances). It does not reset convergence
 * history or unfreeze SIMP / `lambda_reg`. `--max-steps` is a run budget
 * and may change after a walltime restart; everything else that defines
 * the frozen inverse must match or the restart is rejected. A terminal
 * `CONVERGED` / `MAX_STEPS` / `ELASTICITY_FAILURE` bundle is a certified
 * snapshot and is not a continuation restart.
 *
 * Drivers publish a complete generation directory (`gen_<next_step>/`
 * with `h.bin`, `h_prev.bin`, `state.txt`, optional `dump_steps.txt`)
 * and then atomically retarget `CURRENT`. A kill during the write leaves
 * the previous published generation loadable.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <istream>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <inverse_homogenization/inverse_convergence.hpp>

namespace pfc::apps::inverse {

inline constexpr int kInverseCheckpointSchema = 3;
inline constexpr std::string_view kInverseCheckpointMagic =
    "OPENPFC_INVERSE_CHECKPOINT";
inline constexpr std::string_view kInverseCheckpointCurrent = "CURRENT";
inline constexpr std::string_view kInverseCheckpointStaging = ".writing";

struct InverseCheckpoint {
  int schema{kInverseCheckpointSchema};
  int nx{0}, ny{0}, nz{0};
  int next_step{0};
  int continuation_steps{0};
  int max_steps{0};
  int conv_window{0};
  int verify_steps{0};
  int quiet_count{0};
  int verify_left{0};
  int candidate{0};
  int verified{0};
  int have_prev{0};
  int n_snap{0};
  int last_dumped{-1};
  int termination{0};
  int normalize{1};
  int project_volume{0};
  int n_el_iter{0};
  double dx{1.0};
  double E_solid{0.0}, nu_solid{0.0}, E_void{0.0}, nu_void{0.0};
  double volume{0.0};
  double lambda_volume{0.0};
  double lambda_reg{0.0};
  double lambda_reg_end{0.0};
  double simp{0.0};
  double simp_end{0.0};
  double epsilon{0.0};
  double dt{0.0};
  double max_delta{0.0};
  double tol_design{0.0};
  double tol_objective{0.0};
  double tol_tensor{0.0};
  double J_prev{0.0};
  double C_prev[36]{};
  double C_target[36]{};
  double W[36]{};
};

template <typename Tensor> inline void store_voigt6(double *out, const Tensor &C) {
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) out[6 * i + j] = C(i, j);
}

template <typename Tensor> inline void fill_voigt6(Tensor &C, const double *in) {
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) C(i, j) = in[6 * i + j];
}

inline bool same_real(double a, double b) {
  return std::abs(a - b) <= 1.0e-12 * (1.0 + std::max(std::abs(a), std::abs(b)));
}

inline bool same_voigt(const double *a, const double *b) {
  for (int i = 0; i < 36; ++i)
    if (!same_real(a[i], b[i])) return false;
  return true;
}

inline void apply_tracker(const InverseCheckpoint &ck, ConvergenceTracker &tr) {
  tr.cfg.continuation_steps = ck.continuation_steps;
  tr.cfg.max_steps = ck.max_steps;
  tr.cfg.conv_window = ck.conv_window;
  tr.cfg.verify_steps = ck.verify_steps;
  tr.cfg.tol_design = ck.tol_design;
  tr.cfg.tol_objective = ck.tol_objective;
  tr.cfg.tol_tensor = ck.tol_tensor;
  tr.quiet_count = ck.quiet_count;
  tr.verify_left = ck.verify_left;
  tr.candidate = ck.candidate != 0;
  tr.verified = ck.verified != 0;
}

inline void capture_tracker(InverseCheckpoint &ck, const ConvergenceTracker &tr,
                            int next_step) {
  ck.next_step = next_step;
  ck.continuation_steps = tr.cfg.continuation_steps;
  ck.max_steps = tr.cfg.max_steps;
  ck.conv_window = tr.cfg.conv_window;
  ck.verify_steps = tr.cfg.verify_steps;
  ck.quiet_count = tr.quiet_count;
  ck.verify_left = tr.verify_left;
  ck.candidate = tr.candidate ? 1 : 0;
  ck.verified = tr.verified ? 1 : 0;
  ck.tol_design = tr.cfg.tol_design;
  ck.tol_objective = tr.cfg.tol_objective;
  ck.tol_tensor = tr.cfg.tol_tensor;
}

[[nodiscard]] inline bool
checkpoint_is_restartable(const InverseCheckpoint &ck,
                          int requested_max_steps) noexcept {
  // h_next is not the last evaluated state. A restart must evaluate it before
  // finalization can associate an accepted index with the exported field.
  return ck.termination == static_cast<int>(TerminationReason::Running) &&
         requested_max_steps > ck.next_step;
}

template <typename Cfg, typename Tensor>
inline void capture_problem(InverseCheckpoint &ck, const Cfg &cfg,
                            const Tensor &C_target, const Tensor &W) {
  ck.nx = cfg.nx;
  ck.ny = cfg.ny;
  ck.nz = cfg.nz;
  ck.continuation_steps = cfg.continuation_steps;
  ck.dx = cfg.dx;
  ck.E_solid = cfg.E_solid;
  ck.nu_solid = cfg.nu_solid;
  ck.E_void = cfg.E_void;
  ck.nu_void = cfg.nu_void;
  ck.volume = cfg.volume;
  ck.lambda_volume = cfg.lambda_volume;
  ck.lambda_reg = cfg.lambda_reg;
  ck.lambda_reg_end =
      (cfg.lambda_reg_end >= 0.0) ? cfg.lambda_reg_end : cfg.lambda_reg;
  ck.simp = cfg.simp;
  ck.simp_end = (cfg.simp_end > 0.0) ? cfg.simp_end : cfg.simp;
  ck.epsilon = cfg.epsilon;
  ck.dt = cfg.dt;
  ck.max_delta = cfg.max_delta;
  ck.normalize = cfg.normalize;
  ck.project_volume = cfg.project_volume;
  ck.n_el_iter = cfg.n_el_iter;
  ck.conv_window = cfg.conv_window;
  ck.verify_steps = cfg.verify_steps;
  ck.tol_design = cfg.tol_design;
  ck.tol_objective = cfg.tol_objective;
  ck.tol_tensor = cfg.tol_tensor;
  store_voigt6(ck.C_target, C_target);
  store_voigt6(ck.W, W);
}

[[nodiscard]] inline bool checkpoint_matches_problem(const InverseCheckpoint &ck,
                                                     const InverseCheckpoint &want) {
  return ck.nx == want.nx && ck.ny == want.ny && ck.nz == want.nz &&
         ck.continuation_steps == want.continuation_steps &&
         ck.normalize == want.normalize &&
         ck.project_volume == want.project_volume &&
         ck.n_el_iter == want.n_el_iter && ck.conv_window == want.conv_window &&
         ck.verify_steps == want.verify_steps && same_real(ck.dx, want.dx) &&
         same_real(ck.E_solid, want.E_solid) &&
         same_real(ck.nu_solid, want.nu_solid) &&
         same_real(ck.E_void, want.E_void) && same_real(ck.nu_void, want.nu_void) &&
         same_real(ck.volume, want.volume) &&
         same_real(ck.lambda_volume, want.lambda_volume) &&
         same_real(ck.lambda_reg, want.lambda_reg) &&
         same_real(ck.lambda_reg_end, want.lambda_reg_end) &&
         same_real(ck.simp, want.simp) && same_real(ck.simp_end, want.simp_end) &&
         same_real(ck.epsilon, want.epsilon) && same_real(ck.dt, want.dt) &&
         same_real(ck.max_delta, want.max_delta) &&
         same_real(ck.tol_design, want.tol_design) &&
         same_real(ck.tol_objective, want.tol_objective) &&
         same_real(ck.tol_tensor, want.tol_tensor) &&
         same_voigt(ck.C_target, want.C_target) && same_voigt(ck.W, want.W);
}

template <typename Cfg, typename Tensor>
[[nodiscard]] inline bool
checkpoint_matches_problem(const InverseCheckpoint &ck, const Cfg &cfg,
                           const Tensor &C_target, const Tensor &W) {
  InverseCheckpoint want{};
  capture_problem(want, cfg, C_target, W);
  return checkpoint_matches_problem(ck, want);
}

inline void write_voigt_line(std::ostream &os, std::string_view key,
                             const double *v) {
  os << key;
  for (int i = 0; i < 36; ++i) os << ' ' << std::setprecision(17) << v[i];
  os << '\n';
}

inline bool write_checkpoint_text(std::ostream &os, const InverseCheckpoint &ck) {
  os << kInverseCheckpointMagic << ' ' << ck.schema << '\n';
  os << "nx " << ck.nx << "\nny " << ck.ny << "\nnz " << ck.nz << '\n';
  os << "next_step " << ck.next_step << '\n';
  os << "continuation_steps " << ck.continuation_steps << '\n';
  os << "max_steps " << ck.max_steps << '\n';
  os << "conv_window " << ck.conv_window << '\n';
  os << "verify_steps " << ck.verify_steps << '\n';
  os << "quiet_count " << ck.quiet_count << '\n';
  os << "verify_left " << ck.verify_left << '\n';
  os << "candidate " << ck.candidate << '\n';
  os << "verified " << ck.verified << '\n';
  os << "have_prev " << ck.have_prev << '\n';
  os << "n_snap " << ck.n_snap << '\n';
  os << "last_dumped " << ck.last_dumped << '\n';
  os << "termination " << ck.termination << '\n';
  os << "normalize " << ck.normalize << '\n';
  os << "project_volume " << ck.project_volume << '\n';
  os << "n_el_iter " << ck.n_el_iter << '\n';
  os << std::setprecision(17);
  os << "dx " << ck.dx << '\n';
  os << "E_solid " << ck.E_solid << "\nnu_solid " << ck.nu_solid << '\n';
  os << "E_void " << ck.E_void << "\nnu_void " << ck.nu_void << '\n';
  os << "volume " << ck.volume << '\n';
  os << "lambda_volume " << ck.lambda_volume << '\n';
  os << "lambda_reg " << ck.lambda_reg << '\n';
  os << "lambda_reg_end " << ck.lambda_reg_end << '\n';
  os << "simp " << ck.simp << "\nsimp_end " << ck.simp_end << '\n';
  os << "epsilon " << ck.epsilon << "\ndt " << ck.dt << '\n';
  os << "max_delta " << ck.max_delta << '\n';
  os << "tol_design " << ck.tol_design << '\n';
  os << "tol_objective " << ck.tol_objective << '\n';
  os << "tol_tensor " << ck.tol_tensor << '\n';
  os << "J_prev " << ck.J_prev << '\n';
  write_voigt_line(os, "C_prev", ck.C_prev);
  write_voigt_line(os, "C_target", ck.C_target);
  write_voigt_line(os, "W", ck.W);
  return static_cast<bool>(os);
}

inline bool read_voigt36(std::istream &in, double *out) {
  for (int i = 0; i < 36; ++i)
    if (!(in >> out[i])) return false;
  return true;
}

/// Necessary invariants of an accepted-state tracker snapshot. Validate before
/// applying counters: a negative remaining hold must never become convergence.
[[nodiscard]] inline bool checkpoint_state_valid(const InverseCheckpoint &ck) {
  const auto boolean = [](int v) { return v == 0 || v == 1; };
  if (!boolean(ck.candidate) || !boolean(ck.verified) || !boolean(ck.have_prev) ||
      !boolean(ck.normalize) || !boolean(ck.project_volume) || ck.nx <= 0 ||
      ck.ny <= 0 || ck.nz <= 0 || ck.max_steps <= 0 || ck.next_step < 0 ||
      ck.next_step > ck.max_steps || ck.continuation_steps < 0 ||
      ck.conv_window <= 0 || ck.verify_steps < 0 || ck.quiet_count < 0 ||
      ck.verify_left < 0 || ck.n_snap < 0 || ck.last_dumped < -1 ||
      ck.termination < 0 || ck.termination > 3)
    return false;
  for (double v : {ck.dx, ck.E_solid, ck.nu_solid, ck.E_void, ck.nu_void, ck.volume,
                   ck.lambda_volume, ck.lambda_reg, ck.lambda_reg_end, ck.simp,
                   ck.simp_end, ck.epsilon, ck.dt, ck.max_delta, ck.tol_design,
                   ck.tol_objective, ck.tol_tensor, ck.J_prev})
    if (!std::isfinite(v)) return false;
  for (int i = 0; i < 36; ++i)
    if (!std::isfinite(ck.C_prev[i]) || !std::isfinite(ck.C_target[i]) ||
        !std::isfinite(ck.W[i]))
      return false;
  if (ck.dx <= 0 || ck.tol_design < 0 || ck.tol_objective < 0 || ck.tol_tensor < 0)
    return false;
  if (ck.have_prev != (ck.next_step > 0 ? 1 : 0)) return false;
  // Step zero has no predecessor; only frozen accepted transitions are quiet.
  const auto available = std::max<std::int64_t>(
      0, std::int64_t(ck.next_step) - std::max(1, ck.continuation_steps));
  if (ck.quiet_count > available) return false;
  if (!ck.candidate) {
    if (ck.verified || ck.verify_left != 0 || ck.quiet_count >= ck.conv_window)
      return false;
  } else {
    const auto elapsed = std::int64_t(ck.quiet_count) - ck.conv_window;
    if (elapsed < 0 || elapsed > ck.verify_steps ||
        ck.verify_left != std::int64_t(ck.verify_steps) - elapsed ||
        ck.verified != (ck.verify_left == 0 ? 1 : 0))
      return false;
  }
  const auto reason = static_cast<TerminationReason>(ck.termination);
  if ((reason == TerminationReason::Converged) != (ck.verified != 0)) return false;
  if (reason == TerminationReason::Running && ck.next_step >= ck.max_steps)
    return false;
  if (reason == TerminationReason::MaxSteps && ck.next_step != ck.max_steps)
    return false;
  if (reason != TerminationReason::Running && ck.next_step == 0) return false;
  return true;
}

[[nodiscard]] inline bool read_checkpoint_text(std::istream &in,
                                               InverseCheckpoint &ck) {
  std::string magic;
  int schema = 0;
  if (!(in >> magic >> schema) || magic != kInverseCheckpointMagic ||
      schema != kInverseCheckpointSchema)
    return false;
  ck = InverseCheckpoint{};
  ck.schema = schema;
  std::string key;
  std::uint64_t seen = 0;
  auto mark = [&](unsigned bit) {
    const std::uint64_t m = 1ull << bit;
    if (seen & m) return false;
    seen |= m;
    return true;
  };
  while (in >> key) {
    if (key == "nx") {
      if (!mark(0) || !(in >> ck.nx)) return false;
    } else if (key == "ny") {
      if (!mark(1) || !(in >> ck.ny)) return false;
    } else if (key == "nz") {
      if (!mark(2) || !(in >> ck.nz)) return false;
    } else if (key == "next_step") {
      if (!mark(3) || !(in >> ck.next_step) || ck.next_step < 0) return false;
    } else if (key == "continuation_steps") {
      if (!mark(4) || !(in >> ck.continuation_steps)) return false;
    } else if (key == "max_steps") {
      if (!mark(5) || !(in >> ck.max_steps)) return false;
    } else if (key == "conv_window") {
      if (!mark(6) || !(in >> ck.conv_window)) return false;
    } else if (key == "verify_steps") {
      if (!mark(7) || !(in >> ck.verify_steps)) return false;
    } else if (key == "quiet_count") {
      if (!mark(8) || !(in >> ck.quiet_count)) return false;
    } else if (key == "verify_left") {
      if (!mark(9) || !(in >> ck.verify_left)) return false;
    } else if (key == "candidate") {
      if (!mark(10) || !(in >> ck.candidate)) return false;
    } else if (key == "verified") {
      if (!mark(11) || !(in >> ck.verified)) return false;
    } else if (key == "have_prev") {
      if (!mark(12) || !(in >> ck.have_prev)) return false;
    } else if (key == "n_snap") {
      if (!mark(13) || !(in >> ck.n_snap) || ck.n_snap < 0) return false;
    } else if (key == "last_dumped") {
      if (!mark(14) || !(in >> ck.last_dumped)) return false;
    } else if (key == "termination") {
      if (!mark(39) || !(in >> ck.termination) || ck.termination < 0 ||
          ck.termination > 3)
        return false;
    } else if (key == "normalize") {
      if (!mark(15) || !(in >> ck.normalize)) return false;
    } else if (key == "project_volume") {
      if (!mark(16) || !(in >> ck.project_volume)) return false;
    } else if (key == "n_el_iter") {
      if (!mark(17) || !(in >> ck.n_el_iter)) return false;
    } else if (key == "dx") {
      if (!mark(18) || !(in >> ck.dx)) return false;
    } else if (key == "E_solid") {
      if (!mark(19) || !(in >> ck.E_solid)) return false;
    } else if (key == "nu_solid") {
      if (!mark(20) || !(in >> ck.nu_solid)) return false;
    } else if (key == "E_void") {
      if (!mark(21) || !(in >> ck.E_void)) return false;
    } else if (key == "nu_void") {
      if (!mark(22) || !(in >> ck.nu_void)) return false;
    } else if (key == "volume") {
      if (!mark(23) || !(in >> ck.volume)) return false;
    } else if (key == "lambda_volume") {
      if (!mark(24) || !(in >> ck.lambda_volume)) return false;
    } else if (key == "lambda_reg") {
      if (!mark(25) || !(in >> ck.lambda_reg)) return false;
    } else if (key == "lambda_reg_end") {
      if (!mark(26) || !(in >> ck.lambda_reg_end)) return false;
    } else if (key == "simp") {
      if (!mark(27) || !(in >> ck.simp)) return false;
    } else if (key == "simp_end") {
      if (!mark(28) || !(in >> ck.simp_end)) return false;
    } else if (key == "epsilon") {
      if (!mark(29) || !(in >> ck.epsilon)) return false;
    } else if (key == "dt") {
      if (!mark(30) || !(in >> ck.dt)) return false;
    } else if (key == "max_delta") {
      if (!mark(31) || !(in >> ck.max_delta)) return false;
    } else if (key == "tol_design") {
      if (!mark(32) || !(in >> ck.tol_design)) return false;
    } else if (key == "tol_objective") {
      if (!mark(33) || !(in >> ck.tol_objective)) return false;
    } else if (key == "tol_tensor") {
      if (!mark(34) || !(in >> ck.tol_tensor)) return false;
    } else if (key == "J_prev") {
      if (!mark(35) || !(in >> ck.J_prev)) return false;
    } else if (key == "C_prev") {
      if (!mark(36) || !read_voigt36(in, ck.C_prev)) return false;
    } else if (key == "C_target") {
      if (!mark(37) || !read_voigt36(in, ck.C_target)) return false;
    } else if (key == "W") {
      if (!mark(38) || !read_voigt36(in, ck.W)) return false;
    } else {
      return false;
    }
  }
  constexpr std::uint64_t kRequired = (1ull << 40) - 1ull;
  return seen == kRequired && in.eof() && !in.bad() && checkpoint_state_valid(ck);
}

[[nodiscard]] inline std::string
format_checkpoint_text(const InverseCheckpoint &ck) {
  std::ostringstream os;
  write_checkpoint_text(os, ck);
  return os.str();
}

[[nodiscard]] inline std::string checkpoint_generation_name(int next_step) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "gen_%07d", std::max(0, next_step));
  return buf;
}

[[nodiscard]] inline std::filesystem::path
checkpoint_staging_dir(const std::filesystem::path &root) {
  return root / std::string(kInverseCheckpointStaging);
}

[[nodiscard]] inline bool
checkpoint_bundle_complete(const std::filesystem::path &dir) {
  return std::filesystem::is_regular_file(dir / "state.txt") &&
         std::filesystem::is_regular_file(dir / "h.bin") &&
         std::filesystem::is_regular_file(dir / "h_prev.bin");
}

[[nodiscard]] inline bool checkpoint_generation_name_ok(std::string_view gen) {
  if (gen.size() < 5 || gen.rfind("gen_", 0) != 0) return false;
  if (gen.find('/') != std::string_view::npos ||
      gen.find('\\') != std::string_view::npos ||
      gen.find("..") != std::string_view::npos)
    return false;
  return std::all_of(gen.begin() + 4, gen.end(),
                     [](unsigned char c) { return c >= '0' && c <= '9'; });
}

inline bool write_current_pointer(const std::filesystem::path &root,
                                  std::string_view gen) {
  if (!checkpoint_generation_name_ok(gen)) return false;
  std::error_code ec;
  std::filesystem::create_directories(root, ec);
  if (ec) return false;
  const auto tmp = root / "CURRENT.tmp";
  const auto dst = root / std::string(kInverseCheckpointCurrent);
  {
    std::ofstream out(tmp);
    if (!out) return false;
    out << gen << '\n';
    out.close();
    if (!out) return false;
  }
  std::filesystem::rename(tmp, dst, ec);
  return !ec;
}

[[nodiscard]] inline std::optional<std::string>
read_current_pointer(const std::filesystem::path &root) {
  std::ifstream in(root / std::string(kInverseCheckpointCurrent));
  if (!in) return std::nullopt;
  std::string gen;
  if (!(in >> gen) || !checkpoint_generation_name_ok(gen)) return std::nullopt;
  return gen;
}

[[nodiscard]] inline std::filesystem::path
resolve_checkpoint_bundle(const std::filesystem::path &root) {
  if (const auto gen = read_current_pointer(root)) {
    const auto dir = root / *gen;
    if (checkpoint_bundle_complete(dir)) return dir;
    return {};
  }
  if (checkpoint_bundle_complete(root)) return root;
  return {};
}

inline bool prepare_checkpoint_staging(const std::filesystem::path &root) {
  std::error_code ec;
  std::filesystem::create_directories(root, ec);
  if (ec) return false;
  const auto staging = checkpoint_staging_dir(root);
  std::filesystem::remove_all(staging, ec);
  std::filesystem::create_directories(staging, ec);
  return !ec && std::filesystem::is_directory(staging);
}

inline bool write_dump_steps(const std::filesystem::path &path,
                             const std::vector<int> &steps) {
  std::ofstream out(path);
  if (!out) return false;
  for (int s : steps) out << s << '\n';
  out.close();
  return static_cast<bool>(out);
}

inline bool checkpoint_field_sizes_match(const std::filesystem::path &dir,
                                         const InverseCheckpoint &ck) {
  std::uintmax_t bytes = sizeof(double);
  for (int dimension : {ck.nx, ck.ny, ck.nz}) {
    if (dimension <= 0 || bytes > std::numeric_limits<std::uintmax_t>::max() /
                                      static_cast<std::uintmax_t>(dimension))
      return false;
    bytes *= static_cast<std::uintmax_t>(dimension);
  }
  for (const char *name : {"h.bin", "h_prev.bin"}) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(dir / name, ec);
    if (ec || size != bytes) return false;
  }
  return true;
}

inline bool write_checkpoint_file(const std::filesystem::path &path,
                                  const InverseCheckpoint &ck) {
  std::ofstream out(path);
  if (!out) return false;
  write_checkpoint_text(out, ck);
  out.close();
  return static_cast<bool>(out);
}

inline bool read_dump_steps(const std::filesystem::path &path,
                            std::vector<int> &steps, int expected_count,
                            int last_dumped) {
  steps.clear();
  if (expected_count < 0 || (expected_count == 0 && last_dumped != -1)) return false;
  std::ifstream in(path);
  if (!in) return false;
  std::vector<int> parsed;
  for (int i = 0; i < expected_count; ++i) {
    int step = 0;
    if (!(in >> step) || step < 0 || (!parsed.empty() && step <= parsed.back()))
      return false;
    parsed.push_back(step);
  }
  in >> std::ws;
  if (!in.eof() || in.bad() || (!parsed.empty() && parsed.back() != last_dumped))
    return false;
  steps = std::move(parsed);
  return true;
}

inline bool publish_checkpoint_generation(const std::filesystem::path &root,
                                          std::string_view gen) {
  if (!checkpoint_generation_name_ok(gen)) return false;
  const auto staging = checkpoint_staging_dir(root);
  const auto dest = root / std::string(gen);
  if (!checkpoint_bundle_complete(staging)) return false;
  std::error_code ec;
  if (std::filesystem::exists(dest, ec)) {
    const auto cur = read_current_pointer(root);
    if (cur && *cur == gen) return true;
    std::filesystem::remove_all(dest, ec);
    if (ec) return false;
  }
  const auto previous = read_current_pointer(root);
  std::filesystem::rename(staging, dest, ec);
  if (ec) return false;
  if (!write_current_pointer(root, gen)) return false;
  for (const auto &entry : std::filesystem::directory_iterator(root, ec)) {
    if (ec || !entry.is_directory()) continue;
    const auto name = entry.path().filename().string();
    if (!checkpoint_generation_name_ok(name)) continue;
    if (name == gen) continue;
    if (previous && name == *previous) continue;
    std::filesystem::remove_all(entry.path(), ec);
  }
  return true;
}

} // namespace pfc::apps::inverse
