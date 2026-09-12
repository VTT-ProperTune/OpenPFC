// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_dealias_study.cpp
 * @brief What the 2/3 dealias mask costs or saves, as a function of resolution.
 *
 * @details
 * The shipped tungsten presets run the cubic PFC nonlinearity with dealiasing
 * off. This measures whether that matters, by running the *same* seeded
 * solidification twice at each resolution — mask off, then mask on — from an
 * identical initial condition for an identical number of steps, and comparing
 * what comes out.
 *
 * The observables are the ones a PFC run is read for: the selected wavenumber
 * (`k1`, the first moment of the shell-averaged structure factor, i.e. the
 * lattice constant), the structure-factor peak, the total spectral power, and
 * the density extremes.
 *
 * See `tungsten/resolution.hpp` for why the interesting scale is
 * \f$\Delta x=\pi/3\f$: below it the third harmonic aliases with the mask off
 * *and* the mask amputates the second harmonic with it on, so the grid is
 * wrong either way and the comparison has no good side.
 *
 * Usage: `tungsten_dealias_study [output.csv] [steps]`
 *        `tungsten_dealias_study --reserved [output.csv]`
 *
 * `--reserved` is the Paper A RQ2 evaluation point: N=128, dx=pi/3
 * (six points per lattice, not in the four-row exploration CSV), 30
 * steps with 5 warm-up, median wall_step mask on vs off.
 *
 * Single rank by default and not fast — the finest point is a 96³ run twice
 * over. It is a study, not a test; nothing in CI calls it. Regenerate
 * `docs/report/data/tungsten_dealias_resolution.csv` with it when the model or
 * the presets change.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/structure_factor.hpp>
#include <tungsten/resolution.hpp>
#include <tungsten/tungsten_physics.hpp>

namespace {

using json = nlohmann::json;
using Physics = tungsten::TungstenPhysics<double, pfc::HostSpace>;

/// The parameters every shipped tungsten preset uses.
json shipped_params() {
  return json{{"n0", -0.10},        {"alpha", 0.50},
              {"n_sol", -0.047},    {"n_vap", -0.464},
              {"T", 3300.0},        {"T0", 156000.0},
              {"Bx", 0.8582},       {"alpha_farTol", 0.001},
              {"alpha_highOrd", 4}, {"lambda", 0.22},
              {"stabP", 0.2},       {"shift_u", 0.3341},
              {"shift_s", 0.1898},  {"p2", 1.0},
              {"p3", -0.5},         {"p4", 0.333333333},
              {"q20", -0.0037},     {"q21", 1.0},
              {"q30", -12.4567},    {"q31", 20.0},
              {"q40", 45.0}};
}

struct Outcome {
  double k_peak{}, k1{}, s_peak{}, power{}, min_psi{}, max_psi{}, mean_psi{};
  double wall_step_s_median{}, checksum_l2{};
};

int env_warmup(int fallback = 5) {
  const char *v = std::getenv("TUNGSTEN_WARMUP");
  if (v == nullptr || *v == '\0') return fallback;
  return std::atoi(v);
}

double median_of(std::vector<double> s) {
  if (s.empty()) return 0.0;
  std::sort(s.begin(), s.end());
  const std::size_t n = s.size();
  return (n % 2 == 1) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/// One seeded-solidification run. `dealias` is the only thing that varies.
Outcome run_case(int n, double dx, int n_steps, bool dealias, int rank, int nproc,
                 int warmup = 0) {
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, dx, dx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
  auto phys = Physics::from_json(shipped_params(), domain,
                                 stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &psi = state.get_field<double>("psi");

  // The shipped solidification initial condition: a crystal seed in an
  // undercooled vapour. A small perturbation about n0 simply decays -- the
  // cubic term needs an amplitude to act on before any of this is visible.
  const pfc::Box3i box = stack.fft().get_inbox_bounds();
  for (const auto &j : {json{{"type", "constant"}, {"n0", -0.4}},
                        json{{"type", "single_seed"},
                             {"amp_eq", 0.215936},
                             {"rho_seed", -0.047}}}) {
    auto mod = pfc::ui::create_field_modifier(j["type"].get<std::string>(), j);
    mod->apply(psi.vec(), domain, box, 0.0);
  }

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "psi";
  opt.dealias = dealias;
  pfc::sim::SpectralETDSystem<Physics> sys(phys, stack.fft(), state, 1.0, opt);
  double t = 0.0;
  std::vector<double> step_s;
  step_s.reserve(static_cast<std::size_t>(n_steps));
  for (int s = 0; s < n_steps; ++s) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    t = sys.step(t);
    MPI_Barrier(MPI_COMM_WORLD);
    if (s >= warmup) step_s.push_back(MPI_Wtime() - t0);
  }

  pfc::data::Field<std::complex<double>> hat(domain, stack.fft().get_outbox_bounds(),
                                             0);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), psi, hat);
  pfc::apps::StructureFactor sf{};
  hat.with_host_view([&](std::complex<double> *h, std::size_t) {
    sf = pfc::apps::shell_average(stack.fft().get_outbox_bounds(), domain, h,
                                  MPI_COMM_WORLD, 96);
  });

  double lo = 1e300, hi = -1e300, sum = 0.0, cnt = 0.0;
  psi.with_host_view([&](const double *p, std::size_t m) {
    for (std::size_t i = 0; i < m; ++i) {
      lo = std::min(lo, p[i]);
      hi = std::max(hi, p[i]);
      sum += p[i];
      cnt += 1.0;
    }
  });
  double g[2]{}, l[2]{sum, cnt}, glo = 0.0, ghi = 0.0;
  MPI_Allreduce(l, g, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lo, &glo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&hi, &ghi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double l2_local = 0.0, n_local = 0.0;
  psi.with_host_view([&](const double *p, std::size_t m) {
    for (std::size_t i = 0; i < m; ++i) {
      l2_local += p[i] * p[i];
      n_local += 1.0;
    }
  });
  double l2g[2]{}, l2l[2]{l2_local, n_local};
  MPI_Allreduce(l2l, l2g, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double checksum_l2 = std::sqrt(l2g[0] / l2g[1]);

  return Outcome{sf.k_peak, sf.k1,          sf.S_peak, sf.total_power,
                 glo,       ghi,            g[0] / g[1],
                 median_of(step_s), checksum_l2};
}

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const bool reserved =
      (argc > 1 && std::string(argv[1]) == "--reserved");
  const std::string out_path = reserved
      ? ((argc > 2) ? argv[2] : "tungsten_dealias_reserved.csv")
      : ((argc > 1) ? argv[1] : "tungsten_dealias_resolution.csv");
  const int n_steps = reserved ? 30
      : ((argc > 2) ? std::atoi(argv[2]) : 1000);
  const int warmup = reserved ? env_warmup(5) : 0;

  if (reserved) {
    const int n = 128;
    const double dx = std::numbers::pi / 3.0;
    const Outcome off = run_case(n, dx, n_steps, false, rank, nproc, warmup);
    const Outcome on = run_case(n, dx, n_steps, true, rank, nproc, warmup);
    auto rel = [](double a, double b) {
      return (a != 0.0) ? std::abs(b - a) / std::abs(a) : 0.0;
    };
    const double f_before = 3.0 / tungsten::resolution::nyquist_k(dx);
    const double f_after = tungsten::resolution::two_thirds_cut(dx) /
                           tungsten::resolution::nyquist_k(dx);
    const double r = (off.wall_step_s_median > 0.0)
                         ? on.wall_step_s_median / off.wall_step_s_median
                         : 0.0;
    if (rank == 0) {
      const std::filesystem::path p{out_path};
      if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
      std::unique_ptr<std::FILE, int (*)(std::FILE *)> out(std::fopen(out_path.c_str(), "w"),
                                                           std::fclose);
      if (!out) {
        std::fprintf(stderr, "cannot open %s\n", out_path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::fprintf(out.get(),
                   "# Reserved N=128 dx=pi/3 tungsten mask on/off, Paper A RQ2.\n"
                   "N,dx,steps,warmup,pts_per_lattice,f_nl_before,f_nl_after,"
                   "wall_step_ms_off,wall_step_ms_on,r,"
                   "k1_off,k1_on,rel_dk1,s_peak_off,s_peak_on,rel_ds_peak,"
                   "checksum_l2_off,checksum_l2_on\n");
      std::ostringstream line;
      line.imbue(std::locale::classic());
      line << std::setprecision(10) << n << ',' << dx << ',' << n_steps << ','
           << warmup << ',' << tungsten::resolution::points_per_lattice(dx) << ','
           << f_before << ',' << f_after << ','
           << (1000.0 * off.wall_step_s_median) << ','
           << (1000.0 * on.wall_step_s_median) << ',' << r << ',' << off.k1
           << ',' << on.k1 << ',' << rel(off.k1, on.k1) << ',' << off.s_peak
           << ',' << on.s_peak << ',' << rel(off.s_peak, on.s_peak) << ','
           << off.checksum_l2 << ',' << on.checksum_l2 << '\n';
      std::fputs(line.str().c_str(), out.get());
      std::printf("TUNGSTEN_WALL_STEP_MS_OFF=%.6f\n", 1000.0 * off.wall_step_s_median);
      std::printf("TUNGSTEN_WALL_STEP_MS_ON=%.6f\n", 1000.0 * on.wall_step_s_median);
      std::printf("TUNGSTEN_R=%.6f\n", r);
      std::printf("TUNGSTEN_F_NL_BEFORE=%.6f TUNGSTEN_F_NL_AFTER=%.6f\n", f_before,
                  f_after);
      std::printf("rel_dk1=%.6e rel_ds_peak=%.6e\n", rel(off.k1, on.k1),
                  rel(off.s_peak, on.s_peak));
      std::printf("wrote %s\n", out_path.c_str());
    }
    MPI_Finalize();
    return 0;
  }

  // Roughly a fixed physical box (~71 code lengths) at each spacing, so the
  // seed sees the same amount of vapour to grow into.
  struct Point {
    int n;
    double dx;
  };
  const std::vector<Point> points{{64, 1.1107207345395915}, // the shipped preset
                                  {64, std::numbers::pi / 3.0},
                                  {72, 2.0 * std::numbers::pi / 6.6666666667},
                                  {96, std::numbers::pi / 4.0}};

  std::unique_ptr<std::FILE, int (*)(std::FILE *)> out(nullptr, std::fclose);
  if (rank == 0) {
    const std::filesystem::path p{out_path};
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    out.reset(std::fopen(out_path.c_str(), "w"));
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", out_path.c_str());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::fprintf(out.get(),
                 "# Effect of 2/3 dealiasing on tungsten PFC, by resolution, "
                 "from tungsten_dealias_study.\n");
    std::fprintf(out.get(),
                 "N,dx,points_per_lattice,k_nyquist,two_thirds_cut,steps,"
                 "third_harmonic_resolved,mask_spares_second,"
                 "k1_off,k1_on,rel_dk1,"
                 "s_peak_off,s_peak_on,rel_ds_peak,"
                 "power_off,power_on,rel_dpower,"
                 "max_off,max_on,rel_dmax\n");
  }

  for (const auto &pt : points) {
    const Outcome off = run_case(pt.n, pt.dx, n_steps, false, rank, nproc);
    const Outcome on = run_case(pt.n, pt.dx, n_steps, true, rank, nproc);
    auto rel = [](double a, double b) {
      return (a != 0.0) ? std::abs(b - a) / std::abs(a) : 0.0;
    };
    if (rank == 0) {
      std::ostringstream line;
      line.imbue(std::locale::classic());
      line << std::setprecision(10) << pt.n << ',' << pt.dx << ','
           << tungsten::resolution::points_per_lattice(pt.dx) << ','
           << tungsten::resolution::nyquist_k(pt.dx) << ','
           << tungsten::resolution::two_thirds_cut(pt.dx) << ',' << n_steps << ','
           << (tungsten::resolution::third_harmonic_resolved(pt.dx) ? 1 : 0) << ','
           << (tungsten::resolution::mask_spares_second_harmonic(pt.dx) ? 1 : 0)
           << ',' << off.k1 << ',' << on.k1 << ',' << rel(off.k1, on.k1) << ','
           << off.s_peak << ',' << on.s_peak << ',' << rel(off.s_peak, on.s_peak)
           << ',' << off.power << ',' << on.power << ','
           << rel(off.power, on.power) << ',' << off.max_psi << ',' << on.max_psi
           << ',' << rel(off.max_psi, on.max_psi) << '\n';
      std::fputs(line.str().c_str(), out.get());
      std::fflush(out.get());
      std::printf("N=%3d dx=%.6f  pts/lattice=%.2f  |dpower|=%.4f%%  "
                  "|dmax|=%.4f%%  |dk1|=%.2e\n",
                  pt.n, pt.dx, tungsten::resolution::points_per_lattice(pt.dx),
                  100.0 * rel(off.power, on.power),
                  100.0 * rel(off.max_psi, on.max_psi), rel(off.k1, on.k1));
    }
  }

  if (rank == 0) std::printf("wrote %s\n", out_path.c_str());
  MPI_Finalize();
  return 0;
}
