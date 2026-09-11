// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file vlasov_hip_cost.cpp
 * @brief Where the time actually goes in a 1D2V Vlasov step, on the host and
 *        on the device, at several phase-space sizes -- the measurement that
 *        decides what is worth porting.
 *
 * @details
 * ## Why this binary exists before the port and not after it
 *
 * The precedent is `apps/alloy_dendrite_elastic`, where a measurement showed
 * the host round trip was 0.3% of the step while the host solve was three
 * orders of magnitude more expensive than the device step, and that single
 * ratio decided the whole design. The equivalent question here is not
 * "is a gather faster on a GPU" -- it is -- but:
 *
 *  1. **Which phase dominates the host step?** If the spectral `x` shift
 *     (phase A) is most of it, then porting the gathers and the reduction
 *     removes a minority of the time and Amdahl caps the whole exercise.
 *  2. **What does leaving phase A on the host cost?** The distribution is
 *     the largest object in the application. Leaving one phase behind means
 *     four whole-brick copies across the bus per step, and if those cost
 *     more than the three kernels save, the port is not merely unprofitable
 *     but actively harmful, and the right answer is to say so.
 *
 * Both are measurements, both are made here, and
 * `docs/hpc/vlasov_gpu.md` records the answers with the job ids.
 *
 * ## What is timed, and how
 *
 * Every phase is timed with the device drained at both ends
 * (`hipDeviceSynchronize`), on a compute node, after a warm-up iteration
 * that pays for the FFTW plan, the first kernel load and the first
 * allocation. Repetitions are reported as a mean over `--reps` calls, and
 * the per-call spread is reported too, because a single number from a shared
 * node is not a measurement.
 *
 * The host transport is timed with @ref vlasov::TransportWorkspace::measure_mass
 * **off**. That flag adds two extra streaming passes over the brick per call
 * to compute a conservation diagnostic, and charging the operator for a
 * diagnostic the device path does not compute would flatter the device. The
 * header that defines it says as much: "switch it off in a profiling run".
 *
 * ## What is not timed
 *
 * Nothing here writes a CSV ledger, an initial condition or a snapshot.
 * Those are once-per-sample costs in a production run and once-per-run costs
 * in this one; including them would measure the I/O layer.
 *
 * @see device_step_hip.hpp for what was ported and why
 * @see vlasov_hip_parity.cpp for whether it computes the same thing
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "vlasov_hip_cost requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <vlasov_maxwell/cli.hpp>
#include <vlasov_maxwell/device_step_hip.hpp>
#include <vlasov_maxwell/ics.hpp>
#include <vlasov_maxwell/step.hpp>

namespace {

using vlasov::PhaseSpace;
using vlasov::SimParams;
using vlasov::Species;
using vlasov::Stepper;
using vlasov::hip::DeviceStepper;
using vlasov::hip::wall_seconds;

/**
 * @brief `vlasov::view_of`'s data, visited in memory order.
 *
 * Not a change to the application -- it is a *control*, and it exists
 * because without it the host reduction's cost would be misattributed.
 *
 * `moments.hpp` reduces through @ref vlasov::StridedDistribution, whose
 * `for_each_owned` runs `x` outermost and `v_y` innermost. That is the right
 * order for the layout its own `contiguous()` constructor describes
 * (`v_y` fastest), and the wrong order for the layout `step.hpp` actually
 * hands it: the padded phase-space brick is **x-fastest**, so consecutive
 * `v_y` at fixed `(x, v_x)` are `npx npy` doubles -- hundreds of kilobytes --
 * apart, and every single load is a cache miss and a TLB miss.
 *
 * This view answers `operator()` identically and only visits the same cells
 * in a different order, so `reduce_velocity` computes a different rounding
 * of the same sums at a completely different speed. Timing both says how
 * much of the host reduction's cost is the arithmetic and how much is the
 * traversal -- which is the difference between "the GPU is faster" and "the
 * host loop is in the wrong order", and those deserve different conclusions.
 */
struct OrderedView {
  vlasov::StridedDistribution v{};

  [[nodiscard]] double operator()(int i, int j, int k) const noexcept {
    return v(i, j, k);
  }
  template <class Fn> void for_each_owned(Fn &&fn) const {
    for (int k = v.kbegin; k < v.kend; ++k) {
      for (int j = 0; j < v.nvx; ++j) {
        for (int i = 0; i < v.nx; ++i) fn(i, j, k);
      }
    }
  }
};
static_assert(vlasov::DistributionView<OrderedView>,
              "OrderedView must still satisfy what reduce_velocity requires");

/// One phase-space shape to measure.
struct Shape {
  int nx{64};
  int nvx{64};
  int nvy{64};
  [[nodiscard]] double cells() const {
    return static_cast<double>(nx) * static_cast<double>(nvx) *
           static_cast<double>(nvy);
  }
  [[nodiscard]] std::string label() const {
    return std::to_string(nx) + "x" + std::to_string(nvx) + "x" +
           std::to_string(nvy);
  }
};

/// Parse `64,128,256:128:128` -- a comma-separated list where a bare number
/// is a cube and `a:b:c` is `nx:nvx:nvy`.
[[nodiscard]] std::vector<Shape> parse_shapes(const std::string &spec) {
  std::vector<Shape> out;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    Shape s;
    const auto c1 = item.find(':');
    if (c1 == std::string::npos) {
      s.nx = s.nvx = s.nvy = std::atoi(item.c_str());
    } else {
      const auto c2 = item.find(':', c1 + 1);
      if (c2 == std::string::npos) {
        throw std::invalid_argument("--sizes: '" + item +
                                    "' needs either N or nx:nvx:nvy");
      }
      s.nx = std::atoi(item.substr(0, c1).c_str());
      s.nvx = std::atoi(item.substr(c1 + 1, c2 - c1 - 1).c_str());
      s.nvy = std::atoi(item.substr(c2 + 1).c_str());
    }
    out.push_back(s);
  }
  if (out.empty()) throw std::invalid_argument("--sizes selected no shape");
  return out;
}

/// Mean and sample standard deviation of a set of timings, in seconds.
struct Stat {
  double mean{0.0};
  double sd{0.0};
  [[nodiscard]] double rel_sd() const {
    return mean > 0.0 ? sd / mean : 0.0;
  }
};

[[nodiscard]] Stat summarise(const std::vector<double> &v) {
  Stat s;
  if (v.empty()) return s;
  for (double x : v) s.mean += x;
  s.mean /= static_cast<double>(v.size());
  if (v.size() > 1) {
    for (double x : v) s.sd += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(s.sd / static_cast<double>(v.size() - 1));
  }
  return s;
}

/**
 * @brief A runnable Weibel-like state at a given shape.
 *
 * Weibel rather than an empty box or a uniform Maxwellian, for two reasons
 * that both change the timing: the fields are non-zero, so the velocity
 * shifts are genuinely non-integer (an integer shift takes a branch in the
 * host gather that a real run never takes), and `f` is a smooth
 * bi-Maxwellian, so the `log` in the entropy column of the reduction is
 * evaluated on a realistic range rather than on zeros.
 */
struct Bench {
  SimParams p;
  std::unique_ptr<PhaseSpace> ps;
  std::unique_ptr<Stepper> st;
  double dt{0.0};

  Bench(const Shape &shape, int interp_order) {
    const double twopi = 2.0 * std::acos(-1.0);
    const double vth = 0.05;
    const double vthy = 0.15;
    p.nx = shape.nx;
    p.nvx = shape.nvx;
    p.nvy = shape.nvy;
    p.Lx = twopi / 0.5;
    p.v_max = 8.0 * vthy;
    p.v_thermal = vthy;
    p.interp_order = interp_order;
    p.electrostatic = false;
    p.species.clear();
    p.species.push_back(Species{"electron", -1.0, 1.0});
    p.validate();

    const int halo = vlasov::required_halo_width(4.0, interp_order);
    ps = std::make_unique<PhaseSpace>(p, halo, MPI_COMM_WORLD);
    st = std::make_unique<Stepper>(p, *ps);

    const double k = p.k_skin(1);
    const double amp = 1.0e-5;
    ps->initialise(0, [&](double x, double vx, double vy) {
      return vlasov::ics::density_perturbation(x, k, amp) *
             vlasov::ics::bi_maxwellian(vx, vy, vth, vthy);
    });
    st->deposit_all();
    const auto sol = vlasov::solve_gauss(st->line, st->sources.rho,
                                         p.neutrality_tol);
    st->fields.Ex = sol.Ex;
    for (int i = 0; i < p.nx; ++i) {
      // A finite B_z is what makes the two velocity shifts depend on the
      // other velocity coordinate; with B_z = 0 every line on a plane
      // shares one shift and the host gather gets a cache behaviour no real
      // electromagnetic run has.
      st->fields.Bz[i] = 1.0e-2 * std::cos(k * p.x_of(i));
      st->fields.Ey[i] = 1.0e-3 * std::sin(k * p.x_of(i));
    }
    dt = 0.2 * vlasov::step_limit(p, 1.0, 1.0e-2, 1.0e-2, halo);
  }
};

/// One row of the report: a phase, where it ran, and what it cost.
struct Row {
  std::string phase;
  std::string where;
  Stat t;
  double bytes{0.0}; ///< payload moved, for an effective-bandwidth column
};

void print_table(const Shape &shape, double dt, const std::vector<Row> &rows,
                 double cpu_step, double dev_step) {
  std::printf("\n  shape %s = %.0f cells, %.3f GB/brick, dt = %.4g\n",
              shape.label().c_str(), shape.cells(), shape.cells() * 8.0 / 1e9,
              dt);
  std::printf("  %-22s %-7s %12s %8s %12s\n", "phase", "where", "s/call",
              "rel sd", "GB/s");
  std::printf("  %s\n", std::string(66, '-').c_str());
  for (const auto &r : rows) {
    const double bw =
        (r.bytes > 0.0 && r.t.mean > 0.0) ? r.bytes / r.t.mean / 1e9 : 0.0;
    if (bw > 0.0) {
      std::printf("  %-22s %-7s %12.6f %8.3f %12.1f\n", r.phase.c_str(),
                  r.where.c_str(), r.t.mean, r.t.rel_sd(), bw);
    } else {
      std::printf("  %-22s %-7s %12.6f %8.3f %12s\n", r.phase.c_str(),
                  r.where.c_str(), r.t.mean, r.t.rel_sd(), "-");
    }
  }
  std::printf("  %s\n", std::string(66, '-').c_str());
  std::printf("  %-22s %-7s %12.6f\n", "full Strang step", "host", cpu_step);
  std::printf("  %-22s %-7s %12.6f   speedup %.2fx\n", "full Strang step",
              "hybrid", dev_step, dev_step > 0.0 ? cpu_step / dev_step : 0.0);
}

void print_usage(std::ostream &os, const char *exe) {
  os << "usage: " << exe << " [--key=value ...]\n\n"
     << "Per-phase cost of the 1D2V Vlasov step, host and device.\n\n"
     << "  --sizes=LIST     comma list; N is a cube, nx:nvx:nvy otherwise\n"
     << "                   (64,128,192,256)\n"
     << "  --reps=N         timed repetitions per phase, after a warm-up (5)\n"
     << "  --interp=N       Lagrange points/order                        (5)\n"
     << "  --csv=PATH       append one row per (shape, phase)\n"
     << "  --run-id=NAME    identifier written into every CSV row   (hipcost)\n";
}

int run(int argc, char **argv, int rank, int nproc) {
  vlasov::Options opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) print_usage(std::cout, argv[0]);
    return EXIT_SUCCESS;
  }
  const std::string sizes = opt.text("sizes", "64,128,192,256");
  const int reps = opt.integer("reps", 5);
  const int interp = opt.integer("interp", 5);
  const std::string csv = opt.text("csv", "");
  const std::string run_id = opt.text("run-id", "hipcost");
  opt.require_all_consumed();
  if (reps < 1) throw std::invalid_argument("--reps must be >= 1");

  vlasov::hip::bind_local_device(rank);

  if (rank == 0) {
    std::printf("vlasov_hip_cost: %d rank(s), device %s\n", nproc,
                vlasov::hip::device_name());
    std::printf("  host transport timed with measure_mass OFF (the operator,\n"
                "  not the operator plus its conservation diagnostic)\n");
  }

  vlasov::CsvAppender out;
  if (!csv.empty()) {
    out = vlasov::CsvAppender(
        csv,
        "run_id,ranks,nx,nvx,nvy,cells,interp,dt,phase,where,seconds,rel_sd,"
        "bytes,reps",
        rank);
  }

  for (const Shape &shape : parse_shapes(sizes)) {
    Bench b(shape, interp);
    auto &ps = *b.ps;
    auto &st = *b.st;
    const auto Bz = st.effective_bz();
    const double h = 0.5 * b.dt;
    const double qm = b.p.species[0].qm();
    const double brick_bytes =
        static_cast<double>(ps.nx()) * static_cast<double>(ps.nvx()) *
        static_cast<double>(ps.nvy_local()) * 8.0;

    // The operator's own cost, not the operator plus a diagnostic.
    st.work.measure_mass = false;

    std::vector<Row> rows;
    auto time_it = [&](const char *phase, const char *where, double bytes,
                       auto &&fn) {
      fn(); // warm-up: FFTW plan, first kernel load, first allocation
      std::vector<double> samples;
      samples.reserve(static_cast<std::size_t>(reps));
      for (int r = 0; r < reps; ++r) {
        const double t0 = wall_seconds();
        fn();
        samples.push_back(wall_seconds() - t0);
      }
      rows.push_back(Row{phase, where, summarise(samples), bytes});
    };

    // ---- host phases ---------------------------------------------------
    // Payload column: the minimum traffic the phase must move through
    // memory. A gather reads p cells and writes one; the reduction reads
    // every cell once; the spectral shift reads and writes once.
    const double p_d = static_cast<double>(b.p.interp_order);
    time_it("A advect_x (dt/2)", "host", 2.0 * brick_bytes, [&] {
      vlasov::advect_x(ps, ps.f(0), h, st.xplan, st.work);
    });
    time_it("B advect_vx (dt/2)", "host", (p_d + 1.0) * brick_bytes, [&] {
      vlasov::advect_vx(ps, ps.f(0), qm, h, st.fields.Ex, Bz, b.p.interp_order,
                        st.work);
    });
    time_it("C advect_vy (dt)", "host", (p_d + 1.0) * brick_bytes, [&] {
      vlasov::advect_vy(ps, ps.f(0), qm, b.dt, st.fields.Ey, Bz,
                        b.p.interp_order, st.work);
    });
    {
      vlasov::ReductionOptions ropt;
      ropt.v_thermal = b.p.v_thermal;
      ropt.comm = ps.comm();
      // As the application does it: StridedDistribution's own traversal.
      time_it("D moments", "host", brick_bytes, [&] {
        volatile double sink =
            vlasov::reduce_velocity(b.p, vlasov::view_of(ps, ps.f(0)), ropt)
                .number;
        (void)sink;
      });
      // The same reduction, same cells, memory order. See OrderedView.
      OrderedView ordered{vlasov::view_of(ps, ps.f(0))};
      time_it("D moments (mem order)", "host", brick_bytes, [&] {
        volatile double sink = vlasov::reduce_velocity(b.p, ordered, ropt).number;
        (void)sink;
      });
    }
    time_it("E fields", "host", 0.0, [&] { st.update_fields(b.dt); });

    // ---- device phases --------------------------------------------------
    {
      DeviceStepper ds(st, ps);
      ds.upload_all();
      auto &brick = ds.brick(0);

      time_it("  H2D whole brick", "bus", brick_bytes,
              [&] { brick.upload(ps.f(0)); vlasov::hip::device_synchronize(); });
      time_it("  D2H whole brick", "bus", brick_bytes,
              [&] { brick.download(ps.f(0)); vlasov::hip::device_synchronize(); });

      time_it("B advect_vx (dt/2)", "device", (p_d + 1.0) * brick_bytes, [&] {
        ds.device_advect_vx(brick, qm, h, st.fields.Ex, Bz);
      });
      time_it("C advect_vy (dt)", "device", (p_d + 1.0) * brick_bytes, [&] {
        ds.device_advect_vy(brick, qm, b.dt, st.fields.Ey, Bz);
      });
      {
        vlasov::ReductionOptions ropt;
        ropt.v_thermal = b.p.v_thermal;
        ropt.comm = ps.comm();
        time_it("D moments", "device", brick_bytes, [&] {
          volatile double sink = ds.reduce_device(brick, ropt).number;
          (void)sink;
        });
      }
    }

    // ---- end to end -----------------------------------------------------
    // Two whole Strang steps, host and hybrid, from the same state. This is
    // the only number that answers the question the port was built to ask,
    // because it carries the four brick copies phase A forces.
    double cpu_step = 0.0;
    {
      Bench fresh(shape, interp);
      fresh.st->work.measure_mass = false;
      fresh.st->advance(fresh.dt); // warm-up
      std::vector<double> s;
      for (int r = 0; r < reps; ++r) {
        const double t0 = wall_seconds();
        fresh.st->advance(fresh.dt);
        s.push_back(wall_seconds() - t0);
      }
      cpu_step = summarise(s).mean;
    }
    double dev_step = 0.0;
    vlasov::hip::PhaseTimings split;
    {
      Bench fresh(shape, interp);
      fresh.st->work.measure_mass = false;
      DeviceStepper ds(*fresh.st, *fresh.ps);
      ds.upload_all();
      ds.advance(fresh.dt); // warm-up
      ds.timings().clear();
      std::vector<double> s;
      for (int r = 0; r < reps; ++r) {
        const double t0 = wall_seconds();
        ds.advance(fresh.dt);
        s.push_back(wall_seconds() - t0);
      }
      dev_step = summarise(s).mean;
      split = ds.timings();
    }

    if (rank == 0) {
      print_table(shape, b.dt, rows, cpu_step, dev_step);
      const double n = std::max(1, split.steps);
      std::printf("  hybrid step split (s/step): A_host %.6f  B %.6f  C %.6f  "
                  "D %.6f\n",
                  split.advect_x / n, split.advect_vx / n, split.advect_vy / n,
                  split.moments / n);
      std::printf("                              fields %.6f  coeffs %.6f  "
                  "halo %.6f\n",
                  split.fields / n, split.coeffs / n, split.halo / n);
      std::printf("                              H2D %.6f  D2H %.6f  -> "
                  "transfers are %.1f%% of the hybrid step\n",
                  split.h2d / n, split.d2h / n,
                  100.0 * split.transfer_fraction());
    }

    if (out.active()) {
      char buf[512];
      auto emit = [&](const std::string &phase, const std::string &where,
                      const Stat &t, double bytes) {
        std::snprintf(buf, sizeof(buf),
                      "%s,%d,%d,%d,%d,%.0f,%d,%.10g,%s,%s,%.9g,%.4g,%.0f,%d",
                      run_id.c_str(), nproc, shape.nx, shape.nvx, shape.nvy,
                      shape.cells(), interp, b.dt, phase.c_str(), where.c_str(),
                      t.mean, t.rel_sd(), bytes, reps);
        out.row(std::string(buf));
      };
      for (const auto &r : rows) emit(r.phase, r.where, r.t, r.bytes);
      emit("full step", "host", Stat{cpu_step, 0.0}, 0.0);
      emit("full step", "hybrid", Stat{dev_step, 0.0}, 0.0);
      const double n = std::max(1, split.steps);
      emit("split A_host", "hybrid", Stat{split.advect_x / n, 0.0}, 0.0);
      emit("split B", "hybrid", Stat{split.advect_vx / n, 0.0}, 0.0);
      emit("split C", "hybrid", Stat{split.advect_vy / n, 0.0}, 0.0);
      emit("split D", "hybrid", Stat{split.moments / n, 0.0}, 0.0);
      emit("split fields", "hybrid", Stat{split.fields / n, 0.0}, 0.0);
      emit("split coeffs", "hybrid", Stat{split.coeffs / n, 0.0}, 0.0);
      emit("split halo", "hybrid", Stat{split.halo / n, 0.0}, 0.0);
      emit("split H2D", "hybrid", Stat{split.h2d / n, 0.0}, 0.0);
      emit("split D2H", "hybrid", Stat{split.d2h / n, 0.0}, 0.0);
    }
  }

  if (rank == 0) std::printf("\n");
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    status = run(argc, argv, rank, nproc);
  } catch (const std::exception &e) {
    std::cerr << "vlasov_hip_cost[" << rank << "]: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
