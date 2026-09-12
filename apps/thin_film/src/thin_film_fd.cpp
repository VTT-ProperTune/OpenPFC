// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file thin_film_fd.cpp
 * @brief Conservative face-flux FD solver for the full h^3 lubrication
 *        model -- the solver that integrates through rupture where the
 *        spectral `thin_film_nonlinear` cannot.
 *
 * Same equation, same JSON schema as `thin_film_nonlinear` (`#114`), so the
 * *same* case file drives both binaries -- that is the point: it is what
 * lets the two be cross-validated on one physical case rather than two
 * loosely comparable ones. An optional `"fd"` block selects this solver's
 * own knobs (face mobility average, curvature-operator order):
 *
 * @code
 * "fd": { "face_mobility": "harmonic", "order": 2, "hole_threshold_frac": 0.5 }
 * @endcode
 *
 * Usage: `thin_film_fd CASE.json`
 */

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/field_snapshots.hpp>
#include <openpfc_apps/gather.hpp>
#include <openpfc_apps/structure_factor.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

#include <thin_film/fd_flux.hpp>
#include <thin_film/nonlinear.hpp>
#include <thin_film/thin_film_physics.hpp>

namespace {

using json = nlohmann::json;

/// Deterministic broadband perturbation, identical to `thin_film_nonlinear`'s
/// so the two solvers start from bit-identical initial conditions.
double hashed_noise(int i, int j, int k, const pfc::Int3 &n, std::uint64_t seed) {
  std::uint64_t x = seed + std::uint64_t(i) +
                    std::uint64_t(n[0]) *
                        (std::uint64_t(j) + std::uint64_t(n[1]) * std::uint64_t(k));
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return 2.0 * (double(x >> 11) / double(1ULL << 53)) - 1.0;
}

thin_film::FaceMobility parse_face_mobility(const std::string &s) {
  if (s == "harmonic") return thin_film::FaceMobility::Harmonic;
  if (s == "arithmetic") return thin_film::FaceMobility::Arithmetic;
  throw std::invalid_argument("fd.face_mobility must be \"harmonic\" or "
                              "\"arithmetic\", got \"" +
                              s + "\"");
}

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    if (argc != 2) throw std::invalid_argument("usage: thin_film_fd CASE.json");
    std::ifstream in(argv[1]);
    if (!in) throw std::runtime_error(std::string("cannot open ") + argv[1]);
    json cfg = json::parse(in);

    const auto &d = cfg.at("domain");
    const int Lx = d.at("Lx"), Ly = d.at("Ly"), Lz = d.value("Lz", 1);
    const double dx = d.value("dx", 1.0);
    if (Lz != 1) throw std::invalid_argument("thin_film_fd: 2-D only (Lz must be 1)");
    const auto domain = pfc::domain::create(pfc::GridSize({Lx, Ly, Lz}),
                                            pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                            pfc::GridSpacing({dx, dx, dx}));
    const auto decomp = pfc::decomposition::create(domain, nproc);

    thin_film::ThinFilmParams p;
    thin_film::apply_thin_film_json(cfg.at("model").at("params"), p);

    const auto &ts = cfg.at("timestepping");
    const double t1 = ts.at("t1"), dt = ts.at("dt"), saveat = ts.value("saveat", -1.0);

    const auto &ic = cfg.at("initial_conditions");
    const double amp = ic.value("amplitude", 0.01);
    const std::uint64_t seed = ic.value("seed", 1234u);
    const double defect_amp = ic.value("defect_amplitude", 0.0);
    const double defect_sigma = ic.value("defect_sigma", 4.0);

    const json fd_cfg = cfg.value("fd", json::object());
    const auto face_kind =
        parse_face_mobility(fd_cfg.value("face_mobility", std::string("harmonic")));
    const int order = fd_cfg.value("order", 2);
    const double hole_threshold_frac = fd_cfg.value("hole_threshold_frac", 0.5);
    const int sf_bins = fd_cfg.value("structure_factor_bins", 64);

    const auto &box = pfc::decomposition::local_box(decomp, rank);
    const int nx = box.size[0], ny = box.size[1];
    std::vector<double> h(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));

    const auto defect =
        thin_film::GaussianDefect::centred(domain, defect_amp, defect_sigma);
    const pfc::Int3 n{Lx, Ly, Lz};
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = box.low[0] + ix, gy = box.low[1] + iy;
        const double xi = hashed_noise(gx, gy, 0, n, seed);
        const double x = gx * dx, y = gy * dx;
        h[static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * nx] =
            p.h0 * (1.0 + amp * xi + defect(x, y));
      }
    }

    thin_film::FDFluxSolver solver(domain, decomp, rank, MPI_COMM_WORLD, h, order);
    const thin_film::CubicMobility mobility{p.M0, p.h0};
    const thin_film::ThinFilmPointwise pw{.A = p.A, .h0 = p.h0,
                                          .h_star = p.h_star, .Pi0 = p.Pi0,
                                          .Pip0 = p.Pip0};

    // Rank-0-only serial FFT for the dominant-spacing diagnostic: the FD
    // solver's Cartesian decomposition has nothing to do with heffte's
    // pencils, so this is a second, independent, single-process transform
    // over a gathered copy of the field -- a diagnostic, not part of the
    // solve. `MPI_COMM_SELF` keeps it entirely local to rank 0.
    std::unique_ptr<pfc::sim::stacks::SpectralCPUStack> diag_stack;
    if (rank == 0) {
      diag_stack =
          std::make_unique<pfc::sim::stacks::SpectralCPUStack>(domain, 0, 1,
                                                                MPI_COMM_SELF);
    }

    // Rank-0 full-domain field for optional fields[] VTK dumps (gathered).
    std::unique_ptr<pfc::data::Field<double>> snap_field;
    std::unique_ptr<pfc::VTKWriter> snapshots;
    int snapshot_index = 0;
    if (rank == 0 && cfg.contains("fields")) {
      snap_field = std::make_unique<pfc::data::Field<double>>(
          domain, pfc::domain::index_box(domain), 0);
      snapshots = pfc::apps::make_field_snapshot_writer(cfg, "h", *snap_field,
                                                        MPI_COMM_SELF);
    }

    std::unique_ptr<std::FILE, int (*)(std::FILE *)> out(nullptr, std::fclose);
    if (rank == 0 && cfg.contains("diagnostics")) {
      const std::filesystem::path path =
          cfg.at("diagnostics").at("csv").get<std::string>();
      if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
      out.reset(std::fopen(path.string().c_str(), "w"));
      if (!out) throw std::runtime_error("cannot open diagnostics csv");
      std::fprintf(out.get(),
                   "step,time,min_h,max_h,mean_h,volume,hole_area_fraction,"
                   "dominant_spacing,n_holes,ruptured\n");
    }

    double rupture_time = -1.0;
    const int n_steps = int(std::llround(t1 / dt));
    const int every =
        (saveat > 0.0) ? std::max(1, int(std::llround(saveat / dt))) : n_steps;

    std::vector<double> global_xy;
    auto report = [&](int step, double t) {
      auto s = thin_film::sample_film_fd(h, domain, p.h0, 0.05, MPI_COMM_WORLD);
      pfc::apps::gather_global_xy_rank0(decomp, rank, nproc, MPI_COMM_WORLD, h, Lx,
                                        Ly, global_xy);
      int n_holes = 0;
      if (rank == 0) {
        auto &diag_u = diag_stack->u();
        diag_u.with_host_view([&](double *dv, std::size_t cnt) {
          for (std::size_t i = 0; i < cnt && i < global_xy.size(); ++i)
            dv[i] = global_xy[i];
        });
        pfc::data::Field<std::complex<double>> hh(
            domain, diag_stack->fft().get_outbox_bounds(), 0);
        pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(diag_stack->fft(), diag_u,
                                                           hh);
        hh.with_host_view([&](std::complex<double> *hv, std::size_t) {
          const auto sf = pfc::apps::shell_average(diag_stack->fft().get_outbox_bounds(),
                                                    domain, hv, MPI_COMM_SELF, sf_bins);
          s.dominant_spacing = sf.dominant_wavelength();
        });
        n_holes = thin_film::count_dry_regions_rank0(
            global_xy, Lx, Ly, hole_threshold_frac * p.h0);
      }
      if (s.ruptured && rupture_time < 0.0) rupture_time = t;
      if (rank == 0 && snapshots && snap_field && !global_xy.empty()) {
        snap_field->with_host_view([&](double *d, std::size_t n) {
          const std::size_t m = n < global_xy.size() ? n : global_xy.size();
          for (std::size_t i = 0; i < m; ++i) d[i] = global_xy[i];
        });
        pfc::apps::write_field_snapshot(snapshots.get(), snapshot_index++,
                                        *snap_field);
      }
      if (out) {
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << std::setprecision(17) << step << ',' << t << ',' << s.min_h << ','
             << s.max_h << ',' << s.mean_h << ',' << s.volume << ','
             << s.hole_area_fraction << ',' << s.dominant_spacing << ',' << n_holes
             << ',' << (s.ruptured ? 1 : 0) << '\n';
        std::fputs(line.str().c_str(), out.get());
        std::fflush(out.get());
      }
    };

    report(0, 0.0);
    double t = 0.0;
    for (int step = 1; step <= n_steps; ++step) {
      solver.step(h, dt, p.gamma, pw, mobility, face_kind);
      t = dt * step;
      if (step % every == 0 || step == n_steps) report(step, t);
    }

    auto s = thin_film::sample_film_fd(h, domain, p.h0, 0.05, MPI_COMM_WORLD);
    if (rank == 0) {
      std::printf("THIN_FILM_FD min_h=%.17g volume=%.17g hole_fraction=%.17g "
                  "rupture_time=%.17g\n",
                  s.min_h, s.volume, s.hole_area_fraction, rupture_time);
    }
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "thin_film_fd: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
