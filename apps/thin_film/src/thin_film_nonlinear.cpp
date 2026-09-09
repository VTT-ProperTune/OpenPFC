// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file thin_film_nonlinear.cpp
 * @brief Dewetting and rupture with the full \f$h^3\f$ lubrication mobility.
 *
 * The science driver for `#114`. The JSON-session binary `thin_film` keeps the
 * constant-mobility linear model as an analytical verifier; this one solves
 *
 * \f[
 *   \partial_t h = \nabla\cdot\bigl[M(h)\nabla p\bigr],\qquad
 *   p = -\gamma\nabla^2 h - \Pi(h),\qquad
 *   M(h) = M_0 (h/h_0)^3 ,
 * \f]
 *
 * and reports the observables a dewetting experiment would: rupture time,
 * minimum thickness, hole area fraction, dominant spacing, and the liquid
 * volume that must not change.
 *
 * Usage: `thin_film_nonlinear CASE.json`
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/spectral_flux.hpp>
#include <openpfc_apps/structure_factor.hpp>

#include <thin_film/nonlinear.hpp>
#include <thin_film/thin_film_physics.hpp>

namespace {

using json = nlohmann::json;

/// Deterministic broadband perturbation, identical on any decomposition.
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

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: thin_film_nonlinear CASE.json");
    std::ifstream in(argv[1]);
    if (!in) throw std::runtime_error(std::string("cannot open ") + argv[1]);
    json cfg = json::parse(in);

    const auto &d = cfg.at("domain");
    const int Lx = d.at("Lx"), Ly = d.at("Ly"), Lz = d.value("Lz", 1);
    const double dx = d.value("dx", 1.0);
    const auto domain = pfc::domain::create(pfc::GridSize({Lx, Ly, Lz}),
                                            pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                            pfc::GridSpacing({dx, dx, dx}));

    thin_film::ThinFilmParams p;
    thin_film::apply_thin_film_json(cfg.at("model").at("params"), p);

    const auto &ts = cfg.at("timestepping");
    const double t1 = ts.at("t1"), dt = ts.at("dt"), saveat = ts.value("saveat", -1.0);

    const auto &ic = cfg.at("initial_conditions");
    const double amp = ic.value("amplitude", 0.01);
    const std::uint64_t seed = ic.value("seed", 1234u);
    const double defect_amp = ic.value("defect_amplitude", 0.0);
    const double defect_sigma = ic.value("defect_sigma", 4.0);

    pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
    auto &h = stack.u();

    // One initial condition: mean film + broadband noise + optional defect.
    const auto defect =
        thin_film::GaussianDefect::centred(domain, defect_amp, defect_sigma);
    const auto n = pfc::domain::get_size(domain);
    const auto box = h.box();
    h.apply([&](const pfc::Real3 &x) {
      const int i = int(std::lround(x[0] / dx));
      const int j = int(std::lround(x[1] / dx));
      const int k = int(std::lround(x[2] / dx));
      const double xi = hashed_noise(i, j, k, n, seed);
      return p.h0 * (1.0 + amp * xi + defect(x[0], x[1]));
    });
    (void)box;

    // ETD linear part: the constant-mobility operator about h0. The flux
    // remainder carries everything the linearisation leaves out, so the
    // nonlinear solver reduces to the linear one when M is constant.
    const double gamma = p.gamma, M0 = p.M0, Pip0 = p.Pip0;
    pfc::apps::FluxETD stepper(domain, stack.fft(), dt, [=](double k_lap) {
      return -M0 * gamma * k_lap * k_lap - M0 * Pip0 * k_lap;
    });

    const thin_film::CubicMobility mobility{p.M0, p.h0};
    const thin_film::ThinFilmPointwise pw{.A = p.A, .h0 = p.h0,
                                          .h_star = p.h_star, .Pi0 = p.Pi0,
                                          .Pip0 = p.Pip0};

    // p_hat = -gamma * k_lap * h_hat - FFT(Pi(h))
    pfc::data::Field<double> pi_real(domain, stack.fft().get_inbox_bounds(), 0);
    pfc::data::Field<std::complex<double>> pi_hat(
        domain, stack.fft().get_outbox_bounds(), 0);
    std::vector<double> k_lap(stack.fft().size_outbox(), 0.0);
    pfc::fft::kspace::for_each_kpoint(
        stack.fft().get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          k_lap[i] = -(kx * kx + ky * ky + kz * kz);
        });

    auto potential = [&](pfc::data::Field<std::complex<double>> &h_hat,
                         pfc::data::Field<double> &hh,
                         pfc::data::Field<std::complex<double>> &out) {
      hh.with_host_view([&](double *hv, std::size_t cnt) {
        pi_real.with_host_view([&](double *pv, std::size_t) {
          for (std::size_t i = 0; i < cnt; ++i) pv[i] = pw.Pi(hv[i]);
        });
      });
      pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), pi_real,
                                                        pi_hat);
      h_hat.with_host_view([&](std::complex<double> *hv, std::size_t m) {
        pi_hat.with_host_view([&](std::complex<double> *pv, std::size_t) {
          out.with_host_view([&](std::complex<double> *o, std::size_t) {
            for (std::size_t i = 0; i < m; ++i)
              o[i] = -gamma * k_lap[i] * hv[i] - pv[i];
          });
        });
      });
    };

    // Diagnostics CSV, rank 0, never overwriting.
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
                   "dominant_spacing,ruptured\n");
    }

    double rupture_time = -1.0;
    const int n_steps = int(std::llround(t1 / dt));
    const int every =
        (saveat > 0.0) ? std::max(1, int(std::llround(saveat / dt))) : n_steps;

    auto report = [&](int step, double t) {
      auto s = thin_film::sample_film(h, domain, p.h0, 0.05, MPI_COMM_WORLD);
      pfc::data::Field<std::complex<double>> hh(
          domain, stack.fft().get_outbox_bounds(), 0);
      pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), h, hh);
      hh.with_host_view([&](std::complex<double> *hv, std::size_t) {
        const auto sf = pfc::apps::shell_average(stack.fft().get_outbox_bounds(),
                                                 domain, hv, MPI_COMM_WORLD, 64);
        s.dominant_spacing = sf.dominant_wavelength();
      });
      if (s.ruptured && rupture_time < 0.0) rupture_time = t;
      if (out) {
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << std::setprecision(17) << step << ',' << t << ',' << s.min_h << ','
             << s.max_h << ',' << s.mean_h << ',' << s.volume << ','
             << s.hole_area_fraction << ',' << s.dominant_spacing << ','
             << (s.ruptured ? 1 : 0) << '\n';
        std::fputs(line.str().c_str(), out.get());
        std::fflush(out.get());
      }
    };

    report(0, 0.0);
    double t = 0.0;
    for (int step = 1; step <= n_steps; ++step) {
      t = stepper.step(t, h, potential, mobility);
      if (step % every == 0 || step == n_steps) report(step, t);
    }

    if (rank == 0) {
      auto s = thin_film::sample_film(h, domain, p.h0, 0.05, MPI_COMM_WORLD);
      std::printf("THIN_FILM_NONLINEAR min_h=%.17g volume=%.17g "
                  "hole_fraction=%.17g rupture_time=%.17g\n",
                  s.min_h, s.volume, s.hole_area_fraction, rupture_time);
    } else {
      (void)thin_film::sample_film(h, domain, p.h0, 0.05, MPI_COMM_WORLD);
    }
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "thin_film_nonlinear: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
