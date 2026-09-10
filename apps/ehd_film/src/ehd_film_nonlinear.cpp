// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ehd_film_nonlinear.cpp
 * @brief Nonlinear compliant lubrication under a localized load (`#116`).
 *
 * The science driver for `#116`. The JSON-session binary `ehd_film` keeps
 * the constant-mobility linear bending-relaxation model as the exact
 * \f$k^6\f$ verifier; this binary solves the full nonlinear coupling
 *
 * \f[
 *   p = B\nabla^4h - \gamma\nabla^2h - \Pi(h) + p_{\mathrm{ext}}(x,y,t),
 *   \qquad
 *   \partial_t h = \nabla\cdot\bigl[M(h)\nabla p\bigr],
 *   \qquad
 *   M(h) = M_0(h/h_0)^3,
 * \f]
 *
 * with a Gaussian load applied for `0 <= t < t_load` and then removed, and
 * reports central deflection, pressure, spreading radius, displaced volume
 * and total volume conservation. See `ehd_film/nonlinear.hpp` for the sign
 * conventions.
 *
 * Usage: `ehd_film_nonlinear CASE.json`
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/field_snapshots.hpp>
#include <openpfc_apps/spectral_flux.hpp>

#include <ehd_film/ehd_film_physics.hpp>
#include <ehd_film/nonlinear.hpp>

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
      throw std::invalid_argument("usage: ehd_film_nonlinear CASE.json");
    std::ifstream in(argv[1]);
    if (!in) throw std::runtime_error(std::string("cannot open ") + argv[1]);
    json cfg = json::parse(in);

    const auto &d = cfg.at("domain");
    const int Lx = d.at("Lx"), Ly = d.at("Ly"), Lz = d.value("Lz", 1);
    const double dx = d.value("dx", 1.0);
    const auto domain = pfc::domain::create(pfc::GridSize({Lx, Ly, Lz}),
                                            pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                            pfc::GridSpacing({dx, dx, dx}));

    ehd_film::EhdFilmParams p;
    ehd_film::apply_ehd_film_json(cfg.at("model").at("params"), p);

    const auto &ts = cfg.at("timestepping");
    const double t1 = ts.at("t1"), dt = ts.at("dt"), saveat = ts.value("saveat", -1.0);

    const auto &ic = cfg.value("initial_conditions", json::object());
    const double amp = ic.value("amplitude", 0.0);
    const std::uint64_t seed = ic.value("seed", 1234u);

    const auto &lc = cfg.value("load", json::object());
    const double load_p0 = lc.value("p0", 0.0);
    const double load_a = lc.value("a", 4.0);
    const double load_t = lc.value("t_load", 0.0);

    pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
    auto &h = stack.u();

    // h(x,y,0) = h0 (uniform gap): the disturbance in the science preset
    // comes entirely from the applied load, not the initial condition.
    // Optional broadband noise exercises the same path the k^6 verifier and
    // the volume-conservation test use.
    const auto n = pfc::domain::get_size(domain);
    h.apply([&](const pfc::Real3 &x) {
      const int i = int(std::lround(x[0] / dx));
      const int j = int(std::lround(x[1] / dx));
      const int k = int(std::lround(x[2] / dx));
      const double xi = (amp != 0.0) ? hashed_noise(i, j, k, n, seed) : 0.0;
      return p.h0 * (1.0 + amp * xi);
    });

    const auto load =
        ehd_film::GaussianLoad::centred(domain, load_p0, load_a, load_t);

    // ETD linear part: the constant-mobility operator about h0. The flux
    // remainder carries everything the linearisation leaves out (cubic
    // mobility, disjoining nonlinearity, the load), so with A=gamma=0 and a
    // constant mobility the nonlinear solver reduces to the linear k^6
    // verifier -- asserted in the tests, not assumed.
    const double B = p.B, gamma = p.gamma, M0 = p.M0, Pip0 = p.Pip0;
    pfc::apps::FluxETD stepper(domain, stack.fft(), dt, [=](double k_lap) {
      const double k2 = k_lap * k_lap;
      return M0 * B * k_lap * k2 - M0 * gamma * k2 - M0 * Pip0 * k_lap;
    });

    const ehd_film::CubicMobility mobility{p.M0, p.h0};
    const ehd_film::EhdFilmPointwise pw{
        .A = p.A, .h0 = p.h0, .h_star = p.h_star, .Pi0 = p.Pi0, .Pip0 = p.Pip0};

    pfc::data::Field<double> pi_real(domain, stack.fft().get_inbox_bounds(), 0);
    pfc::data::Field<std::complex<double>> pi_hat(
        domain, stack.fft().get_outbox_bounds(), 0);
    pfc::data::Field<double> pext_real(domain, stack.fft().get_inbox_bounds(), 0);
    pfc::data::Field<std::complex<double>> pext_hat(
        domain, stack.fft().get_outbox_bounds(), 0);
    std::vector<double> k_lap(stack.fft().size_outbox(), 0.0);
    pfc::fft::kspace::for_each_kpoint(
        stack.fft().get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          k_lap[i] = -(kx * kx + ky * ky + kz * kz);
        });

    // Evaluated at this time whenever `potential` runs; the driver sets it
    // to the instant the step (or the diagnostic sample) represents before
    // calling into the stepper -- FluxETD's callback signature carries no
    // time argument of its own.
    double load_time = 0.0;

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

      const auto nloc = pext_real.local_size();
      for (int k = 0; k < nloc[2]; ++k)
        for (int j = 0; j < nloc[1]; ++j)
          for (int i = 0; i < nloc[0]; ++i) {
            const auto x = pext_real.coords(i, j, k);
            pext_real(i, j, k) = load(x[0], x[1], load_time);
          }
      pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), pext_real,
                                                        pext_hat);

      h_hat.with_host_view([&](std::complex<double> *hv, std::size_t m) {
        pi_hat.with_host_view([&](std::complex<double> *piv, std::size_t) {
          pext_hat.with_host_view([&](std::complex<double> *pev, std::size_t) {
            out.with_host_view([&](std::complex<double> *o, std::size_t) {
              for (std::size_t i = 0; i < m; ++i)
                o[i] = B * k_lap[i] * k_lap[i] * hv[i] - gamma * k_lap[i] * hv[i] -
                       piv[i] + pev[i];
            });
          });
        });
      });
    };

    // Optional `.vti` snapshots of the gap itself (`fields[]`). The
    // diagnostics CSV reduces the dent to a handful of scalars; the
    // snapshots are what show its shape, and how far the disturbance has
    // spread relative to the periodic box. See
    // openpfc_apps/field_snapshots.hpp.
    auto snapshots =
        pfc::apps::make_field_snapshot_writer(cfg, "h", h, MPI_COMM_WORLD);
    int snapshot_index = 0;

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
                   "step,time,h_center,deflection_center,p_max,p_min,"
                   "spreading_radius,displaced_volume,volume,"
                   "volume_rel_drift\n");
    }

    double volume0 = -1.0;
    double h_center_min = std::numeric_limits<double>::infinity();
    double p_max_max = -std::numeric_limits<double>::infinity();
    double spreading_radius_max = 0.0;
    double volume_rel_drift_max = 0.0;

    pfc::data::Field<std::complex<double>> h_hat_diag(
        domain, stack.fft().get_outbox_bounds(), 0);
    pfc::data::Field<std::complex<double>> p_hat_diag(
        domain, stack.fft().get_outbox_bounds(), 0);
    pfc::data::Field<double> p_real_diag(domain, stack.fft().get_inbox_bounds(), 0);

    auto report = [&](int step, double t) {
      load_time = t;
      pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), h, h_hat_diag);
      potential(h_hat_diag, h, p_hat_diag);
      pfc::sim::SpectralETDOps<pfc::HostSpace>::backward(stack.fft(), p_hat_diag,
                                                          p_real_diag);
      auto s = ehd_film::sample_ehd_film(h, p_real_diag, domain, p.h0,
                                         MPI_COMM_WORLD);
      pfc::apps::write_field_snapshot(snapshots.get(), snapshot_index++, h);
      if (volume0 < 0.0) volume0 = s.volume;
      const double drift = (volume0 != 0.0) ? (s.volume - volume0) / volume0 : 0.0;
      h_center_min = std::min(h_center_min, s.h_center);
      p_max_max = std::max(p_max_max, s.p_max);
      spreading_radius_max = std::max(spreading_radius_max, s.spreading_radius);
      volume_rel_drift_max = std::max(volume_rel_drift_max, std::abs(drift));
      if (out) {
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << std::setprecision(17) << step << ',' << t << ',' << s.h_center
             << ',' << (p.h0 - s.h_center) << ',' << s.p_max << ',' << s.p_min
             << ',' << s.spreading_radius << ',' << s.displaced_volume << ','
             << s.volume << ',' << drift << '\n';
        std::fputs(line.str().c_str(), out.get());
        std::fflush(out.get());
      }
    };

    report(0, 0.0);
    double t = 0.0;
    const int n_steps = int(std::llround(t1 / dt));
    const int every =
        (saveat > 0.0) ? std::max(1, int(std::llround(saveat / dt))) : n_steps;
    for (int step = 1; step <= n_steps; ++step) {
      load_time = t; // forcing over [t, t+dt) uses the pre-step time
      t = stepper.step(t, h, potential, mobility);
      if (step % every == 0 || step == n_steps) report(step, t);
    }

    if (rank == 0) {
      std::printf("EHD_FILM_NONLINEAR h_center_min=%.17g p_max_max=%.17g "
                  "spreading_radius_max=%.17g volume_rel_drift_max=%.17g\n",
                  h_center_min, p_max_max, spreading_radius_max,
                  volume_rel_drift_max);
    }
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "ehd_film_nonlinear: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
