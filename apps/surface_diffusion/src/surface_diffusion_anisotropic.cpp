// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file surface_diffusion_anisotropic.cpp
 * @brief Patterned-nanosurface relaxation and orientation selection (`#115`).
 *
 * The science driver for `#115`. The JSON-session binary `surface_diffusion`
 * keeps the isotropic Mullins model as the analytical `k^4` verifier; this
 * one solves the orientation-dependent generalisation of
 * `surface_diffusion/anisotropic_flux.hpp`,
 *
 * \f[
 *   \partial_t h = \nabla\cdot\bigl[B(\theta)\,\nabla(\nabla^2 h)\bigr],
 *   \qquad B(\theta) = B_0\bigl[1+\epsilon_a\cos(m\theta)\bigr],
 * \f]
 *
 * starting from a *crossed sinusoidal corrugation* -- two orthogonal ridge
 * sets of equal wavenumber, `h0 + A[cos(2 pi nx x/Lx) + cos(2 pi ny y/Ly)]` --
 * so the same run answers both the "patterned surface relaxation" and the
 * "orientation selection" computational experiments in `#115`: an isotropic
 * run damps the x-ridges and y-ridges at the *same* rate (the linear symbol
 * depends only on `|k|`), while an anisotropic run with `m = 6` does not
 * (`theta = 0` and `theta = pi/2` are inequivalent under sixfold symmetry),
 * so the surviving orientation spectrum differs between the two runs even
 * though they start from the identical field.
 *
 * `m = 4` is deliberately *not* the default science-run symmetry: a fourfold
 * stiffness treats `theta = 0` and `theta = pi/2` as equivalent
 * (`cos(4*0) == cos(4*pi/2)`), so a plain x/y crossed corrugation would not
 * show a directional effect under it. `m = 4` is still fully supported and
 * unit-tested (fourfold symmetry of `B(theta)` itself, and the
 * `eps_a = 0` reduction), it is just not the orientation-selection
 * demonstration case here.
 *
 * Usage: `surface_diffusion_anisotropic CASE.json`
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/structure_factor.hpp>

#include <surface_diffusion/anisotropic_flux.hpp>
#include <surface_diffusion/anisotropy.hpp>

namespace {

using json = nlohmann::json;

struct Mode {
  int nx{0};
  int ny{0};
  double amplitude{0.0};
};

/// Surface roughness/geometry observables reported every `saveat`.
struct Sample {
  double mean_h{};
  double rms_roughness{};
  double max_grad_h{};
  double dominant_wavelength{};
  double domain_length{};
  double energy_kx_frac{};
  double energy_ky_frac{};
  double energy_diag_frac{};
};

Sample sample_surface(pfc::data::Field<double> &h, const pfc::Domain &domain,
                      pfc::fft::CPUFFT &fft, MPI_Comm comm) {
  double local_sum = 0.0, local_sumsq = 0.0, local_count = 0.0;
  h.with_host_view([&](const double *d, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      local_sum += d[i];
      local_sumsq += d[i] * d[i];
      local_count += 1.0;
    }
  });
  double g[3]{}, l[3]{local_sum, local_sumsq, local_count};
  MPI_Allreduce(l, g, 3, MPI_DOUBLE, MPI_SUM, comm);

  Sample s;
  s.mean_h = (g[2] > 0.0) ? g[0] / g[2] : 0.0;
  const double mean_sq = (g[2] > 0.0) ? g[1] / g[2] : 0.0;
  const double var = mean_sq - s.mean_h * s.mean_h;
  s.rms_roughness = std::sqrt(std::max(var, 0.0));

  pfc::data::Field<std::complex<double>> h_hat(domain, fft.get_outbox_bounds(), 0);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(fft, h, h_hat);

  pfc::apps::StructureFactor sf;
  h_hat.with_host_view([&](const std::complex<double> *hv, std::size_t) {
    sf = pfc::apps::shell_average(fft.get_outbox_bounds(), domain, hv, comm, 64);
  });
  s.dominant_wavelength = sf.dominant_wavelength();
  s.domain_length = sf.domain_length();

  // Directional spectral energy: split |h_hat|^2 by which axis dominates the
  // local wavevector. The k=0 (mean height) bin carries no orientation and is
  // excluded, matching `shell_average`'s convention.
  //
  // HeFFTe's r2c transform halves storage along x only: the outbox carries
  // kx in [0, Nyquist_x] with ky (and kz) spanning their full signed range.
  // A mode at kx=0 or kx=Nyquist is its own conjugate and appears once in
  // the *full* complex spectrum; every 0<kx<Nyquist mode has an unstored
  // conjugate at -kx that carries equal power. A y-oriented ridge
  // (kx=0, ky=+-k) is therefore stored as *two* explicit points while an
  // x-oriented ridge (kx=+-k, ky=0) is stored as *one* (its -kx twin is
  // implicit) -- weighting interior kx modes by 2 corrects for that, or the
  // kx/ky split is biased 2:1 toward ky regardless of any real anisotropy.
  const int Nx = pfc::domain::get_size(domain)[0];
  const int kx_nyquist = Nx / 2;
  double p_kx = 0.0, p_ky = 0.0, p_diag = 0.0;
  h_hat.with_host_view([&](const std::complex<double> *hv, std::size_t) {
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int ix, int, int) {
          (void)kz;
          const double ax = std::abs(kx), ay = std::abs(ky);
          if (ax == 0.0 && ay == 0.0) return;
          const double weight = (ix == 0 || ix == kx_nyquist) ? 1.0 : 2.0;
          const double power = weight * std::norm(hv[i]);
          if (ax > ay) {
            p_kx += power;
          } else if (ay > ax) {
            p_ky += power;
          } else {
            p_diag += power;
          }
        });
  });
  double gp[3]{}, lp[3]{p_kx, p_ky, p_diag};
  MPI_Allreduce(lp, gp, 3, MPI_DOUBLE, MPI_SUM, comm);
  const double total = gp[0] + gp[1] + gp[2];
  if (total > 0.0) {
    s.energy_kx_frac = gp[0] / total;
    s.energy_ky_frac = gp[1] / total;
    s.energy_diag_frac = gp[2] / total;
  }

  // max|grad h|, evaluated spectrally from the same transform.
  std::vector<double> kx_arr(fft.size_outbox(), 0.0), ky_arr(fft.size_outbox(), 0.0);
  pfc::fft::kspace::for_each_kpoint(
      fft.get_outbox_bounds(), domain,
      [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
        (void)kz;
        kx_arr[i] = kx;
        ky_arr[i] = ky;
      });
  pfc::data::Field<std::complex<double>> gx_hat(domain, fft.get_outbox_bounds(), 0);
  pfc::data::Field<std::complex<double>> gy_hat(domain, fft.get_outbox_bounds(), 0);
  h_hat.with_host_view([&](const std::complex<double> *hv, std::size_t n) {
    gx_hat.with_host_view([&](std::complex<double> *gx, std::size_t) {
      gy_hat.with_host_view([&](std::complex<double> *gy, std::size_t) {
        for (std::size_t i = 0; i < n; ++i) {
          gx[i] = std::complex<double>{0.0, kx_arr[i]} * hv[i];
          gy[i] = std::complex<double>{0.0, ky_arr[i]} * hv[i];
        }
      });
    });
  });
  pfc::data::Field<double> hx(domain, fft.get_inbox_bounds(), 0);
  pfc::data::Field<double> hy(domain, fft.get_inbox_bounds(), 0);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::backward(fft, gx_hat, hx);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::backward(fft, gy_hat, hy);
  double local_max = 0.0;
  hx.with_host_view([&](const double *gx, std::size_t n) {
    hy.with_host_view([&](const double *gy, std::size_t) {
      for (std::size_t i = 0; i < n; ++i)
        local_max = std::max(local_max, std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]));
    });
  });
  double global_max = 0.0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
  s.max_grad_h = global_max;
  return s;
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
      throw std::invalid_argument("usage: surface_diffusion_anisotropic CASE.json");
    std::ifstream in(argv[1]);
    if (!in) throw std::runtime_error(std::string("cannot open ") + argv[1]);
    json cfg = json::parse(in);

    const auto &d = cfg.at("domain");
    const int Lx = d.at("Lx"), Ly = d.at("Ly"), Lz = d.value("Lz", 1);
    if (Lz != 1)
      throw std::invalid_argument(
          "surface_diffusion_anisotropic: requires a 2-D domain (Lz == 1)");
    const double dx = d.value("dx", 1.0);
    const auto domain = pfc::domain::create(pfc::GridSize({Lx, Ly, Lz}),
                                            pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                            pfc::GridSpacing({dx, dx, dx}));

    surface_diffusion::AnisotropyParams params;
    surface_diffusion::apply_anisotropy_json(cfg.at("model").at("params"), params);

    const auto &ts = cfg.at("timestepping");
    const double t1 = ts.at("t1"), dt = ts.at("dt"), saveat = ts.value("saveat", -1.0);

    const auto &ic = cfg.at("initial_conditions");
    const double h0 = ic.value("h0", 0.0);
    std::vector<Mode> modes;
    for (const auto &m : ic.at("modes")) {
      modes.push_back({m.at("nx").get<int>(), m.at("ny").get<int>(),
                       m.at("amplitude").get<double>()});
    }

    pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
    auto &h = stack.u();

    // Crossed sinusoidal corrugation: two orthogonal ridge sets superposed on
    // a mean height, run isotropically and anisotropically from the same
    // deterministic field.
    const double Lx_phys = static_cast<double>(Lx) * dx;
    const double Ly_phys = static_cast<double>(Ly) * dx;
    const double twopi = 2.0 * std::numbers::pi;
    h.apply([&](const pfc::Real3 &x) {
      double v = h0;
      for (const auto &m : modes) {
        const double kx = twopi * static_cast<double>(m.nx) / Lx_phys;
        const double ky = twopi * static_cast<double>(m.ny) / Ly_phys;
        v += m.amplitude * std::cos(kx * x[0] + ky * x[1]);
      }
      return v;
    });

    surface_diffusion::AnisotropicSurfaceDiffusionETD stepper(
        domain, stack.fft(), dt,
        surface_diffusion::SurfaceStiffness::from_params(params));

    std::unique_ptr<std::FILE, int (*)(std::FILE *)> out(nullptr, std::fclose);
    if (rank == 0 && cfg.contains("diagnostics")) {
      const std::filesystem::path path =
          cfg.at("diagnostics").at("csv").get<std::string>();
      if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
      out.reset(std::fopen(path.string().c_str(), "w"));
      if (!out) throw std::runtime_error("cannot open diagnostics csv");
      std::fprintf(out.get(),
                   "step,time,mean_h,rms_roughness,max_grad_h,"
                   "dominant_wavelength,domain_length,energy_kx_frac,"
                   "energy_ky_frac,energy_diag_frac\n");
    }

    auto report = [&](int step, double t) {
      auto s = sample_surface(h, domain, stack.fft(), MPI_COMM_WORLD);
      if (out) {
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << std::setprecision(17) << step << ',' << t << ',' << s.mean_h << ','
             << s.rms_roughness << ',' << s.max_grad_h << ','
             << s.dominant_wavelength << ',' << s.domain_length << ','
             << s.energy_kx_frac << ',' << s.energy_ky_frac << ','
             << s.energy_diag_frac << '\n';
        std::fputs(line.str().c_str(), out.get());
        std::fflush(out.get());
      }
      return s;
    };

    const int n_steps = static_cast<int>(std::llround(t1 / dt));
    const int every =
        (saveat > 0.0) ? std::max(1, int(std::llround(saveat / dt))) : n_steps;

    Sample final_sample = report(0, 0.0);
    double t = 0.0;
    for (int step = 1; step <= n_steps; ++step) {
      t = stepper.step(t, h);
      if (step % every == 0 || step == n_steps) final_sample = report(step, t);
    }

    if (rank == 0) {
      std::printf(
          "SURFACE_DIFFUSION_ANISOTROPIC B0=%.17g eps_a=%.17g m=%d mean_h=%.17g "
          "rms_roughness=%.17g max_grad_h=%.17g energy_kx_frac=%.17g "
          "energy_ky_frac=%.17g energy_diag_frac=%.17g\n",
          params.B0, params.eps_a, params.m, final_sample.mean_h,
          final_sample.rms_roughness, final_sample.max_grad_h,
          final_sample.energy_kx_frac, final_sample.energy_ky_frac,
          final_sample.energy_diag_frac);
    }
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "surface_diffusion_anisotropic: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
