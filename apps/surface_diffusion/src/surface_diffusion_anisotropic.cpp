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
 *   \partial_t h = -\nabla\cdot\bigl[B(\theta)\,\nabla(\nabla^2 h)\bigr],
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/io/diagnostics_series.hpp>
#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/frontend/ui/json_snapshot_fields.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/fft/power_spectrum.hpp>
#include <openpfc/kernel/simulation/observable_reduce.hpp>
#include <openpfc/kernel/simulation/simulation_lifecycle.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

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
  const auto height = pfc::sim::reduce_owned(h, comm);
  Sample s;
  s.mean_h =
      height.count == 0 ? 0.0 : height.sum / static_cast<double>(height.count);
  // RMS roughness is the population standard deviation, not the raw RMS.
  s.rms_roughness = std::sqrt(height.variance);

  pfc::data::Field<std::complex<double>> h_hat(domain, fft.get_outbox_bounds(), 0);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(fft, h, h_hat);

  pfc::fft::RadialSpectrum spectrum;
  pfc::fft::DirectionalPower axes;
  h_hat.with_host_view([&](const std::complex<double> *hv, std::size_t) {
    spectrum =
        pfc::fft::radial_average(fft.get_outbox_bounds(), domain, hv, comm, 64);
    axes = pfc::fft::directional_power(fft.get_outbox_bounds(), domain, hv, comm);
  });
  s.dominant_wavelength = spectrum.dominant_wavelength();
  s.domain_length = spectrum.mean_wavelength();

  const double total = axes.along_x + axes.along_y + axes.unresolved;
  if (total > 0.0) {
    s.energy_kx_frac = axes.along_x / total;
    s.energy_ky_frac = axes.along_y / total;
    s.energy_diag_frac = axes.unresolved / total;
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

    const auto domain = pfc::ui::from_json<pfc::Domain>(cfg);
    const int Lx = pfc::domain::get_size(domain, 0),
              Ly = pfc::domain::get_size(domain, 1),
              Lz = pfc::domain::get_size(domain, 2);
    const double dx = pfc::domain::get_spacing(domain, 0);
    // One spacing for x and y: the driver reads dx only.
    if (pfc::domain::get_spacing(domain, 1) != dx)
      throw std::invalid_argument("surface_diffusion_anisotropic: x and y spacings "
                                  "must match, (upper - lower) / size");
    if (Lz != 1)
      throw std::invalid_argument(
          "surface_diffusion_anisotropic: requires a 2-D domain (Lz == 1)");

    surface_diffusion::AnisotropyParams params;
    surface_diffusion::apply_anisotropy_json(cfg.at("model").at("params"), params);

    const auto &ts = cfg.at("timestepping");
    const double t1 = ts.at("t1"), dt = ts.at("dt"),
                 saveat = ts.value("saveat", -1.0);

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

    // Optional `.vti` snapshots of the surface itself (`fields[]`). The
    // diagnostics CSV carries the orientation split as a number; the
    // snapshots are what let a reader see which ridge set the anisotropic
    // anneal is erasing.
    pfc::io::SnapshotSeries snapshots(
        h.domain(), h.box(), pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_WORLD});
    pfc::ui::bind_snapshot_field(snapshots, cfg, "h", h);
    pfc::ui::finish_snapshot_fields(snapshots, cfg);

    std::optional<pfc::io::DiagnosticsSeries> diag;
    if (cfg.contains("diagnostics")) {
      diag.emplace(std::vector<std::string>{"mean_h", "rms_roughness", "max_grad_h",
                                            "dominant_wavelength", "domain_length",
                                            "energy_kx_frac", "energy_ky_frac",
                                            "energy_diag_frac"},
                   pfc::io::DiagnosticsSeriesOptions{
                       .path = cfg.at("diagnostics").at("csv").get<std::string>(),
                       .comm = MPI_COMM_WORLD,
                       .overwrite = true});
    }

    Sample final_sample{};
    pfc::sim::SimulationLifecycle life(
        pfc::sim::SimulationLifecycle::schedule(0.0, t1, dt, saveat),
        MPI_COMM_WORLD);
    life.bind_field("h", h);
    auto report = [&](const pfc::Time &now) {
      const int step = pfc::time::increment(now);
      const double t = pfc::time::current(now);
      final_sample = sample_surface(h, domain, stack.fft(), MPI_COMM_WORLD);
      snapshots.write(step, t);
      if (diag) {
        diag->write(step, t,
                    {final_sample.mean_h, final_sample.rms_roughness,
                     final_sample.max_grad_h, final_sample.dominant_wavelength,
                     final_sample.domain_length, final_sample.energy_kx_frac,
                     final_sample.energy_ky_frac, final_sample.energy_diag_frac});
      }
    };
    const double span = t1;
    const double n_fixed = std::round(span / dt);
    if (!(dt > 0.0) || std::abs(span - n_fixed * dt) > 1e-8 * std::max(1.0, span)) {
      throw std::invalid_argument(
          "surface_diffusion_anisotropic: t1 must be an integer multiple of dt");
    }
    life.set_save_observer(report);
    life.run([&](double t_now, double interval) {
      if (std::abs(interval - dt) > 1e-8 * std::max(1.0, dt)) {
        throw std::invalid_argument(
            "surface_diffusion_anisotropic: refusing a clipped step; the "
            "ETD coefficients use the full dt");
      }
      stepper.step(t_now - interval, h);
    });

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
    if (diag) diag->close();
    snapshots.close();
  } catch (const std::exception &e) {
    if (rank == 0)
      std::cerr << "surface_diffusion_anisotropic: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
