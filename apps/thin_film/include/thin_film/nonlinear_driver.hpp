// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file nonlinear_driver.hpp
 * @brief `MemorySpace`-templated body shared by `thin_film_nonlinear` (CPU)
 *        and `thin_film_nonlinear_hip` (device).
 *
 * @details
 * Dewetting and rupture with the full \f$h^3\f$ lubrication mobility, the
 * science driver for `#114`. `thin_film` (the JSON-session binary) keeps the
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
 * The `MemorySpace`/`Stack` split keeps this file free of any GPU-specific
 * `#include`: `run()` is instantiated with `pfc::HostSpace` +
 * `pfc::sim::stacks::SpectralCPUStack` from `src/thin_film_nonlinear.cpp`,
 * and with `pfc::HIPSpace` + `pfc::sim::stacks::GPUSpectralStack<HIPSpace>`
 * from `src/hip/thin_film_nonlinear.cpp` — the caller's `main()` picks the
 * stack and therefore the includes.
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>
#include <openpfc_apps/field_snapshots.hpp>
#include <openpfc_apps/spectral_flux.hpp>
#include <openpfc_apps/structure_factor.hpp>

#include <thin_film/nonlinear.hpp>
#include <thin_film/thin_film_physics.hpp>

namespace thin_film {

namespace detail {

/// Deterministic broadband perturbation, identical on any decomposition.
inline double hashed_noise(int i, int j, int k, const pfc::Int3 &n,
                           std::uint64_t seed) {
  std::uint64_t x = seed + std::uint64_t(i) +
                    std::uint64_t(n[0]) *
                        (std::uint64_t(j) + std::uint64_t(n[1]) * std::uint64_t(k));
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return 2.0 * (double(x >> 11) / double(1ULL << 53)) - 1.0;
}

inline pfc::sim::PointwiseGeometry geometry_of(const pfc::Domain &domain,
                                               const pfc::Box3i &box) {
  const auto o = pfc::domain::get_origin(domain);
  const auto s = pfc::domain::get_spacing(domain);
  return pfc::sim::PointwiseGeometry{.nx = box.size[0],
                                     .ny = box.size[1],
                                     .nz = box.size[2],
                                     .low_x = box.low[0],
                                     .low_y = box.low[1],
                                     .low_z = box.low[2],
                                     .origin_x = o[0],
                                     .origin_y = o[1],
                                     .origin_z = o[2],
                                     .dx = s[0],
                                     .dy = s[1],
                                     .dz = s[2]};
}

} // namespace detail

/**
 * @brief Parse @p json_path, run the nonlinear dewetting case, print
 *        `THIN_FILM_NONLINEAR ...` on rank 0.
 *
 * @tparam MemorySpace `pfc::HostSpace`, `pfc::CUDASpace`, or `pfc::HIPSpace`.
 * @tparam Stack       `pfc::sim::stacks::SpectralCPUStack` (host) or
 *                     `pfc::sim::stacks::GPUSpectralStack<MemorySpace>`
 *                     (device); must expose `.u()` and `.fft()`.
 */
template <class MemorySpace, class Stack>
int run_thin_film_nonlinear(int rank, int nproc, MPI_Comm comm,
                            const std::string &json_path) {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;
  using RealField = typename Ops::RealField;
  using ComplexField = typename Ops::ComplexField;
  using real_coeffs = typename Ops::real_coeffs;
  using complex_scratch = typename Ops::complex_scratch;
  using json = nlohmann::json;

  int status = 0;
  try {
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("cannot open " + json_path);
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

    Stack stack(domain, rank, nproc, comm);
    auto &h = stack.u();

    // One initial condition: mean film + broadband noise + optional defect.
    // `Field::apply()` writes through `operator()`, which needs direct
    // element access -- unavailable for a device buffer. `with_host_view` is
    // the memory-space-agnostic way to write a field (see
    // `SpectralETDSession::field_checksum` for the same pattern); origin is
    // always (0,0,0) here, so the physical coordinate of local (i,j,k) is
    // exactly `(box.low + (i,j,k)) * dx` -- the same integers the original
    // `lround(x / dx)` round-trip recovered, computed directly instead.
    const auto defect =
        thin_film::GaussianDefect::centred(domain, defect_amp, defect_sigma);
    const auto n = pfc::domain::get_size(domain);
    {
      const auto box = h.box();
      h.with_host_view([&](double *d, std::size_t) {
        for (int k = 0; k < box.size[2]; ++k) {
          for (int j = 0; j < box.size[1]; ++j) {
            for (int i = 0; i < box.size[0]; ++i) {
              const int gi = box.low[0] + i;
              const int gj = box.low[1] + j;
              const int gk = box.low[2] + k;
              const double x0 = static_cast<double>(gi) * dx;
              const double x1 = static_cast<double>(gj) * dx;
              const double xi = detail::hashed_noise(gi, gj, gk, n, seed);
              d[h.idx(i, j, k)] = p.h0 * (1.0 + amp * xi + defect(x0, x1));
            }
          }
        }
      });
    }

    // ETD linear part: the constant-mobility operator about h0. The flux
    // remainder carries everything the linearisation leaves out, so the
    // nonlinear solver reduces to the linear one when M is constant.
    const double gamma = p.gamma, M0 = p.M0, Pip0 = p.Pip0;
    pfc::apps::FluxETD<MemorySpace> stepper(domain, stack.fft(), dt, [=](double k_lap) {
      return -M0 * gamma * k_lap * k_lap - M0 * Pip0 * k_lap;
    });

    const thin_film::CubicMobility mobility{p.M0, p.h0};
    const thin_film::ThinFilmPointwise pw{.A = p.A, .h0 = p.h0,
                                          .h_star = p.h_star, .Pi0 = p.Pi0,
                                          .Pip0 = p.Pip0};
    const thin_film::PotentialPointwise potential_pw{pw};

    // p_hat = -gamma * k_lap * h_hat - FFT(Pi(h))
    RealField pi_real(domain, stack.fft().get_inbox_bounds(), 0);
    ComplexField pi_hat(domain, stack.fft().get_outbox_bounds(), 0);
    const std::size_t n_out = stack.fft().size_outbox();
    std::vector<double> neg_gamma_klap(n_out, 0.0), neg_ones(n_out, -1.0);
    pfc::fft::kspace::for_each_kpoint(
        stack.fft().get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          const double k_lap = -(kx * kx + ky * ky + kz * kz);
          neg_gamma_klap[i] = -gamma * k_lap;
        });
    real_coeffs neg_gamma_klap_dev = Ops::make_real(n_out);
    real_coeffs neg_ones_dev = Ops::make_real(n_out);
    Ops::upload(neg_gamma_klap_dev, std::span<const double>(neg_gamma_klap));
    Ops::upload(neg_ones_dev, std::span<const double>(neg_ones));
    complex_scratch p_scratch = Ops::make_complex(n_out);
    const auto pw_geometry = detail::geometry_of(domain, h.box());

    auto potential = [&](ComplexField &h_hat, RealField &hh, ComplexField &out) {
      // pi_real = Pi(h)
      Ops::pointwise(pw_geometry, 0.0, hh, nullptr, nullptr, pi_real, nullptr,
                     potential_pw);
      Ops::forward(stack.fft(), pi_real, pi_hat);
      // out = (-gamma * k_lap) * h_hat + (-1) * pi_hat
      Ops::combine(h_hat, pi_hat, neg_gamma_klap_dev, neg_ones_dev, p_scratch);
      Ops::swap(out, p_scratch);
    };

    auto snapshots = pfc::apps::make_field_snapshot_writer(cfg, "h", h, comm);
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
                   "step,time,min_h,max_h,mean_h,volume,hole_area_fraction,"
                   "dominant_spacing,ruptured\n");
    }

    double rupture_time = -1.0;
    const int n_steps = int(std::llround(t1 / dt));
    const int every =
        (saveat > 0.0) ? std::max(1, int(std::llround(saveat / dt))) : n_steps;

    auto report = [&](int step, double t) {
      auto s = thin_film::sample_film(h, domain, p.h0, 0.05, comm);
      ComplexField hh(domain, stack.fft().get_outbox_bounds(), 0);
      Ops::forward(stack.fft(), h, hh);
      hh.with_host_view([&](std::complex<double> *hv, std::size_t) {
        const auto sf = pfc::apps::shell_average(stack.fft().get_outbox_bounds(),
                                                 domain, hv, comm, 64);
        s.dominant_spacing = sf.dominant_wavelength();
      });
      if (s.ruptured && rupture_time < 0.0) rupture_time = t;
      pfc::apps::write_field_snapshot(snapshots.get(), snapshot_index++, h);
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
      auto s = thin_film::sample_film(h, domain, p.h0, 0.05, comm);
      std::printf("THIN_FILM_NONLINEAR min_h=%.17g volume=%.17g "
                  "hole_fraction=%.17g rupture_time=%.17g\n",
                  s.min_h, s.volume, s.hole_area_fraction, rupture_time);
    } else {
      (void)thin_film::sample_film(h, domain, p.h0, 0.05, comm);
    }
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "thin_film_nonlinear: " << e.what() << "\n";
    status = 2;
  }
  return status;
}

} // namespace thin_film
