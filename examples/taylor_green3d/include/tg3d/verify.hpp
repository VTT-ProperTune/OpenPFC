// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file verify.hpp
 * @brief 3-D Taylor–Green on the installed Fourier velocity (issue #229).
 *
 * The cube is [0, 2π]³. The initial velocity is
 * `u = sin(x) cos(y) cos(z)`, `v = -cos(x) sin(y) cos(z)`, `w = 0`.
 * At t = 0 the cell averages are kinetic energy 1/8 and enstrophy 3/4.
 * Viscosity of a single shell with |k|² = 3 would decay that energy as
 * `exp(-6 ν t)`. The nonlinear term moves energy onto other shells, so
 * the exponential is an early-time comparison, not a later oracle.
 * `dE/dt = -ν ⟨|ω|²⟩` still holds for periodic incompressible flow.
 *
 * The step is the installed integrating-factor RK4 of `P(u × ω)` after a
 * state-level 2/3 mask. This file does not own a second projector.
 * A 3/2 padded product is not used. One MPI rank owns every mode.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/power_spectrum.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/fourier_vector.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/solvers/incompressible/rotational.hpp>

namespace tg3d {

using Complex = pfc::field::Complex;

inline constexpr double kinetic_energy_0 = 0.125;
inline constexpr double enstrophy_0 = 0.75;

/// Viscous decay of the initial |k|² = 3 shell. Not a later oracle.
[[nodiscard]] inline double linear_shell_energy(double nu, double time) noexcept {
  return kinetic_energy_0 * std::exp(-6.0 * nu * time);
}

using pfc::fft::kspace::signed_wave_index;

struct ShellEnergy {
  int k2{0};
  double ke{0.0};
};

struct Diagnostics {
  double ke{0.0};
  double enstrophy{0.0};
  double dissipation{0.0};
  double max_vorticity{0.0};
  double div_l2{0.0};
  double div_linf{0.0};
  double modal_div_max{0.0};
  double cfl{0.0};
  double mean_u{0.0};
  double mean_v{0.0};
  double mean_w{0.0};
  double w_l2{0.0};
  double max_abs_w{0.0};
  double outer_ke_fraction{0.0};
  std::vector<ShellEnergy> spectrum;
};

struct FieldError {
  double velocity_l2{0.0};
  double velocity_l2_rel{0.0};
  double vorticity_l2{0.0};
  double vorticity_l2_rel{0.0};
  double vorticity_linf{0.0};
};

struct Hats {
  std::array<int, 3> n{};
  std::array<double, 3> spacing{};
  pfc::fft::Box3i outbox{};
  std::vector<Complex> u;
  std::vector<Complex> v;
  std::vector<Complex> w;
};

struct Flow {
  std::unique_ptr<pfc::sim::stacks::SpectralCPUStack> stack;
  std::array<int, 3> n{};
  std::array<double, 3> spacing{};
  double nu{0.0};
  double dt{0.0};
  std::vector<Complex> u;
  std::vector<Complex> v;
  std::vector<Complex> w;
  std::vector<double> exp_dt;
  std::vector<double> exp_half;

  Flow() = default;
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;
};

[[nodiscard]] inline double cell_count(std::array<int, 3> n) noexcept {
  return static_cast<double>(n[0]) * static_cast<double>(n[1]) *
         static_cast<double>(n[2]);
}

[[nodiscard]] inline Flow make_flow(int n, double nu, double dt, int rank,
                                    int nproc) {
  if (n < 8 || n % 2 != 0) {
    throw std::invalid_argument("taylor_green3d: N must be even and at least 8");
  }
  if (nu < 0.0 || dt <= 0.0) {
    throw std::invalid_argument("taylor_green3d: nu and dt must be positive");
  }
  const double h = pfc::two_pi / static_cast<double>(n);
  auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({h, h, h}));
  Flow flow;
  flow.stack = std::make_unique<pfc::sim::stacks::SpectralCPUStack>(
      std::move(domain), rank, nproc, MPI_COMM_WORLD);
  flow.n = {n, n, n};
  flow.spacing = {h, h, h};
  flow.nu = nu;
  flow.dt = dt;
  auto &fft = flow.stack->fft();
  const auto ncells = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) *
                      static_cast<std::size_t>(n);
  if (fft.size_inbox() != ncells) {
    throw std::runtime_error("taylor_green3d: one rank must own the full real grid");
  }
  const auto nhat = fft.size_outbox();
  flow.u.assign(nhat, Complex{});
  flow.v.assign(nhat, Complex{});
  flow.w.assign(nhat, Complex{});
  pfc::incompressible::viscous_exponentials(fft.get_outbox_bounds(), flow.n,
                                            flow.spacing, nu, dt, flow.exp_dt,
                                            flow.exp_half);
  return flow;
}

inline void initialize_taylor_green(Flow &flow) {
  auto &fft = flow.stack->fft();
  const auto inbox = fft.get_inbox_bounds();
  const auto domain = pfc::domain::create(pfc::GridSize(flow.n),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing(flow.spacing));
  pfc::Field<double> rx(domain, inbox, 0), ry(domain, inbox, 0),
      rz(domain, inbox, 0);
  // Keep model formulas with this independent app/example owner; Field owns
  // the reusable indexing rather than introducing a model API in the SDK.
  rx.apply([](double x, double y, double z) {
    return std::sin(x) * std::cos(y) * std::cos(z);
  });
  ry.apply([](double x, double y, double z) {
    return -std::cos(x) * std::sin(y) * std::cos(z);
  });
  fft.forward(rx.vec(), flow.u);
  fft.forward(ry.vec(), flow.v);
  fft.forward(rz.vec(), flow.w);
  const auto outbox = fft.get_outbox_bounds();
  const auto nhat = flow.u.size();
  pfc::field::leray_project(outbox, flow.n, flow.spacing, flow.u.data(),
                            flow.v.data(), flow.w.data(), nhat);
  pfc::field::apply_two_thirds(outbox, flow.n, flow.spacing, flow.u.data(),
                               flow.v.data(), flow.w.data(), nhat);
}

inline void step(Flow &flow) {
  auto &fft = flow.stack->fft();
  const auto outbox = fft.get_outbox_bounds();
  pfc::incompressible::ifrk4_velocity(
      outbox, flow.n, flow.spacing, flow.u, flow.v, flow.w, flow.exp_dt,
      flow.exp_half, flow.dt,
      [&](const std::vector<Complex> &u, const std::vector<Complex> &v,
          const std::vector<Complex> &w, std::vector<Complex> &tu,
          std::vector<Complex> &tv, std::vector<Complex> &tw) {
        pfc::incompressible::rotational_tendency(fft, flow.n, flow.spacing, u, v, w,
                                                 tu, tv, tw, true);
      },
      true);
}

[[nodiscard]] inline bool state_finite(const Flow &flow) {
  auto ok = [](const std::vector<Complex> &hat) {
    for (const auto &z : hat) {
      if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) return false;
    }
    return true;
  };
  return ok(flow.u) && ok(flow.v) && ok(flow.w);
}

[[nodiscard]] inline Hats copy_hats(const Flow &flow) {
  Hats hats;
  hats.n = flow.n;
  hats.spacing = flow.spacing;
  hats.outbox = flow.stack->fft().get_outbox_bounds();
  hats.u = flow.u;
  hats.v = flow.v;
  hats.w = flow.w;
  return hats;
}

namespace detail {

[[nodiscard]] inline unsigned long long pack_wave(int ki, int kj, int kk) noexcept {
  constexpr unsigned long long mask = (1ull << 21) - 1ull;
  return ((static_cast<unsigned long long>(static_cast<std::uint32_t>(ki)) & mask)
          << 42) |
         ((static_cast<unsigned long long>(static_cast<std::uint32_t>(kj)) & mask)
          << 21) |
         (static_cast<unsigned long long>(static_cast<std::uint32_t>(kk)) & mask);
}

inline void inverse_velocity(Flow &flow, const std::vector<Complex> &u,
                             const std::vector<Complex> &v,
                             const std::vector<Complex> &w, std::vector<double> &rx,
                             std::vector<double> &ry, std::vector<double> &rz) {
  auto &fft = flow.stack->fft();
  rx.resize(fft.size_inbox());
  ry.resize(fft.size_inbox());
  rz.resize(fft.size_inbox());
  fft.backward(u, rx);
  fft.backward(v, ry);
  fft.backward(w, rz);
}

[[nodiscard]] inline std::vector<ShellEnergy> energy_spectrum(const Flow &flow) {
  const double ntot = cell_count(flow.n);
  const double scale = 1.0 / (2.0 * ntot * ntot);
  std::unordered_map<int, double> bins;
  const auto outbox = flow.stack->fft().get_outbox_bounds();
  pfc::fft::kspace::for_each_kpoint(
      outbox, flow.n, flow.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const int ki = signed_wave_index(i, flow.n[0]);
        const int kj = signed_wave_index(j, flow.n[1]);
        const int kk = signed_wave_index(k, flow.n[2]);
        const int k2 = ki * ki + kj * kj + kk * kk;
        const double herm = pfc::fft::r2c_multiplicity(i, flow.n[0]);
        const double mag =
            std::norm(flow.u[idx]) + std::norm(flow.v[idx]) + std::norm(flow.w[idx]);
        bins[k2] += herm * mag * scale;
      });
  std::vector<int> keys;
  keys.reserve(bins.size());
  for (const auto &bin : bins) keys.push_back(bin.first);
  std::sort(keys.begin(), keys.end());
  std::vector<ShellEnergy> shells;
  shells.reserve(keys.size());
  for (int k2 : keys) {
    if (bins[k2] == 0.0) continue;
    shells.push_back(ShellEnergy{k2, bins[k2]});
  }
  return shells;
}

} // namespace detail

[[nodiscard]] inline Diagnostics diagnose(Flow &flow) {
  auto &fft = flow.stack->fft();
  const auto outbox = fft.get_outbox_bounds();
  const auto inbox = fft.get_inbox_bounds();
  const auto nhat = flow.u.size();
  const double ncells = cell_count(flow.n);

  std::vector<double> rx, ry, rz;
  detail::inverse_velocity(flow, flow.u, flow.v, flow.w, rx, ry, rz);

  std::vector<Complex> ox(nhat), oy(nhat), oz(nhat);
  pfc::field::curl_hat(outbox, flow.n, flow.spacing, flow.u.data(), flow.v.data(),
                       flow.w.data(), ox.data(), oy.data(), oz.data(), nhat);
  std::vector<double> oxr, oyr, ozr;
  detail::inverse_velocity(flow, ox, oy, oz, oxr, oyr, ozr);

  std::vector<Complex> div_hat(nhat);
  pfc::fft::kspace::for_each_kpoint(
      outbox, flow.n, flow.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        div_hat[idx] = pfc::field::modal_divergence(
            flow.u[idx], flow.v[idx], flow.w[idx],
            pfc::field::odd_wave(i, j, k, flow.n, flow.spacing));
      });
  std::vector<double> div_real(fft.size_inbox());
  fft.backward(div_hat, div_real);

  Diagnostics diag;
  double ke_sum = 0.0;
  double enstrophy_sum = 0.0;
  double w2 = 0.0;
  double su = 0.0;
  double sv = 0.0;
  double sw = 0.0;
  double peak = 0.0;
  double max_w = 0.0;
  double max_omega = 0.0;
  double div2 = 0.0;
  double div_linf = 0.0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const auto idx = static_cast<std::size_t>(inbox.to_linear({i, j, k}));
        const double u = rx[idx];
        const double v = ry[idx];
        const double w = rz[idx];
        su += u;
        sv += v;
        sw += w;
        ke_sum += u * u + v * v + w * w;
        w2 += w * w;
        peak = std::max(peak,
                        std::max(std::abs(u), std::max(std::abs(v), std::abs(w))));
        max_w = std::max(max_w, std::abs(w));
        const double oxv = oxr[idx];
        const double oyv = oyr[idx];
        const double ozv = ozr[idx];
        enstrophy_sum += oxv * oxv + oyv * oyv + ozv * ozv;
        max_omega = std::max(max_omega, std::hypot(oxv, std::hypot(oyv, ozv)));
        div2 += div_real[idx] * div_real[idx];
        div_linf = std::max(div_linf, std::abs(div_real[idx]));
      }
    }
  }
  diag.ke = 0.5 * ke_sum / ncells;
  diag.enstrophy = enstrophy_sum / ncells;
  diag.dissipation = flow.nu * diag.enstrophy;
  diag.max_vorticity = max_omega;
  diag.div_l2 = std::sqrt(div2 / ncells);
  diag.div_linf = div_linf;
  diag.modal_div_max =
      pfc::field::max_modal_divergence(outbox, flow.n, flow.spacing, flow.u.data(),
                                       flow.v.data(), flow.w.data(), nhat);
  diag.cfl = flow.dt * peak / flow.spacing[0];
  diag.mean_u = su / ncells;
  diag.mean_v = sv / ncells;
  diag.mean_w = sw / ncells;
  diag.w_l2 = std::sqrt(w2 / ncells);
  diag.max_abs_w = max_w;

  const int cut = flow.n[0] / 6;
  std::vector<Complex> uh = flow.u;
  std::vector<Complex> vh = flow.v;
  std::vector<Complex> wh = flow.w;
  pfc::fft::kspace::for_each_kpoint(
      outbox, flow.n, flow.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const int ki = std::abs(signed_wave_index(i, flow.n[0]));
        const int kj = std::abs(signed_wave_index(j, flow.n[1]));
        const int kk = std::abs(signed_wave_index(k, flow.n[2]));
        if (std::max(ki, std::max(kj, kk)) <= cut) {
          uh[idx] = vh[idx] = wh[idx] = Complex{};
        }
      });
  detail::inverse_velocity(flow, uh, vh, wh, rx, ry, rz);
  double outer_sum = 0.0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const auto idx = static_cast<std::size_t>(inbox.to_linear({i, j, k}));
        outer_sum += rx[idx] * rx[idx] + ry[idx] * ry[idx] + rz[idx] * rz[idx];
      }
    }
  }
  const double outer_ke = 0.5 * outer_sum / ncells;
  diag.outer_ke_fraction = (diag.ke > 0.0) ? outer_ke / diag.ke : 0.0;
  diag.spectrum = detail::energy_spectrum(flow);
  return diag;
}

[[nodiscard]] inline FieldError field_error(Flow &coarse, const Hats &fine) {
  const auto &fft = coarse.stack->fft();
  const auto coarse_box = fft.get_outbox_bounds();
  const auto nhat = coarse.u.size();
  const double scale = cell_count(coarse.n) / cell_count(fine.n);

  std::unordered_map<unsigned long long, std::size_t> fmap;
  fmap.reserve(fine.u.size());
  pfc::fft::kspace::for_each_kpoint(
      fine.outbox, fine.n, fine.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        fmap[detail::pack_wave(signed_wave_index(i, fine.n[0]),
                               signed_wave_index(j, fine.n[1]),
                               signed_wave_index(k, fine.n[2]))] = idx;
      });

  std::vector<Complex> ru(nhat), rv(nhat), rw(nhat);
  pfc::fft::kspace::for_each_kpoint(
      coarse_box, coarse.n, coarse.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const auto it = fmap.find(detail::pack_wave(
            signed_wave_index(i, coarse.n[0]), signed_wave_index(j, coarse.n[1]),
            signed_wave_index(k, coarse.n[2])));
        if (it == fmap.end()) return;
        ru[idx] = fine.u[it->second] * scale;
        rv[idx] = fine.v[it->second] * scale;
        rw[idx] = fine.w[it->second] * scale;
      });

  std::vector<double> crx, cry, crz, rrx, rry, rrz;
  detail::inverse_velocity(coarse, coarse.u, coarse.v, coarse.w, crx, cry, crz);
  detail::inverse_velocity(coarse, ru, rv, rw, rrx, rry, rrz);

  std::vector<Complex> cox(nhat), coy(nhat), coz(nhat), fox(nhat), foy(nhat),
      foz(nhat);
  pfc::field::curl_hat(coarse_box, coarse.n, coarse.spacing, coarse.u.data(),
                       coarse.v.data(), coarse.w.data(), cox.data(), coy.data(),
                       coz.data(), nhat);
  pfc::field::curl_hat(coarse_box, coarse.n, coarse.spacing, ru.data(), rv.data(),
                       rw.data(), fox.data(), foy.data(), foz.data(), nhat);
  std::vector<double> coxr, coyr, cozr, foxr, foyr, fozr;
  detail::inverse_velocity(coarse, cox, coy, coz, coxr, coyr, cozr);
  detail::inverse_velocity(coarse, fox, foy, foz, foxr, foyr, fozr);

  double ve2 = 0.0;
  double vr2 = 0.0;
  double we2 = 0.0;
  double wr2 = 0.0;
  double wlinf = 0.0;
  const auto inbox = fft.get_inbox_bounds();
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const auto idx = static_cast<std::size_t>(inbox.to_linear({i, j, k}));
        const double du = crx[idx] - rrx[idx];
        const double dv = cry[idx] - rry[idx];
        const double dw = crz[idx] - rrz[idx];
        ve2 += du * du + dv * dv + dw * dw;
        vr2 += rrx[idx] * rrx[idx] + rry[idx] * rry[idx] + rrz[idx] * rrz[idx];
        const double dox = coxr[idx] - foxr[idx];
        const double doy = coyr[idx] - foyr[idx];
        const double doz = cozr[idx] - fozr[idx];
        we2 += dox * dox + doy * doy + doz * doz;
        wr2 += foxr[idx] * foxr[idx] + foyr[idx] * foyr[idx] + fozr[idx] * fozr[idx];
        wlinf = std::max(wlinf, std::hypot(dox, std::hypot(doy, doz)));
      }
    }
  }
  const double ncells = cell_count(coarse.n);
  FieldError err;
  err.velocity_l2 = std::sqrt(ve2 / ncells);
  const double vref = std::sqrt(vr2 / ncells);
  err.velocity_l2_rel = (vref > 0.0) ? err.velocity_l2 / vref : 0.0;
  err.vorticity_l2 = std::sqrt(we2 / ncells);
  const double wref = std::sqrt(wr2 / ncells);
  err.vorticity_l2_rel = (wref > 0.0) ? err.vorticity_l2 / wref : 0.0;
  err.vorticity_linf = wlinf;
  return err;
}

} // namespace tg3d
