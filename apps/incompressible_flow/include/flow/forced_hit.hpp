// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file forced_hit.hpp
 * @brief Constant-power forcing on the lowest wavevectors.
 *
 * Doering and Petrov, arXiv:physics/0404049 (2004), equations (1)–(5),
 * read 2026-10-01 from the arXiv HTML. The force is proportional to the
 * projection of the velocity onto the smallest wavenumber, and the
 * injected power is the control parameter. On [0, 2π]^3 that shell is
 * the six integer wavevectors with |k| = 1. Their ε is the injection
 * into E = (1/2)⟨u·u⟩, so
 *   f = ε P u / (2 E_band).
 * The zero mode is outside that shell, so the mean velocity is left
 * unchanged. A band energy at or below 1e-24 does not divide: the force
 * stays zero. The same 2/3 mask as the solver is applied, and each mode
 * is passed through the Leray projector. The band energy and the
 * injection are sums over the communicator, so the coefficient uses the
 * global shell. One scheme only. The initial
 * field is the decaying-hit seed.
 *
 * The ladder is fixed before the campaign. Viscosity is 0.02 and the
 * injected power is 0.25. Samples are every 0.5 through t = 20.
 * Equilibration is [0, 10) and the statistics use [10, 20], in five
 * blocks of length 2. Spatial grids are N = 128, 64, 32 at dt = 1/256,
 * finest first. The same N = 64 field is repeated at dt = 1/512 and at
 * dt = 1/128. Speed CFL above 2, or a non-finite field, stops that
 * resolution and keeps the row.
 */

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/fourier_vector.hpp>
#include <openpfc/solvers/incompressible/rotational.hpp>

#include <flow/decaying_hit.hpp>

namespace flow {

inline constexpr double forced_power = 0.25;
inline constexpr double forced_nu = 0.02;
inline constexpr double forced_time = 20.0;
inline constexpr double forced_sample = 0.5;
inline constexpr double forced_equilibration = 10.0;
inline constexpr double forced_block = 2.0;
inline constexpr double forced_dt = 1.0 / 256.0;
inline constexpr double forced_control_dt = 1.0 / 512.0;
inline constexpr double forced_coarse_dt = 1.0 / 128.0;
inline constexpr int forced_temporal_n = 64;
inline constexpr double forced_band_floor = 1.0e-24;
inline constexpr std::array<int, 3> forced_spatial_n{{128, 64, 32}};

struct BandForce {
  double band_ke{0.0};
  double injection{0.0};
  bool applied{false};
};

[[nodiscard]] inline bool lowest_shell(int si, int sj, int sk) noexcept {
  return si * si + sj * sj + sk * sk == 1;
}

[[nodiscard]] inline double
band_kinetic_energy(const pfc::fft::Box3i &outbox, std::array<int, 3> n,
                    std::array<double, 3> spacing, const std::vector<Complex> &u,
                    const std::vector<Complex> &v, const std::vector<Complex> &w) {
  const double ntot = cell_count(n);
  const double scale = 1.0 / (2.0 * ntot * ntot);
  double sum = 0.0;
  pfc::fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double kx, double ky, double kz, int i, int j, int k) {
        const int si = signed_index(i, n[0]);
        const int sj = signed_index(j, n[1]);
        const int sk = signed_index(k, n[2]);
        if (!lowest_shell(si, sj, sk)) return;
        if (!pfc::fft::kspace::two_thirds_keep(kx, ky, kz, spacing)) return;
        const double herm = (i == 0 || i == n[0] / 2) ? 1.0 : 2.0;
        const double mag = std::norm(u[idx]) + std::norm(v[idx]) + std::norm(w[idx]);
        sum += herm * mag * scale;
      });
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sum;
}

[[nodiscard]] inline double
spectral_injection(const pfc::fft::Box3i &outbox, std::array<int, 3> n,
                   std::array<double, 3> spacing, const std::vector<Complex> &u,
                   const std::vector<Complex> &v, const std::vector<Complex> &w,
                   const std::vector<Complex> &fu, const std::vector<Complex> &fv,
                   const std::vector<Complex> &fw) {
  const double ntot = cell_count(n);
  const double scale = 1.0 / (ntot * ntot);
  double sum = 0.0;
  pfc::fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double, double, double, int i, int, int) {
        const double herm = (i == 0 || i == n[0] / 2) ? 1.0 : 2.0;
        const double dot =
            u[idx].real() * fu[idx].real() + u[idx].imag() * fu[idx].imag() +
            v[idx].real() * fv[idx].real() + v[idx].imag() * fv[idx].imag() +
            w[idx].real() * fw[idx].real() + w[idx].imag() * fw[idx].imag();
        sum += herm * dot * scale;
      });
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sum;
}

inline BandForce
write_band_force(const pfc::fft::Box3i &outbox, std::array<int, 3> n,
                 std::array<double, 3> spacing, const std::vector<Complex> &u,
                 const std::vector<Complex> &v, const std::vector<Complex> &w,
                 std::vector<Complex> &fu, std::vector<Complex> &fv,
                 std::vector<Complex> &fw, double power) {
  fu.assign(u.size(), Complex{});
  fv.assign(v.size(), Complex{});
  fw.assign(w.size(), Complex{});
  BandForce info;
  info.band_ke = band_kinetic_energy(outbox, n, spacing, u, v, w);
  if (!(info.band_ke > forced_band_floor) || !(power > 0.0)) return info;
  const double alpha = power / (2.0 * info.band_ke);
  pfc::fft::kspace::for_each_kpoint(
      outbox, n, spacing,
      [&](std::size_t idx, double kx, double ky, double kz, int i, int j, int k) {
        const int si = signed_index(i, n[0]);
        const int sj = signed_index(j, n[1]);
        const int sk = signed_index(k, n[2]);
        if (!lowest_shell(si, sj, sk)) return;
        if (!pfc::fft::kspace::two_thirds_keep(kx, ky, kz, spacing)) return;
        Complex a = alpha * u[idx];
        Complex b = alpha * v[idx];
        Complex c = alpha * w[idx];
        pfc::field::leray_mode(a, b, c, pfc::field::odd_wave(i, j, k, n, spacing));
        fu[idx] = a;
        fv[idx] = b;
        fw[idx] = c;
      });
  info.applied = true;
  info.injection = spectral_injection(outbox, n, spacing, u, v, w, fu, fv, fw);
  return info;
}

inline void step_forced(State &state, double power) {
  auto &fft = state.stack->fft();
  const auto outbox = fft.get_outbox_bounds();
  pfc::incompressible::ifrk4_velocity(
      outbox, state.n, state.spacing, state.u, state.v, state.w, state.exp_dt,
      state.exp_half, state.dt,
      [&](const std::vector<Complex> &u, const std::vector<Complex> &v,
          const std::vector<Complex> &w, std::vector<Complex> &tu,
          std::vector<Complex> &tv, std::vector<Complex> &tw) {
        pfc::incompressible::rotational_tendency(fft, state.n, state.spacing, u, v,
                                                 w, tu, tv, tw, true);
        std::vector<Complex> fu, fv, fw;
        write_band_force(outbox, state.n, state.spacing, u, v, w, fu, fv, fw, power);
        for (std::size_t i = 0; i < tu.size(); ++i) {
          tu[i] += fu[i];
          tv[i] += fv[i];
          tw[i] += fw[i];
        }
      },
      true);
}

/// `which` is `stationary`. Writes the campaign tables under `outdir`.
[[nodiscard]] int run_forced_series(std::string_view which,
                                    const std::string &outdir, int rank);

} // namespace flow
