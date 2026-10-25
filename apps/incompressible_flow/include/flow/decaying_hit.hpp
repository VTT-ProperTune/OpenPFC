// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file decaying_hit.hpp
 * @brief One seeded realization of decaying homogeneous isotropic turbulence.
 *
 * The target density is Yoffe & McComb, arXiv:1805.01238 (2018), equation
 * (10), read 2026-10-01 from the arXiv HTML:
 *   E(k) = 0.266 (k / 3.536)^4 exp(-(k / 3.536)^2).
 * Their runs used a Gaussian ensemble. This case scales one realization
 * onto that density. It does not reproduce their ensemble or their
 * Taylor-Reynolds numbers. Viscosity 0.02 is the value in their Table 2
 * for the 128^3 Gaussian runs. The box is [0, 2π]^3, so an integer wave
 * index is the physical wavenumber and their k_min = 1.
 *
 * Each retained mode gets the isotropic share E(|k|) / (4 π |k|^2).
 * Stored as an r2c coefficient this is
 *   |û|^2 + |v̂|^2 + |ŵ|^2 = N^6 E(|k|) / (2 π |k|^2).
 * The 2/3 mask is the solver's mask. A projected draw whose squared
 * amplitude is below 1e-24 is left at zero. The mean mode is zero.
 * Amplitudes are hashed from the signed wave index and the seed, so the
 * same mode does not depend on the buffer order or on N.
 *
 * The ladder below is the verification protocol. It is fixed before the
 * campaign. Samples are t = 0, 0.5, 1. Spatial grids are N = 128, 64, 32
 * at dt = 1/256, finest first, plus one N = 64 control at dt = 1/512.
 * Temporal steps at N = 64 are dt = 1/256, 1/128, 1/64, 1/32, finest
 * first. The reference is the finest member. An order is
 * log2(e(dt) / e(dt/2)) only when both absolute errors exceed 1e-11 and
 * the finer step is not the reference. Speed CFL above 2 stops that
 * resolution. Failed rows stay in the table.
 */

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <openpfc/kernel/fft/power_spectrum.hpp>
#include <string>
#include <string_view>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/fourier_vector.hpp>

#include <flow/taylor_green.hpp>

namespace flow {

inline constexpr double spectrum_c = 0.266;
inline constexpr double spectrum_k0 = 3.536;
inline constexpr double degenerate_draw = 1.0e-24;
inline constexpr std::uint64_t protocol_seed = 1;
inline constexpr double protocol_nu = 0.02;
inline constexpr double protocol_time = 1.0;
inline constexpr double protocol_dt = 1.0 / 256.0;
inline constexpr double protocol_control_dt = 1.0 / 512.0;
inline constexpr int protocol_temporal_n = 64;
inline constexpr double protocol_order_floor = 1.0e-11;
inline constexpr std::array<int, 3> protocol_spatial_n{{128, 64, 32}};
inline constexpr std::array<double, 4> protocol_temporal_dt{
    {1.0 / 256.0, 1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0}};

[[nodiscard]] inline double energy_density(double k) noexcept {
  if (!(k > 0.0)) return 0.0;
  const double x = k / spectrum_k0;
  const double x2 = x * x;
  return spectrum_c * x2 * x2 * std::exp(-x2);
}

[[nodiscard]] inline int signed_index(int idx, int n) noexcept {
  return pfc::fft::kspace::signed_wave_index(idx, n);
}

[[nodiscard]] inline int retained_k_max(int n) noexcept {
  return static_cast<int>(std::floor((static_cast<double>(n) / 3.0) - 1.0e-12));
}

struct Shell {
  int index{0};
  double ke{0.0};
};

struct Scales {
  double u_rms{0.0};
  double omega_rms{0.0};
  double lambda{0.0};
  double re_lambda{0.0};
  double integral_scale{0.0};
  double eta{0.0};
  int k_max{0};
  double k_max_eta{0.0};
  double modal_ke{0.0};
  double outer_ke_fraction{0.0};
};

namespace detail {

[[nodiscard]] inline std::uint64_t mix64(std::uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

[[nodiscard]] inline double unit_interval(std::uint64_t z) noexcept {
  const double value = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
  return value == 0.0 ? 1.0 : value;
}

[[nodiscard]] inline double normal_draw(std::uint64_t seed, int si, int sj, int sk,
                                        int stream) noexcept {
  std::uint64_t key = mix64(seed ^ 0x9E3779B97F4A7C15ull);
  key = mix64(key ^ static_cast<std::uint32_t>(si));
  key = mix64(key ^ static_cast<std::uint32_t>(sj));
  key = mix64(key ^ static_cast<std::uint32_t>(sk));
  key = mix64(key ^ static_cast<std::uint32_t>(stream));
  const double a = unit_interval(mix64(key ^ 0xD1B54A32D192ED03ull));
  const double b = unit_interval(mix64(key ^ 0x94D049BB133111EBull));
  return std::sqrt(-2.0 * std::log(a)) * std::cos(pfc::two_pi * b);
}

} // namespace detail

inline void initialize_decaying_hit(State &state, std::uint64_t seed) {
  const auto outbox = state.stack->fft().get_outbox_bounds();
  const double ntot = cell_count(state.n);
  const int n = state.n[0];
  const int nyquist = n / 2;
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double kx, double ky, double kz, int i, int j, int k) {
        const int si = signed_index(i, n);
        const int sj = signed_index(j, n);
        const int sk = signed_index(k, n);
        state.u[idx] = state.v[idx] = state.w[idx] = Complex{};
        if (si == 0 && sj == 0 && sk == 0) return;
        if (!pfc::fft::kspace::two_thirds_keep(kx, ky, kz, state.spacing)) return;

        const bool plane = (si == 0 || si == nyquist);
        int csi = si;
        int csj = sj;
        int csk = sk;
        bool conjugated = false;
        bool real_mode = false;
        if (plane) {
          if (sj < 0 || (sj == 0 && sk < 0)) {
            csj = -sj;
            csk = -sk;
            conjugated = true;
          }
          real_mode = (csj == 0 && csk == 0);
        }

        Complex g[3];
        for (int c = 0; c < 3; ++c) {
          const double re = detail::normal_draw(seed, csi, csj, csk, 2 * c);
          const double im =
              real_mode ? 0.0 : detail::normal_draw(seed, csi, csj, csk, 2 * c + 1);
          g[c] = conjugated ? Complex(re, -im) : Complex(re, im);
        }
        pfc::field::leray_mode(
            g[0], g[1], g[2], pfc::field::odd_wave(i, j, k, state.n, state.spacing));
        const double g2 = std::norm(g[0]) + std::norm(g[1]) + std::norm(g[2]);
        if (!(g2 > degenerate_draw)) return;
        const double kmag =
            std::sqrt(static_cast<double>(si * si + sj * sj + sk * sk));
        const double target =
            ntot * ntot * energy_density(kmag) / (pfc::two_pi * kmag * kmag);
        const double scale = std::sqrt(target / g2);
        state.u[idx] = g[0] * scale;
        state.v[idx] = g[1] * scale;
        state.w[idx] = g[2] * scale;
      });
}

[[nodiscard]] inline std::vector<Shell> shell_energies(const State &state) {
  const double ntot = cell_count(state.n);
  const double scale = 1.0 / (2.0 * ntot * ntot);
  const int n = state.n[0];
  std::vector<double> bins(static_cast<std::size_t>(n + 1), 0.0);
  const auto outbox = state.stack->fft().get_outbox_bounds();
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const int si = signed_index(i, n);
        const int sj = signed_index(j, n);
        const int sk = signed_index(k, n);
        const double kmag =
            std::sqrt(static_cast<double>(si * si + sj * sj + sk * sk));
        const int shell = static_cast<int>(std::floor(kmag + 0.5));
        if (shell <= 0 || shell >= static_cast<int>(bins.size())) return;
        const double herm = pfc::fft::r2c_multiplicity(i, n);
        const double mag = std::norm(state.u[idx]) + std::norm(state.v[idx]) +
                           std::norm(state.w[idx]);
        bins[static_cast<std::size_t>(shell)] += herm * mag * scale;
      });
  MPI_Allreduce(MPI_IN_PLACE, bins.data(), static_cast<int>(bins.size()), MPI_DOUBLE,
                MPI_SUM, state.stack->mpi_comm());
  std::vector<Shell> shells;
  for (int s = 1; s < static_cast<int>(bins.size()); ++s) {
    if (bins[static_cast<std::size_t>(s)] == 0.0) continue;
    shells.push_back(Shell{s, bins[static_cast<std::size_t>(s)]});
  }
  return shells;
}

[[nodiscard]] inline double modal_kinetic_energy(const State &state) {
  double sum = 0.0;
  for (const auto &shell : shell_energies(state)) sum += shell.ke;
  return sum;
}

[[nodiscard]] inline Scales measure_scales(const State &state,
                                           const Diagnostics &diag) {
  Scales scales;
  scales.k_max = retained_k_max(state.n[0]);
  scales.omega_rms = std::sqrt(std::max(diag.enstrophy, 0.0));
  scales.u_rms = std::sqrt(std::max(2.0 * diag.ke / 3.0, 0.0));
  if (diag.dissipation > 0.0) {
    scales.lambda = scales.u_rms * std::sqrt(15.0 * state.nu / diag.dissipation);
    scales.re_lambda = scales.u_rms * scales.lambda / state.nu;
    scales.eta = std::pow(state.nu * state.nu * state.nu / diag.dissipation, 0.25);
    scales.k_max_eta = static_cast<double>(scales.k_max) * scales.eta;
  }
  const auto shells = shell_energies(state);
  double moment = 0.0;
  for (const auto &shell : shells) {
    scales.modal_ke += shell.ke;
    moment += shell.ke / static_cast<double>(shell.index);
  }
  if (scales.modal_ke > 0.0) {
    scales.integral_scale = (3.0 * pfc::pi / (4.0 * scales.modal_ke)) * moment;
  }

  const double ntot = cell_count(state.n);
  const double weight = 1.0 / (2.0 * ntot * ntot);
  const int n = state.n[0];
  const int cut = n / 6;
  double outer = 0.0;
  const auto outbox = state.stack->fft().get_outbox_bounds();
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        const int si = std::abs(signed_index(i, n));
        const int sj = std::abs(signed_index(j, n));
        const int sk = std::abs(signed_index(k, n));
        if (std::max(si, std::max(sj, sk)) <= cut) return;
        const double herm = pfc::fft::r2c_multiplicity(i, n);
        const double mag = std::norm(state.u[idx]) + std::norm(state.v[idx]) +
                           std::norm(state.w[idx]);
        outer += herm * mag * weight;
      });
  MPI_Allreduce(MPI_IN_PLACE, &outer, 1, MPI_DOUBLE, MPI_SUM,
                state.stack->mpi_comm());
  scales.outer_ke_fraction = (scales.modal_ke > 0.0) ? outer / scales.modal_ke : 0.0;
  return scales;
}

/// `which` is `spatial` or `temporal`. Writes the campaign tables under
/// `outdir` and returns 0 when every resolution finishes.
[[nodiscard]] int run_series(std::string_view which, const std::string &outdir,
                             int rank);

} // namespace flow
