// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_resolution.cpp
 * @brief The grid criterion behind the dealias question, pinned.
 *
 * `tungsten_dealias_study` takes minutes and is not run by CI. The arithmetic
 * it is built on is instant, so that is checked here, and so is the short
 * gate that selected `tungsten_resolved_bcc.json`.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/power_spectrum.hpp>
#include <tungsten/resolution.hpp>
#include <tungsten/tungsten_pointwise.hpp>
#include <tungsten/tungsten_session.hpp>

using Catch::Matchers::WithinRel;
namespace res = tungsten::resolution;

TEST_CASE("Cubic tungsten nonlinear RHS aliases into a retained low mode",
          "[tungsten][resolution][unit]") {
  // A controlled probe of the *shipped pointwise polynomial*, not an
  // inference from the final solution's high-frequency tail.
  // cos(5x)^3 = (3 cos(5x) + cos(15x))/4. On 16 samples the 15th
  // harmonic folds into +/-1, retained by the standard 2/3 output mask.
  // On 64 samples it is distinguishable from mode 1.
  tungsten::TungstenPointwise cubic{};
  cubic.c_psi3 = 1.0;
  const auto nonlinear_mode = [&](int n, int mode) {
    std::complex<double> coefficient{};
    for (int j = 0; j < n; ++j) {
      const double x = 2.0 * std::numbers::pi * j / n;
      const double psi = std::cos(5.0 * x);
      const double rhs = cubic.nonlinearity(pfc::sim::SpectralCell{.psi = psi});
      const std::complex<double> projector{std::cos(mode * x), -std::sin(mode * x)};
      coefficient += rhs * projector;
    }
    return coefficient / static_cast<double>(n);
  };
  const auto alias = nonlinear_mode(16, 1);
  const auto reference = nonlinear_mode(64, 1);
  REQUIRE(std::abs(alias.real() - 0.125) < 1e-12);
  REQUIRE(std::abs(alias.imag()) < 1e-12);
  REQUIRE(std::abs(reference) < 1e-12);
  REQUIRE(std::abs(nonlinear_mode(64, 15).real() - 0.125) < 1e-12);

  // A retained-mode check makes the rejection of 2/3 as an exact
  // cubic-dealias rule explicit, without changing any production mask.
  REQUIRE(1 * 3 < 16);
}

TEST_CASE("Tungsten resolution: the two conditions meet at pi/3",
          "[tungsten][resolution][unit]") {
  // Both conditions -- third harmonic under Nyquist, and the 2/3 cut above the
  // second harmonic -- reduce to the same spacing. That coincidence is the
  // whole reason a single number characterises the grid.
  const double safe = res::dealias_safe_dx();
  REQUIRE_THAT(safe, WithinRel(std::numbers::pi / 3.0, 1e-15));
  REQUIRE_THAT(res::points_per_lattice(safe), WithinRel(6.0, 1e-12));

  // Just inside and just outside.
  CHECK(res::third_harmonic_resolved(safe * 0.999));
  CHECK(res::mask_spares_second_harmonic(safe * 0.999));
  CHECK_FALSE(res::third_harmonic_resolved(safe * 1.001));
  CHECK_FALSE(res::mask_spares_second_harmonic(safe * 1.001));
}

TEST_CASE("Tungsten resolution: the historical spacing sits on the wrong side",
          "[tungsten][resolution][unit]") {
  // The solidification presets use this spacing. It fails both conditions,
  // which is why the mask cannot simply be switched on there: with it off the
  // third harmonic folds back, with it on the crystal's own second harmonic
  // is deleted. The qualified preset is a different file.
  constexpr double shipped = 1.1107207345395915;
  CHECK_FALSE(res::third_harmonic_resolved(shipped));
  CHECK_FALSE(res::mask_spares_second_harmonic(shipped));
  CHECK(shipped > res::dealias_safe_dx());
  INFO("shipped dx is " << 100.0 * (shipped / res::dealias_safe_dx() - 1.0)
                        << "% coarser than the threshold");
  CHECK(shipped / res::dealias_safe_dx() < 1.10); // 6.1%, not a wild miss

  // Where the third harmonic lands when it folds back.
  const double k_ny = res::nyquist_k(shipped);
  const double folded = 2.0 * k_ny - 3.0 * res::kLatticeWavenumber;
  INFO("3k folds back to " << folded);
  CHECK(folded > 0.0);
  CHECK(folded < k_ny);
  REQUIRE_THAT(folded, WithinRel(2.6568542495, 1e-8));
}

TEST_CASE("Tungsten resolution: the 1/2 rule needs eight points",
          "[tungsten][resolution][unit]") {
  // Exact dealiasing of a cubic term is the 1/2 rule, not the 2/3 rule. This
  // is the spacing at which the measured mask-on/mask-off difference falls to
  // the 1e-4 level.
  const double clean = res::dealias_clean_dx();
  REQUIRE_THAT(res::points_per_lattice(clean), WithinRel(8.0, 1e-12));
  CHECK(clean < res::dealias_safe_dx());
  CHECK(res::third_harmonic_resolved(clean));
  CHECK(res::mask_spares_second_harmonic(clean));
}

namespace {

struct DensityMoments {
  double mean{};
  double rms{};
  double lo{};
  double hi{};
};

DensityMoments density_moments(tungsten::TungstenSession &session) {
  double sum = 0.0;
  double sumsq = 0.0;
  double count = 0.0;
  double lo = 1e300;
  double hi = -1e300;
  auto &psi = session.psi();
  psi.for_each_owned([&](int i, int j, int k) {
    const double value = psi(i, j, k);
    sum += value;
    sumsq += value * value;
    count += 1.0;
    lo = std::min(lo, value);
    hi = std::max(hi, value);
  });
  double reduced[5]{};
  double local[5] = {sum, sumsq, count, lo, hi};
  MPI_Allreduce(local, reduced, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local + 3, reduced + 3, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local + 4, reduced + 4, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  DensityMoments out;
  out.mean = reduced[0] / reduced[2];
  out.rms = std::sqrt(reduced[1] / reduced[2]);
  out.lo = reduced[3];
  out.hi = reduced[4];
  return out;
}

/// Multiplicity-weighted final-solution power for |k|>2. This is a
/// solution-tail diagnostic, NOT a direct nonlinear-RHS aliasing measure:
/// spurious folded products can contaminate retained low modes.
double high_mode_power(tungsten::TungstenSession &session) {
  double local = 0.0;
  auto &hat = session.system().psi_hat();
  const auto outbox = session.fft().get_outbox_bounds();
  const auto size = pfc::domain::get_size(session.domain());
  hat.with_host_view([&](std::complex<double> *values, std::size_t) {
    pfc::fft::kspace::for_each_kpoint(
        outbox, session.domain(),
        [&](std::size_t index, double kx, double ky, double kz, int ix, int, int) {
          const double wavenumber = std::sqrt(kx * kx + ky * ky + kz * kz);
          if (wavenumber > 2.0) {
            local +=
                pfc::fft::r2c_multiplicity(ix, size[0]) * std::norm(values[index]);
          }
        });
  });
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

// Exact physical-grid state and the six normalized reciprocal BCC shell modes
// are retained for direct comparisons across resolutions, not only moments.
// Nonlinear N_hat before the 2/3 output mask, restricted to the same
// 405 retained low Fourier modes on 16^3 and 32^3 one-rank grids.
constexpr int kRhsLow = 4;
constexpr std::size_t kRhsCount = 5u * 9u * 9u;
using RhsModes = std::array<std::complex<double>, kRhsCount>;

struct Evolved {
  DensityMoments density;
  double high{};
  bool dealias{};
  double dx{};
  std::vector<double> state;
  std::array<std::complex<double>, 6> bcc_shell{};
  std::optional<RhsModes> rhs;
};

std::array<std::complex<double>, 6> bcc_shell(tungsten::TungstenSession &session) {
  constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
  const std::array<int, 6> mx{1, 1, 0, 1, 1, 0};
  const std::array<int, 6> my{1, 0, 1, 0, -1, 1};
  const std::array<int, 6> mz{0, 1, 1, -1, 0, -1};
  std::array<std::complex<double>, 6> local{};
  auto &psi = session.psi();
  std::size_t samples = 0;
  psi.for_each_owned([&](int i, int j, int k) {
    const auto x = psi.coords(i, j, k);
    for (std::size_t m = 0; m < local.size(); ++m) {
      const double projection = mx[m] * x[0] + my[m] * x[1] + mz[m] * x[2];
      const double phase = inv_sqrt2 * projection;
      const std::complex<double> weight{std::cos(phase), -std::sin(phase)};
      local[m] += psi(i, j, k) * weight;
    }
    ++samples;
  });
  // This first gate is explicitly a one-rank reference; mode amplitudes are
  // normalized by sample count so 16³ and 32³ coefficients are comparable.
  for (auto &value : local) value /= static_cast<double>(samples);
  return local;
}

// Evaluate the actual shipped N(psi,psi_mf) at the final evolved field.
// attempt() writes the *unmasked* N_hat before applying its optional
// 2/3 output mask. It does not commit a new primary field. The HeFFTe
// forward transform is unnormalized, so divide by global cell count.
// The one-rank gate below is necessary to own every requested mode.
RhsModes nonlinear_rhs_modes(tungsten::TungstenSession &session, double time) {
  (void)session.system().attempt(time);
  auto &n_hat = session.state().get_field<std::complex<double>, pfc::HostSpace>(
      session.system().options().n_hat_name);
  const auto outbox = session.fft().get_outbox_bounds();
  const auto size = pfc::domain::get_size(session.domain());
  const double cells = static_cast<double>(size[0]) * size[1] * size[2];
  RhsModes modes{};
  std::array<bool, kRhsCount> seen{};
  n_hat.with_host_view([&](std::complex<double> *values, std::size_t) {
    pfc::fft::kspace::for_each_kpoint(
        outbox, session.domain(),
        [&](std::size_t index, double, double, double, int ix, int iy, int iz) {
          const int my = pfc::fft::kspace::signed_wave_index(iy, size[1]);
          const int mz = pfc::fft::kspace::signed_wave_index(iz, size[2]);
          const bool outside_y = std::abs(my) > kRhsLow;
          const bool outside_z = std::abs(mz) > kRhsLow;
          if (ix > kRhsLow || outside_y || outside_z) return;
          const int y = my + kRhsLow;
          const int z = mz + kRhsLow;
          const auto slot = static_cast<std::size_t>(ix + 5 * (y + 9 * z));
          modes[slot] = values[index] / cells;
          seen[slot] = true;
        });
  });
  for (bool found : seen) REQUIRE(found);
  return modes;
}

std::pair<double, double> rhs_gap(const RhsModes &left, const RhsModes &right) {
  double peak = 0.0;
  double sumsq = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    const double gap = std::abs(left[i] - right[i]);
    peak = std::max(peak, gap);
    sumsq += gap * gap;
  }
  return {std::sqrt(sumsq / left.size()), peak};
}

Evolved evolve(const nlohmann::json &settings) {
  tungsten::TungstenSession session(settings, 0, 1, MPI_COMM_WORLD);
  const bool dealias = session.system().options().dealias;
  const auto &spacing = session.domain().spacing;
  REQUIRE_THAT(spacing[1], WithinRel(spacing[0], 1e-15));
  REQUIRE_THAT(spacing[2], WithinRel(spacing[0], 1e-15));
  const double dx = spacing[0];
  session.run();
  const auto amplitudes = bcc_shell(session);
  Evolved out;
  out.density = density_moments(session);
  out.high = high_mode_power(session);
  out.dealias = dealias;
  out.dx = dx;
  out.state = session.psi().vec();
  out.bcc_shell = amplitudes;
  // The 8^3 historical control has Nyquist index +4 but no distinct -4;
  // it cannot hold the full fixed [-4,4]^2 RHS comparison window.
  // Keep that diagnostic for grids able to represent every requested mode.
  const auto size = pfc::domain::get_size(session.domain());
  const bool x_complete = size[0] > 2 * kRhsLow;
  const bool y_complete = size[1] > 2 * kRhsLow;
  const bool z_complete = size[2] > 2 * kRhsLow;
  if (x_complete && y_complete && z_complete) {
    const double t1 = settings.at("timestepping").at("t1").get<double>();
    out.rhs = nonlinear_rhs_modes(session, t1);
  }
  return out;
}

// OpenPFC Field::coords(i,j,k) uses x=origin+i*spacing, NOT cell centers.
// On the same periodic physical box, 16^3 points are exactly a subset
// of the 32^3 grid, at (2i,2j,2k). Compare equal physical locations,
// with no averaging/filtering or interpolant phase shift.
auto sampled_field_error(const Evolved &coarse, const Evolved &fine) {
  constexpr int n = 16;
  constexpr int nf = 32;
  REQUIRE(coarse.state.size() == static_cast<std::size_t>(n * n * n));
  REQUIRE(fine.state.size() == static_cast<std::size_t>(nf * nf * nf));
  double peak = 0.0;
  double sumsq = 0.0;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const auto c = static_cast<std::size_t>(i + n * (j + n * k));
        const int fi = 2 * i;
        const int fj = 2 * j;
        const int fk = 2 * k;
        const auto f = static_cast<std::size_t>(fi + nf * (fj + nf * fk));
        const double gap = coarse.state[c] - fine.state[f];
        peak = std::max(peak, std::abs(gap));
        sumsq += gap * gap;
      }
    }
  }
  return std::pair<double, double>{std::sqrt(sumsq / (n * n * n)), peak};
}

double relative_change(double left, double right) {
  return std::abs(right - left) / std::max(std::abs(left), 1e-30);
}

} // namespace

TEST_CASE("Nested OpenPFC nodal grids sample the same physical wave",
          "[tungsten][resolution][unit]") {
  Evolved coarse;
  Evolved fine;
  coarse.state.resize(16u * 16u * 16u);
  fine.state.resize(32u * 32u * 32u);
  const auto fill = [](std::vector<double> &values, int n) {
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          // These are the actual Field::coords nodal coordinates.
          const double x = i / static_cast<double>(n);
          const double y = j / static_cast<double>(n);
          const double z = k / static_cast<double>(n);
          const double phase = 2.0 * x + 3.0 * y - 5.0 * z;
          const double mixed = 0.15 * std::sin(2.0 * std::numbers::pi * phase);
          values[static_cast<std::size_t>(i + n * (j + n * k))] =
              0.3 + 0.4 * std::sin(2.0 * std::numbers::pi * 3.0 * x) +
              0.25 * std::cos(2.0 * std::numbers::pi * 4.0 * y) + mixed;
        }
      }
    }
  };
  fill(coarse.state, 16);
  fill(fine.state, 32);
  const auto [l2, linf] = sampled_field_error(coarse, fine);
  REQUIRE(l2 < 1e-12);
  REQUIRE(linf < 1e-12);
}

TEST_CASE("Tungsten resolved BCC preset passes the finer-grid gate",
          "[tungsten][resolution][preset]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    SKIP("the resolved BCC gate requires a single CPU rank");
  }
  tungsten::register_catalog();

  const auto path =
      std::filesystem::path(TUNGSTEN_INPUTS_JSON_DIR) / "tungsten_resolved_bcc.json";
  const nlohmann::json preset = pfc::ui::load_settings_file(path);
  REQUIRE(preset["model"]["dealias"].get<bool>());
  REQUIRE(preset["domain"]["size"] == nlohmann::json::array({16, 16, 16}));
  REQUIRE(preset["timestepping"]["t0"].get<double>() == 0.0);
  REQUIRE(preset["timestepping"]["t1"].get<double>() == 20.0);
  REQUIRE(preset["timestepping"]["dt"].get<double>() == 1.0);

  // One BCC lattice constant. Sixteen points along that side is 11.31
  // points per `|k| = 1` period, inside the cubic 1/2 rule.
  const double bcc_side = 2.0 * std::numbers::pi * std::sqrt(2.0);
  REQUIRE_THAT(preset["domain"]["upper"][0].get<double>(),
               WithinRel(bcc_side, 1e-15));
  REQUIRE_THAT(preset["domain"]["upper"][1].get<double>(),
               WithinRel(bcc_side, 1e-15));
  REQUIRE_THAT(preset["domain"]["upper"][2].get<double>(),
               WithinRel(bcc_side, 1e-15));

  // Declared gate, with margin over a measured 20-step host run.
  constexpr double rms_tolerance = 1e-6;
  constexpr double max_tolerance = 1e-4;
  constexpr double high_tolerance = 1e-4;

  const Evolved selected = evolve(preset);
  REQUIRE(selected.dealias);
  REQUIRE_THAT(selected.dx, WithinRel(bcc_side / 16.0, 1e-12));
  REQUIRE(res::third_harmonic_resolved(selected.dx));
  REQUIRE(res::mask_spares_second_harmonic(selected.dx));
  REQUIRE(res::points_per_lattice(selected.dx) >= 8.0);
  REQUIRE(selected.dx <= res::dealias_clean_dx());
  REQUIRE(std::isfinite(selected.density.rms));
  REQUIRE(std::isfinite(selected.density.hi));
  REQUIRE(selected.density.hi > selected.density.lo);

  nlohmann::json finer = preset;
  finer["domain"]["size"] = nlohmann::json::array({32, 32, 32});
  const Evolved control = evolve(finer);
  REQUIRE(control.dealias);
  REQUIRE_THAT(control.dx, WithinRel(bcc_side / 32.0, 1e-12));
  REQUIRE(res::third_harmonic_resolved(control.dx));
  REQUIRE(relative_change(selected.density.rms, control.density.rms) <
          rms_tolerance);
  REQUIRE(relative_change(selected.density.hi, control.density.hi) < max_tolerance);
  REQUIRE(relative_change(selected.density.mean, control.density.mean) < 1e-8);

  // Unlike scalar RMS/maximum, this compares the actual field on the same
  // physical sample points and the complex BCC reciprocal modes. A phase
  // difference is visible rather than disappearing in |hat(k)| or sumsq.
  const auto [field_l2, field_linf] = sampled_field_error(selected, control);
  double shell_gap = 0.0;
  for (std::size_t mode = 0; mode < selected.bcc_shell.size(); ++mode) {
    const auto observed = selected.bcc_shell[mode];
    const auto expected = control.bcc_shell[mode];
    const double gap = std::abs(observed - expected);
    shell_gap = std::max(shell_gap, gap);
  }
  std::printf("TUNGSTEN_BCC N=16 reference_N=32 dt=1 steps=20 dealias=1 "
              "rms=%.17g rms32=%.17g peak=%.17g peak32=%.17g "
              "field_l2=%.17g field_linf=%.17g shell_gap=%.17g\n",
              selected.density.rms, control.density.rms, selected.density.hi,
              control.density.hi, field_l2, field_linf, shell_gap);
  REQUIRE(std::isfinite(field_l2));
  REQUIRE(std::isfinite(field_linf));
  REQUIRE(std::isfinite(shell_gap));
  REQUIRE(field_l2 < 1.0e-3);
  REQUIRE(field_linf < 4.0e-3);
  REQUIRE(shell_gap < 1.0e-3);

  // Directly measure the actual retained nonlinear RHS coefficients, not
  // only scalar density moments or the final solution's spectral tail.
  REQUIRE(selected.rhs.has_value());
  REQUIRE(control.rhs.has_value());
  const auto [rhs_l2, rhs_linf] = rhs_gap(*selected.rhs, *control.rhs);
  std::printf("TUNGSTEN_RHS N=16 reference_N=32 retained_modes=%zu "
              "rhs_l2=%.17g rhs_linf=%.17g\n",
              kRhsCount, rhs_l2, rhs_linf);
  REQUIRE(std::isfinite(rhs_l2));
  REQUIRE(std::isfinite(rhs_linf));

  // Same preset, mask off: compare final-solution tail power only. This
  // cannot establish low-mode nonlinear aliasing or exact 1/2 dealiasing.
  nlohmann::json unmasked = preset;
  unmasked["model"]["dealias"] = false;
  const Evolved bare = evolve(unmasked);
  REQUIRE_FALSE(bare.dealias);
  REQUIRE(bare.rhs.has_value());
  const auto [masked_gap, masked_peak] = rhs_gap(*selected.rhs, *bare.rhs);
  std::printf("TUNGSTEN_RHS N=16 mask_on_vs_off retained_modes=%zu "
              "rhs_l2=%.17g rhs_linf=%.17g\n",
              kRhsCount, masked_gap, masked_peak);
  REQUIRE(std::isfinite(masked_gap));
  REQUIRE(std::isfinite(masked_peak));
  REQUIRE(relative_change(selected.density.rms, bare.density.rms) < rms_tolerance);
  REQUIRE(relative_change(selected.density.hi, bare.density.hi) < max_tolerance);
  REQUIRE(relative_change(selected.high, bare.high) < high_tolerance);

  // The historical spacing on this same box. The mask moves the field, and
  // the density maximum leaves the band the finer grid just satisfied.
  nlohmann::json coarse = preset;
  coarse["domain"]["size"] = nlohmann::json::array({8, 8, 8});
  coarse["model"]["dealias"] = false;
  nlohmann::json coarse_masked = coarse;
  coarse_masked["model"]["dealias"] = true;
  const double coarse_dx = bcc_side / 8.0;
  REQUIRE_THAT(coarse_dx, WithinRel(1.1107207345395915, 1e-12));
  REQUIRE_FALSE(res::third_harmonic_resolved(coarse_dx));
  REQUIRE_FALSE(res::mask_spares_second_harmonic(coarse_dx));
  const Evolved coarse_off = evolve(coarse);
  const Evolved coarse_on = evolve(coarse_masked);
  REQUIRE_FALSE(coarse_off.rhs.has_value());
  REQUIRE_FALSE(coarse_on.rhs.has_value());
  REQUIRE_FALSE(coarse_off.dealias);
  REQUIRE(coarse_on.dealias);
  REQUIRE_THAT(coarse_off.dx, WithinRel(coarse_dx, 1e-12));
  REQUIRE_THAT(coarse_on.dx, WithinRel(coarse_dx, 1e-12));
  REQUIRE(relative_change(coarse_off.density.hi, coarse_on.density.hi) > 5e-3);
  REQUIRE(relative_change(coarse_off.high, coarse_on.high) > 0.5);
  REQUIRE(relative_change(coarse_off.density.hi, selected.density.hi) >
          max_tolerance);

  nlohmann::json invalid = preset;
  invalid["model"]["dealias"] = "yes";
  REQUIRE_THROWS_AS((tungsten::TungstenSession(invalid, 0, 1, MPI_COMM_WORLD)),
                    std::invalid_argument);
}
