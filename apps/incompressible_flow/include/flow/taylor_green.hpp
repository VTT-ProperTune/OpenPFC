// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file taylor_green.hpp
 * @brief 3-D Taylor–Green on the installed Fourier velocity.
 *
 * The cube is [0, 2π]³. The initial velocity is
 * `u = sin(x) cos(y) cos(z)`, `v = -cos(x) sin(y) cos(z)`, `w = 0`.
 * The step is integrating-factor RK4 of `P(u × ω)` after a state-level
 * 2/3 mask. This file does not own a projector. Each rank owns the FFT
 * pencil HeFFTe assigns it. Diagnostics are sums and maxima over that
 * pencil, reduced across the communicator. Modal divergence is
 * `|k · û| / N³`: the stored coefficient carries the unnormalized
 * factor `N³`. The 2-D
 * vorticity–streamfunction prototype is not this case.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <limits>
#include <memory>
#include <openpfc/kernel/data/grid_field.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/fourier_vector.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/solvers/incompressible/rotational.hpp>

namespace flow {

using Complex = pfc::field::Complex;

inline constexpr double kinetic_energy_0 = 0.125;
inline constexpr double enstrophy_0 = 0.75;
inline constexpr double cfl_limit = 2.0;

struct Diagnostics {
  double ke{0.0};
  double enstrophy{0.0};
  double dissipation{0.0};
  double div_l2{0.0};
  double div_linf{0.0};
  double modal_div_max{0.0};
  double cfl{0.0};
  double mean_u{0.0};
  double mean_v{0.0};
  double mean_w{0.0};
  double w_l2{0.0};
  double max_abs_w{0.0};
  bool finite{true};
};

struct State {
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

  State() = default;
  State(const State &) = delete;
  State &operator=(const State &) = delete;
  State(State &&) noexcept = default;
  State &operator=(State &&) noexcept = default;
};

[[nodiscard]] inline double cell_count(std::array<int, 3> n) noexcept {
  return static_cast<double>(n[0]) * static_cast<double>(n[1]) *
         static_cast<double>(n[2]);
}

[[nodiscard]] inline long long steps_for(double time, double dt) {
  if (!(time > 0.0) || !(dt > 0.0)) {
    throw std::invalid_argument("incompressible_flow: time and dt must be positive");
  }
  const double ratio = time / dt;
  const auto steps = std::llround(ratio);
  if (steps < 1 || std::abs(ratio - static_cast<double>(steps)) > 1.0e-8) {
    throw std::invalid_argument(
        "incompressible_flow: time must be an integer number of steps");
  }
  return steps;
}

/// `stride == 0` records only the final step. A positive stride records
/// every multiple of that many steps. The stride is required to divide
/// the run, so the final step is one of those multiples.
[[nodiscard]] inline bool record_at(long long step, long long steps,
                                    long long stride) noexcept {
  if (step < 1 || step > steps) return false;
  if (stride <= 0) return step == steps;
  return step % stride == 0;
}

struct VolumeSample {
  double mean_square{0.0};
  long long points{0};
};

[[nodiscard]] inline State make_state(int n, double nu, double dt, int rank,
                                      int nproc) {
  if (n < 8 || n % 2 != 0) {
    throw std::invalid_argument(
        "incompressible_flow: N must be even and at least 8");
  }
  if (nu < 0.0 || !(dt > 0.0)) {
    throw std::invalid_argument("incompressible_flow: nu must be >= 0 and dt > 0");
  }
  const double h = pfc::two_pi / static_cast<double>(n);
  auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({h, h, h}));
  State state;
  state.stack = std::make_unique<pfc::sim::stacks::SpectralCPUStack>(
      std::move(domain), rank, nproc, MPI_COMM_WORLD);
  state.n = {n, n, n};
  state.spacing = {h, h, h};
  state.nu = nu;
  state.dt = dt;
  auto &fft = state.stack->fft();
  const auto nhat = fft.size_outbox();
  state.u.assign(nhat, Complex{});
  state.v.assign(nhat, Complex{});
  state.w.assign(nhat, Complex{});
  pfc::incompressible::viscous_exponentials(fft.get_outbox_bounds(), state.n,
                                            state.spacing, nu, dt, state.exp_dt,
                                            state.exp_half);
  return state;
}

inline void initialize_taylor_green(State &state) {
  auto &fft = state.stack->fft();
  const auto inbox = fft.get_inbox_bounds();
  const auto domain = pfc::domain::create(pfc::GridSize(state.n),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing(state.spacing));
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
  fft.forward(rx.vec(), state.u);
  fft.forward(ry.vec(), state.v);
  fft.forward(rz.vec(), state.w);
  const auto outbox = fft.get_outbox_bounds();
  const auto nhat = state.u.size();
  pfc::field::leray_project(outbox, state.n, state.spacing, state.u.data(),
                            state.v.data(), state.w.data(), nhat);
  pfc::field::apply_two_thirds(outbox, state.n, state.spacing, state.u.data(),
                               state.v.data(), state.w.data(), nhat);
}

inline void step(State &state) {
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
      },
      true);
}

[[nodiscard]] inline bool hats_finite(const State &state) {
  auto ok = [](const std::vector<Complex> &hat) {
    for (const auto &z : hat) {
      if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) return false;
    }
    return true;
  };
  return ok(state.u) && ok(state.v) && ok(state.w);
}

[[nodiscard]] inline Diagnostics diagnose(State &state) {
  auto &fft = state.stack->fft();
  const auto outbox = fft.get_outbox_bounds();
  const auto inbox = fft.get_inbox_bounds();
  const auto nhat = state.u.size();
  const double ncells = cell_count(state.n);

  std::vector<double> rx(fft.size_inbox()), ry(fft.size_inbox()),
      rz(fft.size_inbox());
  fft.backward(state.u, rx);
  fft.backward(state.v, ry);
  fft.backward(state.w, rz);

  std::vector<Complex> ox(nhat), oy(nhat), oz(nhat);
  pfc::field::curl_hat(outbox, state.n, state.spacing, state.u.data(),
                       state.v.data(), state.w.data(), ox.data(), oy.data(),
                       oz.data(), nhat);
  std::vector<double> oxr(fft.size_inbox()), oyr(fft.size_inbox()),
      ozr(fft.size_inbox());
  fft.backward(ox, oxr);
  fft.backward(oy, oyr);
  fft.backward(oz, ozr);

  std::vector<Complex> div_hat(nhat);
  pfc::fft::kspace::for_each_kpoint(
      outbox, state.n, state.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        div_hat[idx] = pfc::field::modal_divergence(
            state.u[idx], state.v[idx], state.w[idx],
            pfc::field::odd_wave(i, j, k, state.n, state.spacing));
      });
  std::vector<double> div_real(fft.size_inbox());
  fft.backward(div_hat, div_real);

  Diagnostics diag;
  diag.finite = hats_finite(state);
  double ke_sum = 0.0;
  double enstrophy_sum = 0.0;
  double w2 = 0.0;
  double su = 0.0;
  double sv = 0.0;
  double sw = 0.0;
  double peak = 0.0;
  double max_w = 0.0;
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
        enstrophy_sum +=
            oxr[idx] * oxr[idx] + oyr[idx] * oyr[idx] + ozr[idx] * ozr[idx];
        div2 += div_real[idx] * div_real[idx];
        div_linf = std::max(div_linf, std::abs(div_real[idx]));
        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(w))
          diag.finite = false;
      }
    }
  }
  double sums[7] = {ke_sum, enstrophy_sum, w2, su, sv, sw, div2};
  double peaks[4] = {peak, max_w, div_linf,
                     pfc::field::max_modal_divergence(outbox, state.n, state.spacing,
                                                      state.u.data(), state.v.data(),
                                                      state.w.data(), nhat)};
  int finite = diag.finite ? 1 : 0;
  const MPI_Comm comm = state.stack->mpi_comm();
  MPI_Allreduce(MPI_IN_PLACE, sums, 7, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, peaks, 4, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(MPI_IN_PLACE, &finite, 1, MPI_INT, MPI_MIN, comm);
  diag.finite = finite != 0;
  diag.ke = 0.5 * sums[0] / ncells;
  diag.enstrophy = sums[1] / ncells;
  diag.dissipation = state.nu * diag.enstrophy;
  diag.div_l2 = std::sqrt(sums[6] / ncells);
  diag.div_linf = peaks[2];
  // |k·û| on the stored hat grows like N³. Divide so one absolute
  // tolerance still means a solenoidal field on a fine grid.
  diag.modal_div_max = peaks[3] / ncells;
  diag.cfl = state.dt * peaks[0] / state.spacing[0];
  diag.mean_u = sums[3] / ncells;
  diag.mean_v = sums[4] / ncells;
  diag.mean_w = sums[5] / ncells;
  diag.w_l2 = std::sqrt(sums[2] / ncells);
  diag.max_abs_w = peaks[1];
  return diag;
}

/// Gather |ω| onto rank 0 and write one x-fastest float32 cube.
/// Every rank blocks until that file is closed.
[[nodiscard]] inline VolumeSample
write_vorticity_magnitude(State &state, const std::string &path) {
  auto &fft = state.stack->fft();
  const auto outbox = fft.get_outbox_bounds();
  const auto inbox = fft.get_inbox_bounds();
  const auto nhat = state.u.size();
  const MPI_Comm comm = state.stack->mpi_comm();
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);

  std::vector<Complex> ox(nhat), oy(nhat), oz(nhat);
  pfc::field::curl_hat(outbox, state.n, state.spacing, state.u.data(),
                       state.v.data(), state.w.data(), ox.data(), oy.data(),
                       oz.data(), nhat);
  std::vector<double> oxr(fft.size_inbox()), oyr(fft.size_inbox()),
      ozr(fft.size_inbox());
  fft.backward(ox, oxr);
  fft.backward(oy, oyr);
  fft.backward(oz, ozr);

  const long long local_n = inbox.count();
  std::vector<float> local(static_cast<std::size_t>(local_n));
  double sum = 0.0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const auto idx = static_cast<std::size_t>(inbox.to_linear({i, j, k}));
        const double mag = std::sqrt(oxr[idx] * oxr[idx] + oyr[idx] * oyr[idx] +
                                     ozr[idx] * ozr[idx]);
        local[idx] = static_cast<float>(mag);
        sum += mag * mag;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  std::array<int, 6> meta{inbox.low[0],  inbox.low[1],  inbox.low[2],
                          inbox.high[0], inbox.high[1], inbox.high[2]};
  std::vector<int> allmeta(static_cast<std::size_t>(nproc) * 6);
  MPI_Allgather(meta.data(), 6, MPI_INT, allmeta.data(), 6, MPI_INT, comm);

  const long long expected = static_cast<long long>(state.n[0]) *
                             static_cast<long long>(state.n[1]) *
                             static_cast<long long>(state.n[2]);
  std::vector<int> counts(static_cast<std::size_t>(nproc));
  std::vector<int> displs(static_cast<std::size_t>(nproc));
  long long total = 0;
  for (int r = 0; r < nproc; ++r) {
    const int *box = allmeta.data() + static_cast<std::size_t>(r) * 6;
    const long long nx = static_cast<long long>(box[3]) - box[0] + 1;
    const long long ny = static_cast<long long>(box[4]) - box[1] + 1;
    const long long nz = static_cast<long long>(box[5]) - box[2] + 1;
    const long long count = nx * ny * nz;
    counts[static_cast<std::size_t>(r)] = static_cast<int>(count);
    displs[static_cast<std::size_t>(r)] = static_cast<int>(total);
    total += count;
  }
  const bool layout_ok =
      expected <= static_cast<long long>(std::numeric_limits<int>::max()) &&
      total == expected &&
      counts[static_cast<std::size_t>(rank)] == static_cast<int>(local.size());
  int layout = layout_ok ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &layout, 1, MPI_INT, MPI_MIN, comm);
  if (layout != 1) {
    throw std::runtime_error(
        "incompressible_flow: vorticity pencils do not cover the grid");
  }

  std::vector<float> gathered(rank == 0 ? static_cast<std::size_t>(expected) : 0);
  MPI_Gatherv(local.data(), counts[static_cast<std::size_t>(rank)], MPI_FLOAT,
              rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(),
              MPI_FLOAT, 0, comm);

  int wrote = 1;
  if (rank == 0) {
    const int n = state.n[0];
    std::vector<float> cube(static_cast<std::size_t>(expected));
    std::vector<unsigned char> hits(static_cast<std::size_t>(expected), 0);
    for (int r = 0; r < nproc; ++r) {
      const int *box = allmeta.data() + static_cast<std::size_t>(r) * 6;
      const int low0 = box[0];
      const int low1 = box[1];
      const int low2 = box[2];
      const int lx = box[3] - box[0] + 1;
      const int ly = box[4] - box[1] + 1;
      const int lz = box[5] - box[2] + 1;
      const float *src = gathered.data() + displs[static_cast<std::size_t>(r)];
      for (int k = 0; k < lz; ++k) {
        for (int j = 0; j < ly; ++j) {
          for (int i = 0; i < lx; ++i) {
            const int gi = low0 + i;
            const int gj = low1 + j;
            const int gk = low2 + k;
            if (gi < 0 || gj < 0 || gk < 0 || gi >= n || gj >= n || gk >= n) {
              wrote = 0;
              break;
            }
            const std::size_t g =
                (static_cast<std::size_t>(gk) * static_cast<std::size_t>(n) +
                 static_cast<std::size_t>(gj)) *
                    static_cast<std::size_t>(n) +
                static_cast<std::size_t>(gi);
            if (hits[g] != 0) wrote = 0;
            hits[g] = 1;
            cube[g] =
                src[(static_cast<std::size_t>(k) * static_cast<std::size_t>(ly) +
                     static_cast<std::size_t>(j)) *
                        static_cast<std::size_t>(lx) +
                    static_cast<std::size_t>(i)];
          }
        }
      }
    }
    if (wrote == 1) {
      for (unsigned char hit : hits) {
        if (hit == 0) wrote = 0;
      }
    }
    if (wrote == 1) {
      std::ofstream out(path, std::ios::binary);
      if (!out) {
        wrote = 0;
      } else {
        out.write(reinterpret_cast<const char *>(cube.data()),
                  static_cast<std::streamsize>(cube.size() * sizeof(float)));
        if (!out) wrote = 0;
      }
    }
  }
  MPI_Bcast(&wrote, 1, MPI_INT, 0, comm);
  if (wrote != 1) {
    throw std::runtime_error("incompressible_flow: failed to write " + path);
  }

  VolumeSample sample;
  sample.points = expected;
  sample.mean_square = sum / cell_count(state.n);
  return sample;
}

/// Stop reason for one series. `ok` means every requested sample finished.
[[nodiscard]] inline std::string advance(State &state, long long steps) {
  auto stop = [&](const Diagnostics &diag) -> std::string {
    if (!diag.finite) return "nonfinite";
    if (diag.cfl > cfl_limit) return "cfl";
    return {};
  };
  const auto initial = diagnose(state);
  if (const auto why = stop(initial); !why.empty()) return why;
  for (long long taken = 0; taken < steps; ++taken) {
    step(state);
    const auto diag = diagnose(state);
    if (const auto why = stop(diag); !why.empty()) return why;
  }
  return "ok";
}

} // namespace flow
