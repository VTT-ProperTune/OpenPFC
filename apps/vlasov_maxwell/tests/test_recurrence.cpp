// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_recurrence.cpp
 * @brief Measure the discrete-velocity recurrence time across N_v, not
 *        merely constrain runs to sit below it.
 *
 * @details
 * Free-streaming a Landau density perturbation on a uniform \(v_x\) grid
 * reconstructs \(|\hat\rho(k)|\) at \(T_R = 2\pi/(k\Delta v)\). That is a
 * fact about the grid: the self-consistent field is off, so the only
 * operator that moves is the exact spectral \(x\)-shift, and the revival
 * cannot be blamed on Landau damping or on a field solver.
 *
 * Cell-centred \(v_j = -v_{\max}+(j+\tfrac12)\Delta v\) puts a global
 * minus sign on the *signed* mode at \(t = T_R\). The assertion is on
 * \(|\hat\rho|\), not on signed `mode_ex` (which is frozen anyway, because
 * the fields are frozen).
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <vlasov_maxwell/diagnostics.hpp>
#include <vlasov_maxwell/ics.hpp>
#include <vlasov_maxwell/parameters.hpp>
#include <vlasov_maxwell/phase_space.hpp>
#include <vlasov_maxwell/step.hpp>

using Catch::Approx;

namespace {

constexpr double kPi = 3.14159265358979323846;

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

/// Least-squares slope of y against x. Used for T_meas vs 1/dv.
double fitted_slope(const std::vector<double> &x, const std::vector<double> &y) {
  const std::size_t n = x.size();
  REQUIRE(n == y.size());
  REQUIRE(n >= 3);
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  const double nd = static_cast<double>(n);
  const double den = nd * sxx - sx * sx;
  REQUIRE(std::fabs(den) > 0.0);
  return (nd * sxy - sx * sy) / den;
}

/// Cosine coefficient of mode m, signed. The trap the modulus exists to
/// catch: at T_R this flips rather than reconstructing.
double signed_cosine_mode(const vlasov::SpectralLine1D &line,
                          const std::vector<double> &g, int m) {
  const auto h = line.forward(g);
  if (m < 0 || static_cast<std::size_t>(m) >= h.size()) return 0.0;
  const double n = static_cast<double>(h.size());
  return 2.0 * h[static_cast<std::size_t>(m)].real() / n;
}

struct RecurrenceRow {
  int nvx{0};
  double dv{0.0};
  double t_pred{0.0};
  double t_meas{0.0};
  double ratio{0.0};
  double a0{0.0};
  double a_rev{0.0};
  double signed0{0.0};
  double signed_rev{0.0};
};

/// Stream a Landau IC with frozen zero fields; return the measured revival.
RecurrenceRow run_streaming(int nvx, int nvy, double t_end_over_tr) {
  const double vth = 0.05;
  const double twopi = 2.0 * kPi;
  vlasov::SimParams p;
  p.nx = 32;
  p.nvx = nvx;
  p.nvy = nvy;
  p.Lx = twopi * vth / 0.5; // k lambda_D = 0.5 at mode 1
  p.v_max = 8.0 * vth;
  p.v_thermal = vth;
  p.self_consistent = false;
  p.electrostatic = true;
  p.interp_order = 5;
  p.validate();

  const double k = p.k_skin(1);
  const double dv = p.dvx();
  const double t_pred = vlasov::recurrence_time(k, dv);
  const double t_end = t_end_over_tr * t_pred;
  const double dt = 0.05;
  const int n_steps = std::max(1, static_cast<int>(std::llround(t_end / dt)));
  const int halo = vlasov::required_halo_width(0.0, p.interp_order);

  vlasov::PhaseSpace ps(p, halo, MPI_COMM_SELF);
  const double amp = 0.01;
  ps.initialise(0, [&](double x, double vx, double vy) {
    return vlasov::ics::density_perturbation(x, k, amp) *
           vlasov::ics::maxwellian(vx, vy, vth);
  });
  vlasov::Stepper st(p, ps);
  // Fields stay at zero: update_fields is a no-op, so the Lorentz pair is
  // the identity and the only motion is exact streaming in x.
  st.deposit_all();

  std::vector<double> ts, amp_abs, amp_signed;
  auto sample = [&](double t) {
    ts.push_back(t);
    amp_abs.push_back(vlasov::mode_amplitude(st.line, st.sources.rho, 1));
    amp_signed.push_back(signed_cosine_mode(st.line, st.sources.rho, 1));
  };
  sample(0.0);
  for (int step = 1; step <= n_steps; ++step) {
    st.advance(dt);
    sample(static_cast<double>(step) * dt);
  }

  const auto peak = vlasov::find_recurrence_peak(ts, amp_abs, t_pred);
  RecurrenceRow row;
  row.nvx = nvx;
  row.dv = dv;
  row.t_pred = t_pred;
  row.a0 = amp_abs.empty() ? 0.0 : amp_abs.front();
  row.signed0 = amp_signed.empty() ? 0.0 : amp_signed.front();
  if (peak.kind == vlasov::RevivalKind::found) {
    row.t_meas = peak.t;
    row.a_rev = peak.amplitude;
    row.ratio = peak.t / t_pred;
    // Signed mode at the nearest sample to T_meas: the trap.
    std::size_t j = 0;
    double best = std::fabs(ts.front() - peak.t);
    for (std::size_t i = 1; i < ts.size(); ++i) {
      const double d = std::fabs(ts[i] - peak.t);
      if (d < best) {
        best = d;
        j = i;
      }
    }
    row.signed_rev = amp_signed[j];
  }
  return row;
}

} // namespace

TEST_CASE("discrete-velocity recurrence time matches 2 pi / (k dv) at three N_v",
          "[recurrence]") {
  if (world_size() != 1) {
    SKIP("recurrence table is a rank-local streaming problem");
  }

  const std::vector<int> levels{24, 32, 48};
  const int nvy = 16;
  std::vector<RecurrenceRow> rows;
  std::string table =
      "  nvx     dv       T_pred     T_meas     ratio    |rho|(0)  |rho|(T_R)\n";
  for (int nvx : levels) {
    const auto row = run_streaming(nvx, nvy, 1.3);
    rows.push_back(row);
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "  %3d  %.6f  %.6f  %.6f  %.4f  %.4e  %.4e\n", row.nvx,
                  row.dv, row.t_pred, row.t_meas, row.ratio, row.a0,
                  row.a_rev);
    table += buf;
  }
  INFO("recurrence table\n" << table);
  REQUIRE(rows.size() >= 3);

  std::vector<double> inv_dv, t_meas;
  for (const auto &row : rows) {
    REQUIRE(row.a0 > 0.0); // empty reduction is not a pass
    REQUIRE(std::isfinite(row.t_meas));
    REQUIRE(row.ratio == Approx(1.0).epsilon(0.05));
    // The cell-centred trap: signed mode flips at T_R, modulus comes back.
    REQUIRE(row.signed0 * row.signed_rev < 0.0);
    REQUIRE(row.a_rev == Approx(row.a0).epsilon(0.2));
    inv_dv.push_back(1.0 / row.dv);
    t_meas.push_back(row.t_meas);
  }

  const double k = rows.front().t_pred > 0.0
                       ? (2.0 * kPi) / (rows.front().t_pred * rows.front().dv)
                       : 0.0;
  const double slope = fitted_slope(inv_dv, t_meas);
  const double want = 2.0 * kPi / k;
  INFO("fitted T_meas vs 1/dv slope = " << slope << "  2 pi/k = " << want);
  REQUIRE(slope == Approx(want).epsilon(0.05));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
