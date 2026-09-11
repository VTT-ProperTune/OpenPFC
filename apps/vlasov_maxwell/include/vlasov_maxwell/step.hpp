// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file step.hpp
 * @brief The Strang step: transport, deposition, field update, and the
 *        adapter that joins the padded phase space to the flat moment sum.
 *
 * @details
 * ## The adapter is the dangerous part
 *
 * `phase_space.hpp` stores `f` in a **padded, x-fastest** brick: the halo
 * along `v_y` means every axis is inset by `hw`, and the fastest-varying
 * index is `x`. `moments.hpp` reduces through a `StridedDistribution` whose
 * convenience constructor assumes an **unpadded, v_y-fastest** slab. Those
 * two layouts are both contiguous and both three-dimensional, so a wrong
 * adapter does not crash, does not read out of bounds, and does not produce
 * obviously silly numbers -- it transposes the distribution and quietly
 * deposits the charge of a different plasma.
 *
 * @ref view_of therefore builds the strides explicitly rather than calling
 * `StridedDistribution::contiguous`, and points `data` at the **first owned
 * cell** so that the global indices the reduction walks in land where they
 * should. @ref check_adapter is a runtime assertion that it did: it plants
 * a value at a known `(i, j, k)` through the field's own accessor and reads
 * it back through the view. That check costs one field write and is worth
 * it, because this is the single place in the application where two
 * independently written components have to agree about memory.
 *
 * ## The splitting
 *
 * Strang, second order, with the exact steps on the outside:
 *
 *     advect_x (dt/2)
 *     deposit -> fields (dt)
 *     advect_vx (dt/2) ; advect_vy (dt) ; advect_vx (dt/2)
 *     advect_x (dt/2)
 *
 * The inner velocity triple is itself a Strang composition, so the whole
 * step is second order in `dt` and the two `advect_x` halves at the ends of
 * consecutive steps fuse in a long run -- which is not exploited here,
 * because the field update between them needs the moments of the half-
 * advected state, and correctness is worth more than one transform.
 *
 * ## The field half-step, and why Gauss is still only a diagnostic
 *
 * With `electrostatic = true` the transverse fields are held at zero and
 * `E_x` comes from Gauss. That path is a *reduction of this same stepper*,
 * selected at runtime; there is deliberately no second implementation, so
 * a Vlasov-Poisson result and a Vlasov-Maxwell result are produced by the
 * same transport, the same deposition and the same diagnostics.
 *
 * With `electrostatic = false`, `E_x` is advanced by Ampere from the
 * deposited `J_x`, and Gauss is *measured* rather than imposed. The two
 * agree only to the extent that the discrete scheme conserves charge, which
 * is the whole content of the identity in `parameters.hpp` and the reason
 * the residual is a reported column of every run.
 *
 * @see issue #84 for the specification
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <vlasov_maxwell/advect.hpp>
#include <vlasov_maxwell/diagnostics.hpp>
#include <vlasov_maxwell/maxwell.hpp>
#include <vlasov_maxwell/moments.hpp>
#include <vlasov_maxwell/parameters.hpp>
#include <vlasov_maxwell/phase_space.hpp>

namespace vlasov {

/**
 * @brief A `StridedDistribution` over the owned cells of a padded
 *        `PhaseField`.
 *
 * Strides for the padded, x-fastest layout of `PhaseSpace`:
 * `stride_x = 1`, `stride_vx = npx`, `stride_vy = npx * npy`, with
 * `npx = nx + 2 hw` and `npy = nvx + 2 hw`; and `data` offset to the first
 * owned cell, `hw + npx*(hw + npy*hw)`.
 *
 * The view's `operator()` subtracts `kbegin` from the `v_y` index but not
 * from `x` or `v_x` -- which is right, because those two axes are local in
 * full and their global indices already start at zero.
 */
[[nodiscard]] inline StridedDistribution view_of(const PhaseSpace &ps,
                                                 const PhaseField &f) {
  const int hw = ps.halo_width();
  const std::ptrdiff_t npx = static_cast<std::ptrdiff_t>(ps.nx()) + 2 * hw;
  const std::ptrdiff_t npy = static_cast<std::ptrdiff_t>(ps.nvx()) + 2 * hw;
  StridedDistribution v;
  v.nx = ps.nx();
  v.nvx = ps.nvx();
  v.kbegin = ps.vy_offset();
  v.kend = ps.vy_offset() + ps.nvy_local();
  v.stride_x = 1;
  v.stride_vx = npx;
  v.stride_vy = npx * npy;
  const std::ptrdiff_t first_owned = hw + npx * (hw + npy * hw);
  v.data = f.vec().data() + first_owned;
  return v;
}

/**
 * @brief Verify @ref view_of against the field's own accessor.
 *
 * Writes a marker at three well-separated owned cells -- including a corner
 * of the owned box on each axis, because a stride error that is invisible
 * at the origin is not invisible there -- reads them back through the view,
 * and restores the field. Throws on any mismatch.
 *
 * This exists because a transposed adapter is silent: both layouts are
 * contiguous bricks of the same size, so the wrong one reads valid memory
 * and deposits a plausible charge density for a plasma that is not the one
 * being simulated.
 */
inline void check_adapter(const PhaseSpace &ps, PhaseField &f) {
  const auto v = view_of(ps, f);
  const int i1 = ps.nx() - 1;
  const int j1 = ps.nvx() - 1;
  const int k0 = ps.vy_offset();
  const int k1 = ps.vy_offset() + ps.nvy_local() - 1;
  const int probes[4][3] = {
      {0, 0, k0}, {i1, 0, k0}, {0, j1, k0}, {i1, j1, k1}};
  double saved[4];
  for (int t = 0; t < 4; ++t) {
    saved[t] = f(probes[t][0], probes[t][1], probes[t][2] - ps.vy_offset());
  }
  for (int t = 0; t < 4; ++t) {
    f(probes[t][0], probes[t][1], probes[t][2] - ps.vy_offset()) =
        1.0 + static_cast<double>(t);
  }
  std::string bad;
  for (int t = 0; t < 4; ++t) {
    const double got = v(probes[t][0], probes[t][1], probes[t][2]);
    const double want = 1.0 + static_cast<double>(t);
    if (got != want) {
      bad += " probe" + std::to_string(t) + "(" +
             std::to_string(probes[t][0]) + "," + std::to_string(probes[t][1]) +
             "," + std::to_string(probes[t][2]) + ") got " +
             std::to_string(got) + " want " + std::to_string(want);
    }
  }
  for (int t = 0; t < 4; ++t) {
    f(probes[t][0], probes[t][1], probes[t][2] - ps.vy_offset()) = saved[t];
  }
  f.note_host_write();
  if (!bad.empty()) {
    throw std::runtime_error(
        "view_of does not agree with PhaseField indexing --"
        " the padded/flat adapter is transposed or mis-offset:" + bad);
  }
}

/// Everything the time loop needs that outlives one step.
struct Stepper {
  const SimParams *p{nullptr};
  PhaseSpace *ps{nullptr};
  SpectralLine1D line;
  FieldState fields;
  TransportWorkspace work;
  XShiftPlan xplan;
  std::vector<VelocityMoments> moments;
  Sources sources;
  /**
   * @brief `J_y` from the previous field update.
   *
   * The second-order Duhamel term is exact for a current that varies
   * linearly over the step, so it needs two samples. On the first step
   * there is only one, and `Jy_prev` is seeded equal to `J_y` -- which
   * makes that step ETD1 rather than ETD2. That is the right degradation
   * (a first-order error on one step of a second-order run) and it is
   * stated here because silently reusing an uninitialised previous current
   * would instead put a large first-order error in exactly the step whose
   * output every growth-rate fit starts from.
   */
  std::vector<double> Jy_prev;
  bool have_prev{false};
  GaussDiagnostic gauss;
  /// Largest halo width any step of the run actually needed. Reported, so
  /// that "the guard never fired" is a measurement and not a hope.
  int peak_halo_used{0};

  Stepper(const SimParams &params, PhaseSpace &space)
      : p(&space.params()), ps(&space), line(space.params().nx, space.params().Lx),
        fields(FieldState::zeros(space.params().nx)), xplan(space.params().nx) {
    // Bind to PhaseSpace's owned SimParams, not the constructor argument.
    // Callers that return a copied SimParams (make_landau in the HIP science
    // path) would otherwise leave p dangling; vector::at on the wreckage
    // is how that showed up on LUMI job 21943338.
    (void)params;
    work.resize_for_vx(space);
    work.resize_for_vy(space);
    moments.resize(space.params().species.size());
  }

  /// Deposit `rho` and `J` from the current state, and refresh the ledger
  /// inputs. Separated from @ref advance because the initial ledger row and
  /// the initial field solve both need it before any transport happens.
  void deposit_all() {
    ReductionOptions opt;
    opt.v_thermal = p->v_thermal;
    opt.comm = ps->comm();
    sources = Sources::zeros(p->nx);
    for (std::size_t s = 0; s < p->species.size(); ++s) {
      moments[s] = reduce_velocity(*p, view_of(*ps, ps->f(s)), opt);
      add_species(p->species[s], moments[s], sources);
    }
    apply_background(*p, sources);
  }

  /// Solve or advance the fields over @p dt from the current sources.
  void update_fields(double dt) {
    if (!p->self_consistent) {
      return; // stage 3: imposed fields, frozen
    }
    if (p->electrostatic) {
      const auto sol = solve_gauss(line, sources.rho, p->neutrality_tol);
      fields.Ex = sol.Ex;
      std::fill(fields.Ey.begin(), fields.Ey.end(), 0.0);
      std::fill(fields.Bz.begin(), fields.Bz.end(), 0.0);
    } else {
      ampere_ex(fields.Ex, sources.Jx, dt);
      if (!have_prev) {
        Jy_prev = sources.Jy;
        have_prev = true;
      }
      advance_transverse_etd2(line, fields.Ey, fields.Bz, sources.Jy, Jy_prev,
                              dt);
      Jy_prev = sources.Jy;
    }
    gauss = p->gauss_correction
                ? correct_divergence(line, fields.Ex, sources.rho, true)
                : gauss_residual(line, fields.Ex, sources.rho);
  }

  /// One Strang step. Assumes the sources are current for the state on
  /// entry, and leaves them current for the state on exit.
  void advance(double dt) {
    const double h = 0.5 * dt;
    for (std::size_t s = 0; s < p->species.size(); ++s) {
      note(advect_x(*ps, ps->f(s), h, xplan, work));
    }
    deposit_all();
    update_fields(dt);
    for (std::size_t s = 0; s < p->species.size(); ++s) {
      const double qm = p->species[s].qm();
      const auto Bz = effective_bz();
      note(advect_vx(*ps, ps->f(s), qm, h, fields.Ex, Bz, p->interp_order, work));
      note(advect_vy(*ps, ps->f(s), qm, dt, fields.Ey, Bz, p->interp_order, work));
      note(advect_vx(*ps, ps->f(s), qm, h, fields.Ex, Bz, p->interp_order, work));
    }
    for (std::size_t s = 0; s < p->species.size(); ++s) {
      note(advect_x(*ps, ps->f(s), h, xplan, work));
    }
    deposit_all();
  }

  /// `B_z` seen by the Lorentz force: the self-consistent field plus any
  /// imposed uniform `b_ext`. Stage 3 sets `b_ext` and switches the
  /// self-consistent update off, which makes this a constant.
  [[nodiscard]] std::vector<double> effective_bz() const {
    if (p->b_ext == 0.0) {
      return fields.Bz;
    }
    std::vector<double> b = fields.Bz;
    for (auto &v : b) v += p->b_ext;
    return b;
  }

private:
  void note(const TransportReport &r) {
    peak_halo_used = std::max(peak_halo_used, r.required_halo);
  }
};

/**
 * @brief The largest stable step, and which constraint set it.
 *
 * Three bound the step here and only two of them are the usual ones.
 *
 *  - The **light** CFL does not appear. The transverse pair is integrated
 *    exactly (`maxwell.hpp`), so `c dt/dx` is unbounded -- which for an
 *    explicit electromagnetic solver is the restriction one would normally
 *    expect to dominate, and its absence is the single biggest numerical
 *    result of this application.
 *  - The **spatial transport** CFL does not appear either, because
 *    `advect_x` is an exact shift on a periodic axis.
 *  - What is left is **accuracy**, not stability: the splitting is second
 *    order, so `dt` must resolve the plasma period, and the velocity shift
 *    must stay inside the halo. The plasma-period bound is the binding one
 *    in every benchmark here.
 */
[[nodiscard]] inline double step_limit(const SimParams &p, double qm_max,
                                       double e_max, double b_max,
                                       int halo) noexcept {
  // Resolve the plasma oscillation: omega_pe = 1 in these units.
  double lim = 0.25;
  // Keep the velocity shift inside the halo along the distributed axis.
  const double a_max = std::fabs(qm_max) * (e_max + p.v_max * b_max);
  if (a_max > 0.0) {
    const double usable = static_cast<double>(
        std::max(1, halo - (p.interp_order - 1) / 2 - 1));
    lim = std::fmin(lim, usable * p.dvy() / a_max);
  }
  return lim;
}

} // namespace vlasov
