// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#if !defined(OpenPFC_ENABLE_HIP)
#error "alloy_pf_directional/device_step_hip.hpp requires HIP"
#endif

namespace alloy_pf_directional {

struct HipPhys {
  double ke, clo, W0, lambda, tau0, tau_beta, tau_a2, eps_c, eps_k, omega, G, mle, Vp, x_tl;
  double delta_iso, a_at, A_trap, DL, dt, dx, inv_dx, dV;
  double R[9];
  int n_dim;
  int n_grains;
  bool dim3;
  bool use_glasner;
  bool periodic_y;
  bool periodic_z;
  bool omega_zhong;
  double noise_F0;
  unsigned noise_seed;
  int shift_cells;
  int i0, j0, k0;
};

void alcu_grain_step_hip(const double *pf, const double *phi_self, const double *phi_other,
                         const double *c, double *dphi, int nx, int ny, int nz, int hw,
                         const HipPhys &P, int istep, int grain_id);

void alcu_euler_hip(double *psi, double *phi, double *dphi, double *psi2, double *phi2,
                    double *dphi2, int nx, int ny, int nz, int hw, double dt, bool glasner,
                    int n_grains);

void alcu_fill_eu_u_hip(double *eu, double *u, const double *phi1, const double *phi2,
                        const double *c, int nx, int ny, int nz, int hw, const HipPhys &P);

void alcu_fill_solute_nodal_hip(double *a_diff, double *a_at, double *beta, double *u,
                                const double *phi1, const double *phi2, const double *c,
                                const double *dphi1, const double *dphi2, const double *eu,
                                int nx, int ny, int nz, int hw, const HipPhys &P);

void alcu_solute_iso_hip(double *c, double *dc, const double *a_diff, const double *a_at,
                         const double *u, const double *beta, int nx, int ny, int nz, int hw,
                         const HipPhys &P);

void alcu_noflux_x_hip(double *f, int nx, int ny, int nz, int hw, bool lo, bool hi);

void alcu_shift_left_hip(double *f, int nx, int ny, int nz, int hw, double fill);

} // namespace alloy_pf_directional
