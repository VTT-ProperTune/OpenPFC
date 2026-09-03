// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

/**
 * @file defaults.hpp
 * @brief Karma (2001) present-model isothermal dendrite with Al-Cu SI scales.
 *
 * Phase field φ = +1 (solid) / −1 (liquid). Thin-interface a1, a2 match
 * g'(φ) = (1−φ²)² as in Karma & Rappel, Phys. Rev. E 57, 4323 (1998).
 *
 * The original Karma 2001 protocol has no unique d₀ or D_L (W₀=τ₀=1). Physical
 * capillary length, liquid diffusivity, Γ, m_l and c_l^0 are taken from the
 * Al-Cu directional cases so length, time, velocity and undercooling can be
 * reported in SI. Partition k and supersaturation Ω stay at the Karma values.
 *
 * The advertised product is the present model of Karma, PRL 87, 115701 (2001):
 * A = β₀ = ε_k = 0, k = 0.15, ε_c = 0.02, Ω = 0.55. Optional extras (env or
 * `am`) restore solute trapping and/or uniform cooling. Trapping uses modified
 * antitrapping a'(φ)=a(1−A(1−φ²)), A = D_L/(V_D^{PF} W₀), and a2^α(A) with
 * partial drag α=0.38 (Pinomaa & Provatas, Acta Mater. 2019). Capillary and
 * kinetic anisotropies are a_s(n) and a_k(n). Local τ inverts
 *   β = a1 τ/(λ W) − a1 a2 (W/D_L) e^u
 * at W_s = W₀ a_s and β_k = β₀ a_k (Pinomaa, J. Cryst. Growth 532, 125418
 * (2020); thesis eq. (4.12)). τ₀ for Δt uses e^u = 1.
 *
 * Extra isothermal trapping uses a magnified V_D so k(V) moves at tip speeds
 * and β₀=4 s/m (Δ_k a few percent of Ω at V~7 mm/s, Δt=0.1 τ₀ still inside
 * the solute Fourier limit). The `am` protocol uses physical Al–Cu V_D and β₀.
 *
 * Glasner (2001) preconditioning: φ = tanh(ψ/√2). Cubic a_s(n) in the crystal
 * frame; n_lab is rotated by a Bunge ZXZ matrix; 2D is n_z = 0.
 */
namespace alloy_pf_karma2001_benchmark {

inline constexpr double kA1 = 0.8839;
inline constexpr double kA2 = 0.6267;
inline constexpr double kDragAlpha = 0.38;
inline constexpr double kW0 = 1.0;
inline constexpr double kTau0 = 1.0;
inline constexpr double kDx = 0.4;
inline constexpr double kDt = 0.008;
inline constexpr double kDxGlasner = 1.0;
inline constexpr double kDtGlasner = 0.02;
/** Cubic capillary anisotropy; equals Karma ε₄ in 2D (n_z = 0). */
inline constexpr double kEpsC = 0.02;
/**
 * Kinetic anisotropy strength for trapping/AM extras. a_k = 1 + 3ε_k − 4ε_k ∑ n_i⁴
 * so β is smallest on [100] (Pinomaa & Provatas, Acta Mater. 2019, eq. 13).
 * Present-model glasner runs set ε_k = 0 (β₀ = 0).
 */
inline constexpr double kEpsK = 0.12;
inline constexpr double kPartition = 0.15;
inline constexpr double kOmega = 0.55;
inline constexpr double kCl0 = 1.0;
inline constexpr double kSeedRadiusOverD0 = 22.0;
inline constexpr double kGradEps = 1.0e-8;
inline constexpr double kCMin = 1.0e-16;

/** Al-Cu physical scales from the directional FTA cases. */
inline constexpr double kD0Phys = 12.17e-9;   // m
inline constexpr double kDLPhys = 4.4e-9;     // m^2/s
inline constexpr double kGammaPhys = 2.41e-7; // K m
inline constexpr double kMlePhys = -5.3;      // K / at%
inline constexpr double kCloPhys = 4.5;       // at%
/**
 * Extra isothermal trapping only (not the PRL present model). Magnified vs
 * directional V_D^{PF}=2 m/s so V/V_D is large enough at the isothermal tip
 * (V ~ 7 mm/s) that k(V) moves. A = D_L/(V_D W) is O(1) at W₀=22 nm.
 */
inline constexpr double kVDPf = 0.15; // m/s
/**
 * Compromise vs directional β₀=0.1 s/m and the old isothermal 40 s/m.
 * At V~7 mm/s, Δ_k=β₀ V≈0.03 (~5% of Ω=0.55): enough to split V(t) from
 * β₀=0, while τ₀ is small enough that Δt=0.1 τ₀ stays under the solute
 * Fourier limit (Fo≈0.25). 40 s/m forced Δt=0.02 τ₀ and is already at max
 * physical Δt; lowering β₀ cannot give a 10× trap speedup.
 */
inline constexpr double kBeta0 = 4.0; // s/m
/** AM protocol: physical Al-Cu kinetics (not the magnified isothermal values). */
inline constexpr double kVDPfAm = 2.0;   // m/s
inline constexpr double kBeta0Am = 0.1;  // s/m
/** Initial cooling rate (K/s). With kTDecayAm > 0, Ṫ(t)=Ṫ₀ e^{−t/τ} so ΔT saturates. */
inline constexpr double kTdotAm = 1.0e7;
/** Decay time of Ṫ (s). ΔT_cool → Ṫ₀ τ (80 K at the defaults). 0 = linear Ṫ t. */
inline constexpr double kTDecayAm = 8.0e-6;
inline constexpr double kTendAm = 15.0e-6; // s
/** Physical box size (m). ~3.5 μm keeps W=5 nm laptop-feasible and the tip off the wall. */
inline constexpr double kLAm = 3.5e-6;
/**
 * Abort buffer on the far Neumann walls (x = L, y = L, and z = L in 3D).
 * The origin faces are symmetry planes for the quarter-seed and are not a stop.
 * 0 disables. Solid (φ ≥ 0) past this fraction of L, or the diffuse interface /
 * solute field reaching a far face, all count as a BC interaction.
 */
inline constexpr double kStopFrac = 0.80;
/** Far-face φ above this (liquid is −1) means the interface has reached the wall. */
inline constexpr double kWallPhiLiq = -0.99;
/** Relative |c − c_∞| on a far face that counts as a solute–wall interaction. */
inline constexpr double kWallCRel = 0.01;
/** Extra undercooling so a circular seed of radius r_seed barely grows (K). */
inline constexpr double kDTExtra = 0.05;
/** Dimensionless FDT strength F / W0^d. 0 disables noise. ∼10^{-3} is a mild interface rumble. */
inline constexpr double kNoiseF0 = 1.0e-3;
inline constexpr unsigned kNoiseSeed = 1u;

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

inline double antitrapping_a() noexcept { return 1.0 / (2.0 * std::sqrt(2.0)); }

inline double f_prime(double phi) noexcept { return phi * phi * phi - phi; }

inline double g_prime(double phi) noexcept {
  const double one_m_phi2 = 1.0 - phi * phi;
  return one_m_phi2 * one_m_phi2;
}

inline double h_of(double phi) noexcept { return phi; }

inline double q_of(double phi, double k) noexcept {
  const double h = h_of(phi);
  return (1.0 - phi) / (1.0 + k - (1.0 - k) * h);
}

inline double denom_c(double phi, double k) noexcept {
  return 1.0 + k - (1.0 - k) * h_of(phi);
}

inline double a_prime_trap(double phi, double a_at, double A_trap) noexcept {
  return a_at * (1.0 - A_trap * (1.0 - phi * phi));
}

/** a2^±(A) = 8/(5√2) (K̄ + F̄^±); a2^+ zero drag, a2^- full drag. */
inline double a2_pm(double A, bool full_drag) noexcept {
  const double pref = 8.0 / (5.0 * std::sqrt(2.0));
  const double Kbar = 0.0638 - 0.0505 * A;
  const double F0 = 0.5 * std::sqrt(2.0) * std::log(2.0);
  const double F = full_drag ? (F0 + 0.75 * std::sqrt(2.0) * A)
                             : (F0 - 0.25 * std::sqrt(2.0) * A);
  return pref * (Kbar + F);
}

/** Partial drag: a2^α = (1−α) a2^+ + α a2^-. α=0.38 from MD. A=0 recovers kA2. */
inline double a2_of_A(double A, double alpha = kDragAlpha) noexcept {
  return (1.0 - alpha) * a2_pm(A, false) + alpha * a2_pm(A, true);
}

inline double u_corr_from_eu(double eu) noexcept {
  // Pinomaa thesis eq. (4.12) / J. Crystal Growth 532, 125418 (2020): the a2
  // term in β(τ) is multiplied by e^u (Echebarria high-undercooling correction).
  return std::max(0.2, eu);
}

inline double k_cgm(double k, double V, double VD) noexcept {
  if (!(VD > 0.0)) {
    return k;
  }
  const double vvd = V / VD;
  return (k + vvd) / (1.0 + vvd);
}

inline Mat3 matmul(const Mat3 &A, const Mat3 &B) noexcept {
  Mat3 C{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return C;
}

inline Vec3 matvec(const Mat3 &A, const Vec3 &v) noexcept {
  return {A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
          A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
          A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]};
}

inline Mat3 transpose(const Mat3 &A) noexcept {
  return {{{A[0][0], A[1][0], A[2][0]},
           {A[0][1], A[1][1], A[2][1]},
           {A[0][2], A[1][2], A[2][2]}}};
}

inline Mat3 rot_x(double a) noexcept {
  const double c = std::cos(a);
  const double s = std::sin(a);
  return {{{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}};
}

inline Mat3 rot_z(double a) noexcept {
  const double c = std::cos(a);
  const double s = std::sin(a);
  return {{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}};
}

/** Bunge ZXZ: R maps crystal → lab, so n_crystal = Rᵀ n_lab. */
inline Mat3 bunge_crystal_to_lab(double phi1, double Phi, double phi2) noexcept {
  return matmul(rot_z(phi1), matmul(rot_x(Phi), rot_z(phi2)));
}

inline Vec3 fast_growth_lab(const Mat3 &R_crystal_to_lab) noexcept {
  /** Crystal [100] in lab coordinates (first column of R). */
  return {R_crystal_to_lab[0][0], R_crystal_to_lab[1][0], R_crystal_to_lab[2][0]};
}

struct CubicAniso {
  double a_s = 1.0;
  double a_k = 1.0;
  double jx = 0.0;
  double jy = 0.0;
  double jz = 0.0;
};

/**
 * Capillary a_s = 1 − 3ε_c + 4ε_c ∑ n_i⁴ (max on [100]).
 * Kinetic a_k = 1 + 3ε_k − 4ε_k ∑ n_i⁴ (min on [100]; Pinomaa β anisotropy).
 * Flux j = W₀² a_s |∇φ| [ a_s n + da_s − n (n·da_s) ].
 */
inline CubicAniso cubic_aniso_from_grad(double gx, double gy, double gz, double eps_c, double eps_k,
                                        double W0, const Mat3 &R_c2l) noexcept {
  CubicAniso out;
  const double q2 = gx * gx + gy * gy + gz * gz;
  if (q2 < 1.0e-30) {
    return out;
  }
  const double q = std::sqrt(q2);
  const Vec3 n_lab{gx / q, gy / q, gz / q};
  const Vec3 n_c = matvec(transpose(R_c2l), n_lab);
  const double n4 = n_c[0] * n_c[0] * n_c[0] * n_c[0] + n_c[1] * n_c[1] * n_c[1] * n_c[1] +
                    n_c[2] * n_c[2] * n_c[2] * n_c[2];
  out.a_s = 1.0 - 3.0 * eps_c + 4.0 * eps_c * n4;
  out.a_k = 1.0 + 3.0 * eps_k - 4.0 * eps_k * n4;
  const Vec3 das_c{16.0 * eps_c * n_c[0] * n_c[0] * n_c[0],
                   16.0 * eps_c * n_c[1] * n_c[1] * n_c[1],
                   16.0 * eps_c * n_c[2] * n_c[2] * n_c[2]};
  const Vec3 das = matvec(R_c2l, das_c);
  const double ndas = n_lab[0] * das[0] + n_lab[1] * das[1] + n_lab[2] * das[2];
  const double pref = W0 * W0 * out.a_s * q;
  out.jx = pref * (out.a_s * n_lab[0] + das[0] - n_lab[0] * ndas);
  out.jy = pref * (out.a_s * n_lab[1] + das[1] - n_lab[1] * ndas);
  out.jz = pref * (out.a_s * n_lab[2] + das[2] - n_lab[2] * ndas);
  return out;
}

struct Physics {
  double d0_over_W = 0.544;
  double W0 = kW0;
  double tau0 = kTau0;
  double dx = kDxGlasner;
  double dt = kDtGlasner;
  double eps_c = kEpsC;
  double eps_k = 0.0;
  double k = kPartition;
  double Omega = kOmega;
  double cl0 = kCl0;
  double D = 0.0;
  double lambda = 0.0;
  double d0 = 0.0;
  double a_at = 0.0;
  double A_trap = 0.0;
  double a2 = kA2;
  double alpha_drag = kDragAlpha;
  double VD_pf = 0.0;
  double beta0 = 0.0;
  double Gamma = kGammaPhys;
  double mle = kMlePhys;
  double clo_phys = kCloPhys;
  double dT_scale = 0.0;
  double c_inf = 0.0;
  double u_inf = 0.0;
  double r_seed = 0.0;
  /** Uniform cooling. 0 keeps the Karma Ω protocol. */
  double Tdot = 0.0;
  /** If >0, Ṫ(t)=Ṫ₀ exp(−t/τ) so the thermal drive saturates at Ṫ₀ τ. */
  double t_decay = 0.0;
  double dT_gt = 0.0;
  double dT_extra = 0.0;
  /** Bunge angles (radians): n_crystal = Rᵀ n_lab. 2D 45° is φ1=π/4, Φ=φ2=0. */
  double phi1 = 0.0;
  double Phi = 0.0;
  double phi2 = 0.0;
  /** If true, local τ uses e^u in the a2 term; if false, τ is frozen at e^u=1 (U=0). */
  bool tau_eu_local = true;
};

/**
 * Invert Pinomaa (2020) eq. (7) / thesis (4.12) for τ:
 *   β = a1 τ / (λ W) − a1 a2 W/D_L e^u
 * at W_s = W₀ a_s and β_k = β₀ a_k. e^u is the dilute 1+(1−k)U factor.
 */
inline double tau_aniso(const Physics &p, double W_s, double beta_k, double eu) noexcept {
  const double u_corr = u_corr_from_eu(eu);
  return (p.lambda * W_s / kA1) * (beta_k + kA1 * p.a2 * (W_s / p.D) * u_corr);
}

/** Recompute λ, trapping A, a2, isotropic τ₀ after W₀, V_D or β₀ changes. */
inline void refresh_derived(Physics &p) noexcept {
  p.lambda = kA1 * p.W0 / p.d0;
  p.a_at = antitrapping_a();
  p.A_trap = (p.VD_pf > 0.0) ? (p.D / (p.VD_pf * p.W0)) : 0.0;
  p.a2 = a2_of_A(p.A_trap, p.alpha_drag);
  p.tau0 = tau_aniso(p, p.W0, p.beta0, 1.0);
  p.dT_scale = std::abs(p.mle) * p.clo_phys * (1.0 - p.k);
  p.c_inf = p.cl0 * (1.0 - (1.0 - p.k) * p.Omega);
  p.u_inf = std::log(1.0 - (1.0 - p.k) * p.Omega);
  p.r_seed = kSeedRadiusOverD0 * p.d0;
  p.dT_gt = (p.r_seed > 0.0) ? (p.Gamma / p.r_seed) : 0.0;
}

inline void set_dx_over_W(Physics &p, double dx_over_W) noexcept {
  p.dx = dx_over_W * p.W0;
  p.dt = kDtGlasner * (dx_over_W / kDxGlasner) * p.tau0;
}

/** Δt = (dt/τ₀) · (Δx/W₀) · τ₀. Glasner default is dt/τ₀ = 0.02 at Δx = W₀. */
inline void set_dt_over_tau(Physics &p, double dt_over_tau) noexcept {
  p.dt = dt_over_tau * (p.dx / p.W0) * p.tau0;
}

inline double dt_over_tau_of(const Physics &p) noexcept {
  const double dxW = p.dx / p.W0;
  return (p.tau0 > 0.0 && dxW > 0.0) ? (p.dt / (p.tau0 * dxW)) : 0.0;
}

/** History stride so Δt* between samples is ≲ 8 (centered LS window is 80). */
inline int default_n_hist(const Physics &p) noexcept {
  const double dtt = dt_over_tau_of(p);
  const double dxw = (p.W0 > 0.0) ? (p.dx / p.W0) : 1.0;
  const double d0w = std::max(p.d0_over_W, 1.0e-12);
  const double tstar_step = dtt * dxw * kA1 * kA2 / (d0w * d0w * d0w);
  return std::max(1, static_cast<int>(std::lround(8.0 / std::max(tstar_step, 1.0e-9))));
}

inline Physics make_physics(double d0_over_W) noexcept {
  Physics p;
  p.d0_over_W = d0_over_W;
  p.d0 = kD0Phys;
  p.D = kDLPhys;
  p.W0 = p.d0 / d0_over_W;
  p.VD_pf = 0.0;
  p.beta0 = 0.0;
  p.eps_k = 0.0;
  refresh_derived(p);
  set_dx_over_W(p, kDxGlasner);
  return p;
}

inline Physics make_physics_w0(double W0) noexcept { return make_physics(kD0Phys / W0); }

/**
 * Spatially isothermal AM cooling: Ω=0 (no bulk supersaturation), physical V_D
 * and β₀, T chosen so a 2D circle of radius r_seed is just above GT equilibrium.
 */
inline void set_am_cooling(Physics &p, double Tdot, double dT_extra = kDTExtra) noexcept {
  p.Omega = 0.0;
  p.Tdot = Tdot;
  p.t_decay = kTDecayAm;
  p.dT_extra = dT_extra;
  p.VD_pf = kVDPfAm;
  p.beta0 = kBeta0Am;
  p.eps_k = kEpsK;
  refresh_derived(p);
  set_dx_over_W(p, p.dx / p.W0);
}

/** Imposed bulk undercooling T_L(c_∞) − T(t) in kelvin. */
inline double dT_thermal(const Physics &p, double t) noexcept {
  const double t0 = std::max(t, 0.0);
  const double cool = (p.t_decay > 0.0)
                          ? (p.Tdot * p.t_decay * (1.0 - std::exp(-t0 / p.t_decay)))
                          : (p.Tdot * t0);
  return p.dT_gt + p.dT_extra + cool;
}

/** Dimensionless thermal drive (eu − 1 − therm) from uniform T(t). 0 if Ṫ=0. */
inline double therm_drive(const Physics &p, double t) noexcept {
  if (!(p.Tdot > 0.0)) {
    return 0.0;
  }
  const double denom = std::abs(p.mle) * p.clo_phys;
  return (denom > 0.0) ? (dT_thermal(p, t) / denom) : 0.0;
}

inline Mat3 rotation_of(const Physics &p) noexcept {
  return bunge_crystal_to_lab(p.phi1, p.Phi, p.phi2);
}

/** φ = tanh(ψ/√2); invert with √2 artanh(φ). */
inline double phi_from_psi(double psi) noexcept { return std::tanh(psi / std::sqrt(2.0)); }

inline double dphi_dpsi_from_phi(double phi) noexcept {
  return (1.0 - phi * phi) / std::sqrt(2.0);
}

} // namespace alloy_pf_karma2001_benchmark
