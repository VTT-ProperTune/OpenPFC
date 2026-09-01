// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

/**
 * @file defaults.hpp
 * @brief Dilute Al-Cu FTA two-grain model (Pinomaa et al., J. Crystal Growth 2020).
 *
 * Frozen temperature (Bridgman): T = T_l + G(x − x_s − V_p t), growth left → right.
 * x_s is the initial solidus; T = T_l there at t = 0. Solid (left) is colder;
 * liquid ahead is hotter; pulling cools a fixed x.
 * Directional BCs: no-flux in x; periodic in y (and in z in 3D).
 * Glasner default Δx = W₀. Cubic a_s(n) with Bunge rotation; 2D is n_z = 0.
 * Two order parameters φ_α ∈ [−1, 1] (α = 1, 2). Combined solid–liquid field
 * ψ = −1 + Σ_α (1+φ_α) = 1+φ₁+φ₂ (clamped), as in Zhong et al., Nat. Commun.
 * 16, 11698 (2025). Grain repulsion −ω φ̂_α φ̂_β² with φ̂=(1+φ)/2. ω(T) from a
 * stationary 1D grain boundary (same method as their Eq. 5; FTA interpolant).
 * Modified antitrapping a'(φ)=a(1−A(1−φ²)), A=D_L/(V_D^{PF} W₀).
 * Kinetic time uses partial-drag a2^α = (1−α) a2^+ + α a2^- (α=0.38 from MD)
 * with a2^± = 8/(5√2) (K̄ + F̄^±) from Pinomaa & Provatas (Acta Mater. 2019):
 * β(n)=a1 τ/(λ W) − a1 a2^α(A) (W/D_L)(1+(1−k_e)U).
 * Time step: min of ½ the Laplacian von Neumann limits for D_L and W₀²/τ₀,
 * 0.05 τ₀ (τ₀ at U=0), and 0.8 Δx/V_p (interface must not skip a cell at the
 * pulling speed; keeps the discrete ∂tφ that feeds antitrapping on-grid).
 * n_dim = 2 (default, Nz=1) or 3 (regular 3D brick).
 * Langevin noise on φ (Model A / FDT structure): ∂tφ += η, interface weight (1−φ²),
 * Euler–Maruyama ⟨ηη⟩ ∼ 2 (F/τ) / ΔV δ_{tt'} with F = F0 W0^d ([Karma & Rappel 1999](https://doi.org/10.1103/PhysRevE.60.3614)).
 *
 * Inactive-region strategy: moving window + optional 16³/32³ block skip.
 * Classic octree AMR is deferred until those are insufficient (see README).
 */
namespace alloy_pf_directional {

inline constexpr double kA1 = 0.8839;
inline constexpr double kA2 = 0.6267; // A=0 (equilibrium antitrapping) value; see a2_of_A
inline constexpr double kDragAlpha = 0.38; // partial drag from MD; 0 = zero drag, 1 = full drag
inline constexpr double kKe = 0.17;
inline constexpr double kMle = -5.3;          // K / at%
inline constexpr double kClo = 4.5;           // at%
inline constexpr double kGamma = 2.41e-7;     // K m
inline constexpr double kD0 = 12.17e-9;       // m
inline constexpr double kDL = 4.4e-9;         // m^2/s
inline constexpr double kBeta0 = 0.1;         // s/m
inline constexpr double kEpsC = 0.018;
inline constexpr double kEpsK = 0.12;
inline constexpr double kVDPf = 2.0;          // m/s
inline constexpr double kW0 = 5.0e-9;         // m
inline constexpr double kG = 5.0e6;           // K/m
inline constexpr double kVp = 0.3;            // m/s
inline constexpr double kDxOverW0 = 1.0;
inline constexpr int kNDim = 2;
inline constexpr double kDtCflSafety = 0.5;   // fraction of Laplacian von Neumann limit
inline constexpr double kDtOverTau = 0.05;    // cap as a fraction of τ0
inline constexpr double kDtIfaceSafety = 0.8; // fraction of Δx/V_p (one cell per pull step)
inline constexpr double kGradEps = 1.0e-8;
inline constexpr double kCMin = 1.0e-16;
/** Abort if φ or c is non-finite or |value| exceeds this (unphysical blow-up). */
inline constexpr double kFieldBlowAbs = 1.0e6;
inline constexpr double kSeedRadiusOverW0 = 20.0;
/** Laptop directional-solidification box (single grain, left → right). */
inline constexpr double kDsLx = 6.40e-6;           // m
inline constexpr double kDsLy = 0.80e-6;           // m
/** 0 = 2D (Nz=1). Set Lz > 0 or OPENPFC_ALCU_NDIM=3 for a 3D brick. */
inline constexpr double kDsLz = 0.0;               // m
inline constexpr double kDsTend = 80.0e-6;         // s  safety cap; stop at right wall
/** Moving-window defaults (still a regular grid). 0 disables. */
inline constexpr int kWindowNxDefault = 256;
inline constexpr double kWindowMarginLeft = 0.20e-6;  // m of leftover solid
inline constexpr double kWindowMarginRight = 0.40e-6; // m of liquid ahead
/** Block skip: 0 off; 16 or 32. Saves time, not memory. */
inline constexpr int kBlockSkipDefault = 0;
inline constexpr double kBlockSkipTolPhi = 1.0e-4;
inline constexpr double kBlockSkipTolC = 1.0e-4;
inline constexpr int kBlockSkipRefresh = 10;
/** Relative |c − c_∞| on a far (liquid) face that counts as a solute–wall hit. */
inline constexpr double kWallCRel = 0.01;
inline constexpr double kDsSeedDepth = 0.20e-6;    // m  planar front near left wall; T=Tl here at t=0
inline constexpr double kDsSeedBump = 0.05e-6;     // m
inline constexpr double kDsSeedBumpSigma = 0.08e-6; // m
/** Bicrystal: grain 1 at +θ, grain 2 at −θ (Bunge φ1). */
inline constexpr double kThetaDeg = 30.0;
/** Minimum liquid between the two φ=0 contours, in units of W0 (periodic y). */
inline constexpr double kBicrystalGapW0 = 16.0;
/** PNG/VTK stride by default; field log is 1/10 of this. Scaled with W0 in ds runs. */
inline constexpr int kIoEverySnapshot = 1000;
inline constexpr int kIoEveryLog = 100;
/** Dimensionless FDT strength F / W0^d. 0 disables noise. ∼10^{-3} is a mild interface rumble. */
inline constexpr double kNoiseF0 = 1.0e-3;
inline constexpr unsigned kNoiseSeed = 1u;

/** Directional I/O: finer W0 has many more Euler steps, so dump less often. */
inline int io_every_snapshot_for_w0(double W0) noexcept {
  if (W0 > 15.0e-9) {
    return 1000; // ~20 nm
  }
  if (W0 > 7.5e-9) {
    return 5000; // ~10 nm
  }
  if (W0 > 3.5e-9) {
    return 20000; // ~5 nm
  }
  return 50000; // ~2.5 nm and below
}

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

/** Equilibrium interpolation: e^u = 1 (U = 0) ⇒ c = c_l^o [1+k−(1−k)h(φ)]/2. */
inline double c_eq(double phi, double k, double clo) noexcept {
  return 0.5 * clo * denom_c(phi, k);
}

/**
 * Factor 1+(1−k_e)U in the β–τ relation. In the dilute e^u model,
 * e^u = 1+(1−k_e)U. Floor keeps τ from collapsing if c undershoots.
 */
inline double u_corr_from_eu(double eu) noexcept {
  return std::max(0.2, eu);
}

inline double phi_hat(double phi) noexcept { return 0.5 * (1.0 + phi); }

/** ψ = 1 + φ₁ + φ₂ (Zhong mapping, two grains), clamped to [−1, 1]. */
inline double phi_eff_two(double p1, double p2) noexcept {
  return std::max(-1.0, std::min(1.0, p1 + p2 + 1.0));
}

/**
 * Repulsive grain coupling in the φ_α residual (Zhong et al. 2025, Eq. 4):
 *   −ω φ̂_α Σ_{β≠α} φ̂_β² ,  φ̂ = (1+φ)/2.
 * Positive ω shrinks overlap where both grains would be solid.
 */
inline double grain_repulsion(double phi_self, double phi_other, double omega) noexcept {
  const double ho = phi_hat(phi_other);
  return -omega * phi_hat(phi_self) * ho * ho;
}

/**
 * ω from a 1D stationary grain boundary for this FTA/Karma interpolant.
 *
 * Zhong et al. fix ω so a GB with ψ = 1 (φ₂ = −φ₁, fully solid) has equal
 * well depths: multiply the 1D φ equation by dφ/dx and integrate. Their
 * Eq. 5 is for the Ji exponential-c free energy. Here
 *   τ ∂_t φ = ⋯ + φ − φ³ − [λ/(1−k)] g'(φ) (e^u − 1 − therm) − ω φ̂_α φ̂_β²
 * with g'(φ) = (1−φ²)², e^u from ψ, and therm = (T−T_l)/(m_l c_∞).
 * At the GB, ψ = 1 ⇒ e^u = 1 (solid equilibrium), and
 *   ∫_{-1}^{1} g'(φ) dφ = 16/15 ,  ∫ grain-shape dφ = 1/6 ,
 * so ω = (32/5) λ therm / (1−k_e). therm > 0 in the solid (T < T_l);
 * ω is clamped to 0 in the liquid. OPENPFC_ALCU_OMEGA overrides with a constant.
 */
inline double omega_zhong_fta(double lambda, double ke, double therm) noexcept {
  const double den = 1.0 - ke;
  if (!(den > 0.0) || !std::isfinite(lambda) || !std::isfinite(therm) || !(therm > 0.0)) {
    return 0.0;
  }
  return (32.0 / 5.0) * (lambda / den) * therm;
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

inline Mat3 bunge_crystal_to_lab(double phi1, double Phi, double phi2) noexcept {
  return matmul(rot_z(phi1), matmul(rot_x(Phi), rot_z(phi2)));
}

inline void cubic_aniso_from_grad(double gx, double gy, double gz, double eps_c, double W0,
                                  double tau0, const Mat3 &R_c2l, double &jx, double &jy,
                                  double &jz, double &tau, double &A) noexcept {
  const double q2 = gx * gx + gy * gy + gz * gz;
  if (q2 < 1.0e-30) {
    jx = 0.0;
    jy = 0.0;
    jz = 0.0;
    tau = tau0;
    A = 1.0;
    return;
  }
  const double q = std::sqrt(q2);
  const Vec3 n_lab{gx / q, gy / q, gz / q};
  const Vec3 n_c = matvec(transpose(R_c2l), n_lab);
  const double n4 = n_c[0] * n_c[0] * n_c[0] * n_c[0] + n_c[1] * n_c[1] * n_c[1] * n_c[1] +
                    n_c[2] * n_c[2] * n_c[2] * n_c[2];
  A = 1.0 - 3.0 * eps_c + 4.0 * eps_c * n4;
  const Vec3 dA_c{16.0 * eps_c * n_c[0] * n_c[0] * n_c[0],
                  16.0 * eps_c * n_c[1] * n_c[1] * n_c[1],
                  16.0 * eps_c * n_c[2] * n_c[2] * n_c[2]};
  const Vec3 dA = matvec(R_c2l, dA_c);
  const double ndA = n_lab[0] * dA[0] + n_lab[1] * dA[1] + n_lab[2] * dA[2];
  const double pref = W0 * W0 * A * q;
  jx = pref * (A * n_lab[0] + dA[0] - n_lab[0] * ndA);
  jy = pref * (A * n_lab[1] + dA[1] - n_lab[1] * ndA);
  jz = pref * (A * n_lab[2] + dA[2] - n_lab[2] * ndA);
  const double a_k = 1.0 - 3.0 * kEpsK + 4.0 * kEpsK * n4;
  tau = tau0 * A * A * a_k;
}

inline double phi_from_psi(double psi) noexcept { return std::tanh(psi / std::sqrt(2.0)); }

inline double dphi_dpsi_from_phi(double phi) noexcept {
  return (1.0 - phi * phi) / std::sqrt(2.0);
}

/** Signed distance (in units of W0) to a semicircle/hemisphere on the left wall x=0. */
inline void two_grain_seed_s(double x, double y, double z, double y1, double y2, double zmid,
                             double R, double W0, bool dim3, double &s1, double &s2) noexcept {
  const auto s_at = [&](double yc) {
    const double dy = y - yc;
    const double dz = dim3 ? (z - zmid) : 0.0;
    const double r = std::sqrt(x * x + dy * dy + dz * dz);
    return -(r - R) / W0;
  };
  s1 = s_at(y1);
  s2 = s_at(y2);
}

/** Seed centres at 1/4 and 3/4 of the periodic y-span (Ny cells). */
inline void two_grain_seed_ys(int Ny, double dx, double &y1, double &y2) noexcept {
  const double L = static_cast<double>(Ny) * dx;
  y1 = 0.25 * L;
  y2 = 0.75 * L;
}

/** Shrink R if needed so the φ=0 contours stay at least kBicrystalGapW0 apart. */
inline double two_grain_seed_radius(double R_want, int Ny, double dx, double W0) noexcept {
  const double dist = 0.5 * static_cast<double>(Ny) * dx;
  const double rmax = 0.5 * (dist - kBicrystalGapW0 * W0);
  if (!(rmax > 2.0 * W0)) {
    return std::min(R_want, 2.0 * W0);
  }
  return std::min(R_want, rmax);
}

/**
 * Independent Glasner/tanh seeds (no exclusive Voronoi cut).
 * A Voronoi partition puts a jump in ψ along the midline for all x and
 * seeds a spurious |∇ψ| in the liquid; keep both fields smooth instead.
 * Callers must keep the discs from overlapping (two_grain_seed_radius).
 */
inline void apply_two_grain_seed(double s1, double s2, bool glasner, double &p1, double &p2,
                                 double *psi1, double *psi2) noexcept {
  if (glasner) {
    *psi1 = std::max(-8.0, std::min(8.0, s1));
    *psi2 = std::max(-8.0, std::min(8.0, s2));
    p1 = phi_from_psi(*psi1);
    p2 = phi_from_psi(*psi2);
  } else {
    p1 = -std::tanh(-s1 / std::sqrt(2.0));
    p2 = -std::tanh(-s2 / std::sqrt(2.0));
  }
}

/** Glasner dψ/dt from the φ-residual grain term, capped to ~2 units of ψ per step. */
inline double grain_dpsi_dt(double grain, double tau, double phi, double dt) noexcept {
  const double dps = std::max(dphi_dpsi_from_phi(phi), 1.0e-12);
  double g = grain / (dps * tau);
  if (dt > 0.0) {
    const double cap = 2.0 / dt;
    g = std::max(-cap, std::min(cap, g));
  }
  return g;
}

struct Physics {
  double ke = kKe;
  double mle = kMle;
  double clo = kClo;
  double W0 = kW0;
  double d0 = kD0;
  double DL = kDL;
  double beta0 = kBeta0;
  double eps_c = kEpsC;
  double eps_k = kEpsK;
  double VD_pf = kVDPf;
  /** If true, ω(x,t) = (32/5) λ therm/(1−k) (Zhong 1D GB). Else use omega. */
  bool omega_zhong = true;
  double omega = 0.0;
  double G = kG;
  double Vp = kVp;
  double x_tl = 0.0; // x where T = Tl at t = 0 (initial solidus)
  /** Extra uniform Δ = (Tl−T)/((1−ke)|ml|c0). Used with G=0 for 1D quenches. */
  double delta_iso = 0.0;
  double dx = kW0;
  double dt = 0.0;
  int n_dim = kNDim;
  double dt_cfl_c = 0.0;
  double dt_cfl_phi = 0.0;
  double dt_cfl_iface = 0.0; // 0 = unused (V_p ≤ 0)
  double dt_tau = 0.0;
  double lambda = 0.0;
  double tau0 = 0.0;      // U = 0 reference (dt, meta)
  double tau_beta = 0.0;  // λ W0² β0 / a1
  double tau_a2 = 0.0;    // λ W0² a2(A) / D_L
  double a2 = kA2;        // a2^α(A_trap)
  double alpha_drag = kDragAlpha;
  double a_at = 0.0;
  double A_trap = 0.0;
  double r_seed = 0.0;
  /** Single crystal: [100] along +x. Bicrystal: ±θ via set_symmetric_misorientation. */
  double phi1_g1 = 0.0;
  double phi1_g2 = 0.0;
  double Phi_g1 = 0.0;
  double Phi_g2 = 0.0;
  double phi2_g1 = 0.0;
  double phi2_g2 = 0.0;
};

inline double dt_laplacian_cfl(double dx, double D, int n_dim) noexcept {
  return kDtCflSafety * (dx * dx) / (2.0 * static_cast<double>(n_dim) * D);
}

/** Forward-Euler bound: the φ=0 isosurface (and the T=T_l isotherm) must not
 *  jump more than one cell per step at speed V_p. Returns 0 if V_p ≤ 0. */
inline double dt_iface_cfl(double dx, double Vp) noexcept {
  if (!(Vp > 0.0) || !std::isfinite(Vp) || !(dx > 0.0)) {
    return 0.0;
  }
  return kDtIfaceSafety * dx / Vp;
}

inline void apply_dt_limits(Physics &p) noexcept {
  const double Dphi = (p.W0 * p.W0) / p.tau0;
  p.dt_cfl_c = dt_laplacian_cfl(p.dx, p.DL, p.n_dim);
  p.dt_cfl_phi = dt_laplacian_cfl(p.dx, Dphi, p.n_dim);
  p.dt_tau = kDtOverTau * p.tau0;
  p.dt_cfl_iface = dt_iface_cfl(p.dx, p.Vp);
  p.dt = std::min(p.dt_cfl_c, std::min(p.dt_cfl_phi, p.dt_tau));
  if (p.dt_cfl_iface > 0.0) {
    p.dt = std::min(p.dt, p.dt_cfl_iface);
  }
}

inline Physics make_physics(double W0 = kW0, double dx_over_W0 = kDxOverW0,
                            int n_dim = kNDim) noexcept {
  Physics p;
  p.W0 = W0;
  p.n_dim = n_dim;
  p.lambda = kA1 * p.W0 / p.d0;
  p.a_at = antitrapping_a();
  p.A_trap = p.DL / (p.VD_pf * p.W0);
  p.a2 = a2_of_A(p.A_trap, p.alpha_drag);
  p.dx = dx_over_W0 * p.W0;
  p.tau_beta = p.lambda * p.W0 * p.W0 * (p.beta0 / kA1);
  p.tau_a2 = p.lambda * p.W0 * p.W0 * (p.a2 / p.DL);
  p.tau0 = p.tau_beta + p.tau_a2;
  apply_dt_limits(p);
  p.r_seed = kSeedRadiusOverW0 * p.W0;
  return p;
}

/** Grain 1 at +θ, grain 2 at −θ (Bunge φ1, degrees). */
inline void set_symmetric_misorientation(Physics &p, double theta_deg) noexcept {
  const double th = theta_deg * std::acos(-1.0) / 180.0;
  p.phi1_g1 = th;
  p.phi1_g2 = -th;
}

inline double T_minus_Tl(const Physics &p, double x, double t) noexcept {
  return p.G * (x - p.x_tl - p.Vp * t);
}

inline double thermal_drive(const Physics &p, double x, double t) noexcept {
  // PDE uses (Tl−T)/(|ml| c0) = (1−ke) Δ_iso + Bridgman term (mle < 0).
  return (1.0 - p.ke) * p.delta_iso + T_minus_Tl(p, x, t) / (p.mle * p.clo);
}

/** ω used in the φ residual: Zhong local value, or a constant override. */
inline double omega_used(const Physics &p, double therm) noexcept {
  if (!p.omega_zhong) {
    return p.omega;
  }
  return omega_zhong_fta(p.lambda, p.ke, therm);
}

/** ω at the solidus (therm = 1−k_e): (32/5) λ. Logged in meta. */
inline double omega_at_solidus(const Physics &p) noexcept {
  return omega_zhong_fta(p.lambda, p.ke, 1.0 - p.ke);
}

/** Local τ(n,U) from β = a1 τ/(λ W) − a1 a2(A) (W/D_L)(1+(1−k_e)U). */
inline double tau_with_u_corr(double tau_as_U0, const Physics &p, double eu) noexcept {
  return tau_as_U0 * (p.tau_beta + p.tau_a2 * u_corr_from_eu(eu)) / p.tau0;
}

} // namespace alloy_pf_directional
