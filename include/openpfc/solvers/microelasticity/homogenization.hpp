// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file homogenization.hpp
 * @brief Periodic FFT homogenization of a two-phase linear-elastic cell.
 *
 * @details
 * Given a scalar stiffness weight \f$h(\mathbf x)\in[0,1]\f$ on the
 * periodic grid of `EigenstrainMicroelasticity`, `PeriodicHomogenizer`
 * computes the homogenized stiffness \f$\mathbf C_H\f$ by six (3-D)
 * independent unit-cell solves:
 *
 * \f[
 *   \nabla\cdot\boldsymbol\sigma^{(\alpha)}=0,\qquad
 *   \boldsymbol\sigma^{(\alpha)}
 *     =\mathbf C(h):
 *       \bigl(\bar{\boldsymbol\varepsilon}^{(\alpha)}
 *             +\boldsymbol\varepsilon(\mathbf u^{\#(\alpha)})\bigr),
 * \qquad
 *   \langle\boldsymbol\sigma^{(\alpha)}\rangle
 *     =\mathbf C_H:\bar{\boldsymbol\varepsilon}^{(\alpha)}.
 * \f]
 *
 * The local modulus is the solver's linear blend
 * \f$\mathbf C(h)=h\mathbf C_s+(1-h)\mathbf C_l\f$; with a SIMP weight
 * \f$h=w(\phi)\f$ (`simp.hpp`) that is the SIMP law. Eigenstrain is
 * identically zero; the macroscopic strain is imposed through
 * `MicroelasticityParams::applied_strain`
 * (\f$\hat\varepsilon(\mathbf k=0)\f$). There is no second elasticity
 * implementation.
 *
 * The homogenizer takes the solver as a template parameter, the host
 * `EigenstrainMicroelasticity` by default or
 * `DeviceEigenstrainMicroelasticity<Space>` from `microelasticity_device.hpp`.
 *
 * ## Voigt convention
 *
 * `Sym3` in `microelasticity.hpp` stores **tensor** shear
 * (\f$\varepsilon_{yz}\f$, not \f$\gamma_{yz}=2\varepsilon_{yz}\f$). The
 * \f$6\times 6\f$ matrices in this header are **engineering Voigt**: they
 * act on \f$(\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},
 * \gamma_{23},\gamma_{13},\gamma_{12})\f$ with \f$\gamma=2\varepsilon\f$,
 * so a homogeneous isotropic material reports \f$C_{44}=c_{44}=\mu\f$ on
 * the diagonal, not \f$2\mu\f$. The unit loads for the three shears
 * therefore impose tensor strain \f$1/2\f$ (\f$\gamma=1\f$). Index order
 * matches `SymIndex`.
 *
 * ## Sensitivity
 *
 * Linear elastic homogenization is self-adjoint. After the six solves the
 * strain fields \f$\boldsymbol\varepsilon^{(\alpha)}\f$ are the adjoints,
 * and
 *
 * \f[
 *   \frac{\partial (C_H)_{\alpha\beta}}{\partial h_e}
 *   =\frac{1}{N}\,
 *     \boldsymbol\varepsilon^{(\alpha)}_e
 *     :\frac{\partial\mathbf C}{\partial h}
 *     :\boldsymbol\varepsilon^{(\beta)}_e
 * \f]
 *
 * with \f$\partial\mathbf C/\partial h=\mathbf C_s-\mathbf C_l\f$
 * (Hill–Mandel cancels the implicit strain derivatives). The tensor
 * objective \f$J=\tfrac12\lVert W\odot(C_H-C_{\mathrm{target}})\rVert_F^2\f$
 * then differentiates in closed form.
 *
 * @see Sigmund, *Int. J. Solids Struct.* **31**, 2313 (1994)
 * @see Moulinec & Suquet, *Comput. Methods Appl. Mech. Engrg.* **157**, 69 (1998)
 * @see Bendsøe & Sigmund, *Arch. Appl. Mech.* **69**, 635 (1999)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/solvers/microelasticity/microelasticity.hpp>
#include <openpfc/solvers/microelasticity/simp.hpp>

namespace pfc::solvers {

/// Engineering Voigt index order: 11, 22, 33, 23, 13, 12 (same as `SymIndex`).
inline constexpr int kVoigtDim = 6;

/**
 * @brief A \f$6\times 6\f$ engineering-Voigt matrix.
 *
 * Row/column \f$0,1,2\f$ are normal, \f$3,4,5\f$ are engineering shear.
 * The Frobenius product and Hadamard product used by the inverse-design
 * objective live here so a weight matrix and a stiffness share one type.
 */
struct Voigt6 {
  std::array<std::array<double, kVoigtDim>, kVoigtDim> a{};

  [[nodiscard]] double &operator()(int i, int j) noexcept {
    return a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
  }
  [[nodiscard]] double operator()(int i, int j) const noexcept {
    return a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
  }

  [[nodiscard]] static Voigt6 identity() noexcept {
    Voigt6 v;
    for (int i = 0; i < kVoigtDim; ++i) v(i, i) = 1.0;
    return v;
  }

  [[nodiscard]] static Voigt6 ones() noexcept {
    Voigt6 v;
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j) v(i, j) = 1.0;
    return v;
  }

  [[nodiscard]] Voigt6 transposed() const noexcept {
    Voigt6 t;
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j) t(i, j) = (*this)(j, i);
    return t;
  }

  /// \f$\tfrac12(A+A^T)\f$. Homogenized stiffness is symmetric; FFT round-off
  /// leaves a residual of that order which we do not want in \f$J\f$.
  [[nodiscard]] Voigt6 symmetrized() const noexcept {
    Voigt6 s;
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j)
        s(i, j) = 0.5 * ((*this)(i, j) + (*this)(j, i));
    return s;
  }

  [[nodiscard]] double frobenius_norm() const noexcept {
    double s = 0.0;
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j) {
        const double x = (*this)(i, j);
        s += x * x;
      }
    return std::sqrt(s);
  }

  [[nodiscard]] double max_abs() const noexcept {
    double m = 0.0;
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j) m = std::max(m, std::abs((*this)(i, j)));
    return m;
  }

  Voigt6 &operator+=(const Voigt6 &o) noexcept {
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j) (*this)(i, j) += o(i, j);
    return *this;
  }
  Voigt6 &operator-=(const Voigt6 &o) noexcept {
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j) (*this)(i, j) -= o(i, j);
    return *this;
  }
  Voigt6 &operator*=(double s) noexcept {
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j) (*this)(i, j) *= s;
    return *this;
  }
};

[[nodiscard]] inline Voigt6 operator+(Voigt6 a, const Voigt6 &b) noexcept {
  a += b;
  return a;
}
[[nodiscard]] inline Voigt6 operator-(Voigt6 a, const Voigt6 &b) noexcept {
  a -= b;
  return a;
}
[[nodiscard]] inline Voigt6 operator*(Voigt6 a, double s) noexcept {
  a *= s;
  return a;
}
[[nodiscard]] inline Voigt6 operator*(double s, Voigt6 a) noexcept {
  a *= s;
  return a;
}

[[nodiscard]] inline Voigt6 hadamard(const Voigt6 &a, const Voigt6 &b) noexcept {
  Voigt6 o;
  for (int i = 0; i < kVoigtDim; ++i)
    for (int j = 0; j < kVoigtDim; ++j) o(i, j) = a(i, j) * b(i, j);
  return o;
}

[[nodiscard]] inline double frobenius_inner(const Voigt6 &a,
                                            const Voigt6 &b) noexcept {
  double s = 0.0;
  for (int i = 0; i < kVoigtDim; ++i)
    for (int j = 0; j < kVoigtDim; ++j) s += a(i, j) * b(i, j);
  return s;
}

/// \f$\tfrac12\lVert W\odot(C-C_\ast)\rVert_F^2\f$.
[[nodiscard]] inline double tensor_mismatch(const Voigt6 &C, const Voigt6 &Cstar,
                                            const Voigt6 &W) noexcept {
  const Voigt6 r = hadamard(W, C - Cstar);
  return 0.5 * frobenius_inner(r, r);
}

/**
 * @brief Cubic `Stiffness` as an engineering Voigt matrix.
 *
 * Diagonal shear entries are \f$c_{44}\f$, not \f$2c_{44}\f$.
 */
[[nodiscard]] inline Voigt6 voigt_from_stiffness(const Stiffness &c) noexcept {
  Voigt6 v;
  v(0, 0) = v(1, 1) = v(2, 2) = c.c11;
  v(0, 1) = v(1, 0) = v(0, 2) = v(2, 0) = v(1, 2) = v(2, 1) = c.c12;
  v(3, 3) = v(4, 4) = v(5, 5) = c.c44;
  return v;
}

/// In-place Gauss–Jordan invert. Returns false if a pivot vanishes.
inline bool invert_voigt(Voigt6 &m) {
  Voigt6 inv = Voigt6::identity();
  for (int col = 0; col < kVoigtDim; ++col) {
    int piv = col;
    double best = std::abs(m(col, col));
    for (int r = col + 1; r < kVoigtDim; ++r) {
      const double a = std::abs(m(r, col));
      if (a > best) {
        best = a;
        piv = r;
      }
    }
    if (best == 0.0) return false;
    if (piv != col) {
      for (int c = 0; c < kVoigtDim; ++c) {
        std::swap(m(col, c), m(piv, c));
        std::swap(inv(col, c), inv(piv, c));
      }
    }
    const double den = m(col, col);
    for (int c = 0; c < kVoigtDim; ++c) {
      m(col, c) /= den;
      inv(col, c) /= den;
    }
    for (int r = 0; r < kVoigtDim; ++r) {
      if (r == col) continue;
      const double f = m(r, col);
      for (int c = 0; c < kVoigtDim; ++c) {
        m(r, c) -= f * m(col, c);
        inv(r, c) -= f * inv(col, c);
      }
    }
  }
  m = inv;
  return true;
}

[[nodiscard]] inline Voigt6 inverted(Voigt6 m) {
  if (!invert_voigt(m)) {
    throw std::invalid_argument("invert_voigt: singular 6x6 matrix");
  }
  return m;
}

/**
 * @brief Engineering unit strain for homogenization load \p alpha.
 *
 * \p alpha in \f$\{0,1,2\}\f$ imposes \f$\varepsilon_{ii}=1\f$.
 * \p alpha in \f$\{3,4,5\}\f$ imposes tensor shear \f$1/2\f$ so
 * \f$\gamma=1\f$ and the resulting mean-stress column is the engineering
 * Voigt column of \f$C_H\f$.
 */
[[nodiscard]] inline Sym3 engineering_unit_strain(int alpha) {
  if (alpha < 0 || alpha >= kVoigtDim) {
    throw std::invalid_argument(
        "engineering_unit_strain: load index must be in [0, 5]");
  }
  Sym3 e{};
  e[alpha] = (alpha < 3) ? 1.0 : 0.5;
  return e;
}

/// Outcome of one `PeriodicHomogenizer::compute`.
struct HomogenizationResult {
  Voigt6 stiffness{};
  std::array<Sym3, kVoigtDim> mean_stress{};
  std::array<Sym3, kVoigtDim> mean_strain{};
  std::array<MicroelasticityReport, kVoigtDim> reports{};
  double volume_fraction{0.0};
  int n_loads{kVoigtDim};

  [[nodiscard]] bool all_converged() const noexcept {
    return std::all_of(reports.begin(), reports.end(),
                       [](const MicroelasticityReport &r) { return r.converged; });
  }
};

namespace detail {

[[nodiscard]] inline double global_sum(double local, MPI_Comm comm) {
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global;
}

[[nodiscard]] inline double global_cell_count(const pfc::data::Field<double> &f) {
  const auto gs = f.global_size();
  return static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
         static_cast<double>(gs[2]);
}

} // namespace detail

/**
 * @brief Periodic two-phase homogenizer on a Green-operator elasticity solver.
 *
 * `Solver` is `EigenstrainMicroelasticity` (host) or
 * `DeviceEigenstrainMicroelasticity<Space>`; one implementation serves both,
 * because both return host copies of the strain and stress after `solve()`.
 * Construct once per `Domain`+FFT. `compute(h)` runs six unit-cell problems
 * and stores the strain fields for the mutual-energy sensitivity. Not a
 * general optimization framework: the only design variable is the scalar
 * stiffness weight already used by the solver.
 */
template <class Solver = EigenstrainMicroelasticity> class PeriodicHomogenizer {
public:
  using RealField = pfc::data::Field<double>;
  using SymRealFields = EigenstrainMicroelasticity::SymRealFields;
  using FFT = typename Solver::FFT;

  PeriodicHomogenizer(const pfc::Domain &domain, FFT &fft,
                      MicroelasticityParams params)
      : m_solver(domain, fft, with_homogenization_defaults(std::move(params))),
        m_amp(pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_n_local(m_amp.size()) {
    const auto box = fft.get_inbox_bounds();
    for (int a = 0; a < kVoigtDim; ++a) {
      m_strain[static_cast<std::size_t>(a)] = make_sym(domain, box);
    }
    std::fill(m_amp.vec().begin(), m_amp.vec().end(), 0.0);
    m_amp.note_host_write();
    m_n_global = detail::global_cell_count(m_amp);
  }

  [[nodiscard]] Solver &solver() noexcept { return m_solver; }
  [[nodiscard]] const Solver &solver() const noexcept { return m_solver; }

  /// Volume fraction \f$\langle h\rangle\f$ (cell average, not a quadrature).
  [[nodiscard]] double volume_fraction(const RealField &h) const {
    check_size(h, "h");
    double local = 0.0;
    const double *p = h.data();
    for (std::size_t i = 0; i < m_n_local; ++i) local += p[i];
    return detail::global_sum(local, comm()) / m_n_global;
  }

  /**
   * @brief Assemble \f$C_H\f$ from six periodic elasticity solves.
   *
   * On return `strain(alpha)` holds the total strain (macro + fluctuation)
   * of load \p alpha, which `objective_sensitivity` consumes. Warm-start
   * between loads is disabled: the six applied strains are orthogonal.
   */
  HomogenizationResult compute(const RealField &h) {
    check_size(h, "h");
    HomogenizationResult out;
    out.volume_fraction = volume_fraction(h);
    m_solver.params().warm_start = false;
    for (int a = 0; a < kVoigtDim; ++a) {
      const auto ai = static_cast<std::size_t>(a);
      m_solver.params().applied_strain = engineering_unit_strain(a);
      m_solver.reset();
      out.reports[ai] = m_solver.solve(h, m_amp);
      copy_sym(m_solver.strain(), m_strain[ai]);
      out.mean_strain[ai] = mean_tensor(m_solver.strain());
      out.mean_stress[ai] = mean_tensor(m_solver.stress());
      for (int i = 0; i < kVoigtDim; ++i)
        out.stiffness(i, a) = out.mean_stress[ai][i];
    }
    out.stiffness = out.stiffness.symmetrized();
    m_has_result = true;
    m_last = out;
    return out;
  }

  [[nodiscard]] const HomogenizationResult &last() const {
    if (!m_has_result) {
      throw std::logic_error(
          "PeriodicHomogenizer::last: compute() has not been called");
    }
    return m_last;
  }

  /// Total strain of load @p alpha from the last `compute()`.
  [[nodiscard]] const SymRealFields &strain(int alpha) const {
    if (alpha < 0 || alpha >= kVoigtDim) {
      throw std::invalid_argument("PeriodicHomogenizer::strain: load out of range");
    }
    if (!m_has_result) {
      throw std::logic_error(
          "PeriodicHomogenizer::strain: compute() has not been called");
    }
    return m_strain[static_cast<std::size_t>(alpha)];
  }

  [[nodiscard]] double objective(const HomogenizationResult &r, const Voigt6 &Cstar,
                                 const Voigt6 &W) const noexcept {
    return tensor_mismatch(r.stiffness, Cstar, W);
  }

  /**
   * @brief Discrete mutual-energy gradient of the tensor-mismatch objective.
   *
   * Writes \f$\partial J/\partial h_e\f$ into @p dJdh (same inbox as @p h).
   * Requires a preceding `compute(h)` on the same field. The linear blend
   * \f$\partial C/\partial h=C_s-C_l\f$ is used throughout.
   */
  void objective_sensitivity(const RealField &h, const Voigt6 &Cstar,
                             const Voigt6 &W, RealField &dJdh) const {
    check_size(h, "h");
    check_size(dJdh, "dJdh");
    if (!m_has_result) {
      throw std::logic_error(
          "PeriodicHomogenizer::objective_sensitivity: compute() has not been "
          "called");
    }
    const Stiffness dC = Stiffness::blend(m_solver.params().stiffness_at_one, 1.0,
                                          m_solver.params().stiffness_at_zero, -1.0);
    const Voigt6 &C = m_last.stiffness;
    Voigt6 dJdC;
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j)
        dJdC(i, j) = W(i, j) * W(i, j) * (C(i, j) - Cstar(i, j));

    std::array<std::array<const double *, kVoigtDim>, kVoigtDim> ep{};
    for (int a = 0; a < kVoigtDim; ++a)
      for (int c = 0; c < kVoigtDim; ++c)
        ep[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)] =
            m_strain[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)]
                .data();

    double *g = dJdh.data();
    for (std::size_t i = 0; i < m_n_local; ++i) {
      double acc = 0.0;
      for (int a = 0; a < kVoigtDim; ++a) {
        Sym3 ea;
        for (int c = 0; c < kVoigtDim; ++c)
          ea[c] = ep[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)][i];
        const Sym3 dCea = dC.contract(ea);
        for (int b = 0; b < kVoigtDim; ++b) {
          Sym3 eb;
          for (int c = 0; c < kVoigtDim; ++c)
            eb[c] = ep[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)][i];
          const double dCab = ddot(dCea, eb) / m_n_global;
          acc += dJdC(a, b) * dCab;
        }
      }
      g[i] = acc;
    }
    dJdh.note_host_write();
  }

  /// Directional derivative \f$\sum_e (\partial J/\partial h_e)\,\tilde h_e\f$.
  [[nodiscard]] double directional_derivative(const RealField &dJdh,
                                              const RealField &hdir) const {
    check_size(dJdh, "dJdh");
    check_size(hdir, "hdir");
    double local = 0.0;
    const double *g = dJdh.data();
    const double *d = hdir.data();
    for (std::size_t i = 0; i < m_n_local; ++i) local += g[i] * d[i];
    return detail::global_sum(local, comm());
  }

private:
  static MicroelasticityParams
  with_homogenization_defaults(MicroelasticityParams p) {
    p.eigenstrain_pattern = Sym3{};
    p.warm_start = false;
    return p;
  }

  static SymRealFields make_sym(const pfc::Domain &domain, const pfc::Box3i &box) {
    return SymRealFields{pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box)};
  }

  void copy_sym(const SymRealFields &src, SymRealFields &dst) const {
    for (int c = 0; c < kVoigtDim; ++c) {
      const auto ci = static_cast<std::size_t>(c);
      std::copy(src[ci].vec().begin(), src[ci].vec().end(), dst[ci].vec().begin());
      dst[ci].note_host_write();
    }
  }

  [[nodiscard]] Sym3 mean_tensor(const SymRealFields &f) const {
    Sym3 s{};
    for (int c = 0; c < kVoigtDim; ++c) {
      double local = 0.0;
      const double *p = f[static_cast<std::size_t>(c)].data();
      for (std::size_t i = 0; i < m_n_local; ++i) local += p[i];
      s[c] = detail::global_sum(local, comm()) / m_n_global;
    }
    return s;
  }

  void check_size(const RealField &f, const char *what) const {
    if (f.size() != m_n_local) {
      throw std::invalid_argument(std::string("PeriodicHomogenizer: '") + what +
                                  "' has the wrong local size");
    }
  }

  [[nodiscard]] MPI_Comm comm() const noexcept { return m_solver.params().comm; }

  Solver m_solver;
  RealField m_amp;
  std::array<SymRealFields, kVoigtDim> m_strain{};
  std::size_t m_n_local{0};
  double m_n_global{1.0};
  bool m_has_result{false};
  HomogenizationResult m_last{};
};

} // namespace pfc::solvers
