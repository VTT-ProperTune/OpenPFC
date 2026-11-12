// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file lattice.hpp
 * @brief One description of the BCC and FCC lattices shared by seeds and
 *        spectral operators.
 *
 * @details
 * A one-mode PFC crystal is the sum of plane waves over the shortest
 * reciprocal-lattice vectors: six `{110}` vectors for BCC, four `<111>` for
 * FCC. The same lattice also fixes where the isotropic two-point correlation
 * must peak: the first shell `|k| = 1` for both lattices, plus the `{200}`
 * shell `|k| = 2/sqrt(3)` that stabilises FCC over BCC. Seeds read the
 * vectors; the tungsten and aluminum linear operators read the shells, so one
 * `lattice` choice drives both the initial condition and the operator.
 *
 * All vectors have unit length, i.e. the first peak sits at `|k| = 1`.
 */

#ifndef PFC_SIMULATION_LATTICE_HPP
#define PFC_SIMULATION_LATTICE_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pfc {

/** @brief Crystal lattice selectable from JSON/TOML (`"bcc"`, `"fcc"`). */
enum class Lattice { bcc, fcc };

/** @brief Most reciprocal vectors any supported lattice uses (BCC: six). */
inline constexpr std::size_t kMaxReciprocalVectors = 6;

[[nodiscard]] constexpr std::string_view to_string(Lattice lattice) noexcept {
  switch (lattice) {
  case Lattice::bcc: return "bcc";
  case Lattice::fcc: return "fcc";
  }
  return "bcc";
}

/** @throws std::invalid_argument for anything but `bcc` or `fcc`. */
[[nodiscard]] inline Lattice lattice_from_string(std::string_view name) {
  if (name == "bcc") return Lattice::bcc;
  if (name == "fcc") return Lattice::fcc;
  throw std::invalid_argument("unknown lattice '" + std::string(name) +
                              "' (expected bcc or fcc)");
}

namespace detail {

inline constexpr double kInvSqrt2 = 0.70710678118654752440;
inline constexpr double kInvSqrt3 = 0.57735026918962576451;

inline constexpr std::array<std::array<double, 3>, 6> kBccVectors{{
    {kInvSqrt2, kInvSqrt2, 0.0},
    {kInvSqrt2, 0.0, kInvSqrt2},
    {0.0, kInvSqrt2, kInvSqrt2},
    {kInvSqrt2, 0.0, -kInvSqrt2},
    {kInvSqrt2, -kInvSqrt2, 0.0},
    {0.0, kInvSqrt2, -kInvSqrt2},
}};

inline constexpr std::array<std::array<double, 3>, 4> kFccVectors{{
    {-kInvSqrt3, kInvSqrt3, kInvSqrt3},
    {kInvSqrt3, -kInvSqrt3, kInvSqrt3},
    {kInvSqrt3, kInvSqrt3, -kInvSqrt3},
    {kInvSqrt3, kInvSqrt3, kInvSqrt3},
}};

inline constexpr std::array<double, 1> kBccShells{1.0};
inline constexpr std::array<double, 2> kFccShells{1.0, 2.0 * kInvSqrt3};

} // namespace detail

/**
 * @brief Unit reciprocal vectors of the first shell (`{110}` BCC, `<111>` FCC).
 *
 * The order is the historical one used by the seed classes, so existing
 * pinned checksums do not move.
 */
[[nodiscard]] inline std::span<const std::array<double, 3>>
reciprocal_vectors(Lattice lattice) noexcept {
  switch (lattice) {
  case Lattice::fcc: return detail::kFccVectors;
  case Lattice::bcc: break;
  }
  return detail::kBccVectors;
}

/**
 * @brief Radii `|k|` of the correlation-kernel peaks, in units of the first
 *        shell (BCC: `{1}`; FCC: `{1, 2/sqrt(3)}`).
 */
[[nodiscard]] inline std::span<const double>
correlation_shells(Lattice lattice) noexcept {
  switch (lattice) {
  case Lattice::fcc: return detail::kFccShells;
  case Lattice::bcc: break;
  }
  return detail::kBccShells;
}

/**
 * @brief One-mode density `rho + 2 A sum_i cos(q_i . X)` over the given
 *        (possibly rotated) reciprocal vectors.
 */
[[nodiscard]] inline double
one_mode_density(std::span<const std::array<double, 3>> q,
                 const std::array<double, 3> &X, double rho,
                 double amplitude) noexcept {
  double u = rho;
  for (const auto &qi : q) {
    u += 2.0 * amplitude * std::cos(qi[0] * X[0] + qi[1] * X[1] + qi[2] * X[2]);
  }
  return u;
}

} // namespace pfc

#endif // PFC_SIMULATION_LATTICE_HPP
