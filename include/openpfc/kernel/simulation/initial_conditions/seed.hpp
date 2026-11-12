// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file seed.hpp
 * @brief Helper class for constructing spherical seed initial conditions
 *
 * @details
 * This file defines the Seed class, a helper for creating spherical crystalline
 * seeds with specific orientations. The Seed class:
 * - Generates density profiles for spherical regions
 * - Supports crystallographic orientations (BCC, FCC, etc.)
 * - Handles coordinate transformations (yaw, pitch, roll)
 * - Used by SingleSeed, SeedGrid, and RandomSeeds classes
 *
 * The Seed class is a building block for various seeding strategies in
 * solidification simulations.
 *
 * @see single_seed.hpp for single seed placement
 * @see seed_grid.hpp for regular grid of seeds
 * @see random_seeds.hpp for random seed distribution
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_INITIAL_CONDITIONS_SEED_HPP
#define PFC_INITIAL_CONDITIONS_SEED_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include <openpfc/kernel/simulation/lattice.hpp>

namespace pfc {

/**
 * @brief Seed is a helper class to construct various of initial conditions.
 *
 */
class Seed {
private:
  using vec3 = std::array<double, 3>;
  using mat3 = std::array<vec3, 3>;
  using vec36 = std::array<vec3, kMaxReciprocalVectors>;

  const vec3 location_;
  const vec3 orientation_;
  const Lattice lattice_;
  const std::size_t nq_;
  const vec36 q_;
  const double rho_;
  const double radius_;
  const double amplitude_;

  static mat3 yaw(double a) {
    double ca = cos(a);
    double sa = sin(a);
    return {vec3({ca, -sa, 0.0}), vec3({sa, ca, 0.0}), vec3({0.0, 0.0, 1.0})};
  }

  static mat3 pitch(double b) {
    double cb = cos(b);
    double sb = sin(b);
    return {vec3({cb, 0.0, sb}), vec3({0.0, 1.0, 0.0}), vec3({-sb, 0.0, cb})};
  }

  static mat3 roll(double c) {
    double cc = cos(c);
    double sc = sin(c);
    return {vec3({1.0, 0.0, 0.0}), vec3({0.0, cc, -sc}), vec3({0.0, sc, cc})};
  }

  static mat3 mult3(const mat3 &A, const mat3 &B) {
    mat3 C = {vec3{0.0}};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  static vec3 mult3(const mat3 &A, const vec3 &b) {
    vec3 c = {};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        c[i] += A[i][j] * b[j];
      }
    }
    return c;
  }

  /// The lattice's first-shell reciprocal vectors rotated by the Euler
  /// angles (yaw, pitch, roll); unused slots stay zero.
  static vec36 rotate(const vec3 &orientation, Lattice lattice) {
    mat3 Ra = yaw(orientation[0]);
    mat3 Rb = pitch(orientation[1]);
    mat3 Rc = roll(orientation[2]);
    mat3 R = mult3(Ra, mult3(Rb, Rc));
    vec36 q = {vec3{0.0, 0.0, 0.0}};
    std::size_t i = 0;
    for (const vec3 &qi : pfc::reciprocal_vectors(lattice)) {
      q[i++] = mult3(R, qi);
    }
    return q;
  }

  double get_radius() const { return radius_; }
  double get_rho() const { return rho_; }
  double get_amplitude() const { return amplitude_; }
  vec3 get_location() const { return location_; }
  vec36 get_q() const { return q_; }

public:
  /**
   * @param lattice BCC (six `{110}` waves, default) or FCC (four `<111>`).
   */
  Seed(const vec3 &location, const vec3 &orientation, const double radius,
       const double rho, const double amplitude, Lattice lattice = Lattice::bcc)
      : location_(location), orientation_(orientation), lattice_(lattice),
        nq_(pfc::reciprocal_vectors(lattice).size()),
        q_(rotate(orientation_, lattice)), rho_(rho), radius_(radius),
        amplitude_(amplitude) {}

  Lattice get_lattice() const { return lattice_; }

  /// Rotated reciprocal vectors actually in use (six for BCC, four for FCC).
  std::span<const vec3> reciprocal_vectors() const {
    return std::span<const vec3>(q_.data(), nq_);
  }

  bool is_inside(const vec3 &X) const {
    const vec3 Y = get_location();
    double x = X[0] - Y[0];
    double y = X[1] - Y[1];
    double z = X[2] - Y[2];
    double r = get_radius();

    return x * x + y * y + z * z < r * r;
  }

  double get_value(const vec3 &location) const {
    return one_mode_density(reciprocal_vectors(), location, get_rho(),
                            get_amplitude());
  }
};
} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_SEED_HPP
