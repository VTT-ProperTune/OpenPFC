// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef SEEDFCC_HPP
#define SEEDFCC_HPP

#include <array>
#include <cmath>

/**
 * @brief SeedFCC
 *
 * This class is used to define a seed for the FCC crystal structure.
 *
 */
class SeedFCC {

private:
  typedef std::array<double, 3> vec3; // 3D vector
  typedef std::array<vec3, 3> mat3;   // 3x3 matrix
  typedef std::array<vec3, 6> vec36;  // 6x3 matrix
  typedef std::array<vec3, 2> vec32;  // 2x3 matrix

  const vec3 location_;
  const vec3 orientation_;
  const vec36 q_;
  const vec32 bbox_;
  const double rho_;
  const double radius_;
  const double amplitude_;

  mat3 yaw(double a) {
    double ca = cos(a);
    double sa = sin(a);
    return {vec3({ca, -sa, 0.0}), vec3({sa, ca, 0.0}), vec3({0.0, 0.0, 1.0})};
  }

  mat3 pitch(double b) {
    double cb = cos(b);
    double sb = sin(b);
    return {vec3({cb, 0.0, sb}), vec3({0.0, 1.0, 0.0}), vec3({-sb, 0.0, cb})};
  }

  mat3 roll(double c) {
    double cc = cos(c);
    double sc = sin(c);
    return {vec3({1.0, 0.0, 0.0}), vec3({0.0, cc, -sc}), vec3({0.0, sc, cc})};
  }

  mat3 mult3(const mat3 &A, const mat3 &B) {
    mat3 C = {0};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  vec3 mult3(const mat3 &A, const vec3 &b) {
    vec3 c = {0};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        c[i] += A[i][j] * b[j];
      }
    }
    return c;
  }

  /**
   * @brief Rotate a seed. This is used to orient the seed in the crystal. The
   * seed is rotated by the Euler angles.
   *
   * @param orientation Euler angles
   * @return vec36
   */
  vec36 rotate(const vec3 &orientation) {
    const double s = 1.0 / sqrt(3.0);
    const vec3 q1 = {-s, s, s};
    const vec3 q2 = {s, -s, s};
    const vec3 q3 = {s, s, -s};
    const vec3 q4 = {s, s, s};
    mat3 Ra = yaw(orientation[0]);
    mat3 Rb = pitch(orientation[1]);
    mat3 Rc = roll(orientation[2]);
    mat3 R = mult3(Ra, mult3(Rb, Rc));
    const vec36 q = {mult3(R, q1), mult3(R, q2), mult3(R, q3), mult3(R, q4)};
    return q;
  }

  /**
   * @brief Get the bounding box of a seed.
   *
   * @param location
   * @param radius
   * @return vec32
   */
  vec32 bounding_box(const vec3 &location, double radius) {
    const vec3 low = {location[0] - radius, location[1] - radius, location[2] - radius};
    const vec3 high = {location[0] + radius, location[1] + radius, location[2] + radius};
    const vec32 bbox = {low, high};
    return bbox;
  }

  /**
   * @brief Check if a point is inside the bounding box of a seed.
   *
   * @param location
   * @return true
   * @return false
   */
  inline bool is_inside_bbox(const vec3 &location) const {
    const vec32 bbox = get_bbox();
    return (location[0] > bbox[0][0]) && (location[0] < bbox[1][0]) && (location[1] > bbox[0][1]) &&
           (location[1] < bbox[1][1]) && (location[2] > bbox[0][2]) && (location[2] < bbox[1][2]);
  }

  double get_radius() const { return radius_; }
  double get_rho() const { return rho_; }
  double get_amplitude() const { return amplitude_; }
  vec3 get_location() const { return location_; }
  vec36 get_q() const { return q_; }
  vec32 get_bbox() const { return bbox_; }

public:
  /**
   * @brief Construct a new SeedFCC object.
   *
   * @param location Cartesian coordinates
   * @param orientation Euler angles
   * @param radius Radius of the seed
   * @param rho Density of the seed
   * @param amplitude Amplitude of the seed
   */
  SeedFCC(const vec3 &location, const vec3 &orientation, const double radius, const double rho, const double amplitude)
      : location_(location), orientation_(orientation), q_(rotate(orientation)), bbox_(bounding_box(location, radius)),
        rho_(rho), radius_(radius), amplitude_(amplitude) {}

  /**
   * @brief Check if a point is inside the seed.
   *
   * @param X
   * @return true
   * @return false
   */
  bool is_inside(const vec3 &X) const {
    const vec3 Y = get_location();
    double x = X[0] - Y[0];
    double y = X[1] - Y[1];
    double z = X[2] - Y[2];
    double r = get_radius();
    return x * x + y * y + z * z < r * r;
  }

  /**
   * @brief Get the value of the seed at a given location.
   *
   * @param location
   * @return double
   */
  double get_value(const vec3 &location) const {
    double x = location[0];
    double y = location[1];
    double z = location[2];
    double u = get_rho();
    double a = get_amplitude();
    vec36 q = get_q();
    for (int i = 0; i < 4; i++) {
      u += 2.0 * a * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
    }
    return u;
  }
}; // SeedFCC

#endif // SEEDFCC_HPP
