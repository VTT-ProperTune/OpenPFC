// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file transform_functors.hpp
 * @brief Test functors for `pfc::core::transform`, shared by the host test
 *        and the device instantiation TU.
 */

#include <complex>

#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::test {

struct Square {
  OPENPFC_HD double operator()(double x) const { return x * x; }
};

struct Halve {
  OPENPFC_HD float operator()(float x) const { return 0.5F * x; }
};

/// Reaction-style term `grad + l1 * (phi - phi0) + l2 * phi (phi^2 - 1) + c`.
struct Reaction {
  double l1;
  double l2;
  double c;
  OPENPFC_HD double operator()(double grad, double phi, double phi0) const {
    return grad + l1 * (phi - phi0) + l2 * phi * (phi * phi - 1.0) + c;
  }
};

struct Weighted4 {
  OPENPFC_HD double operator()(double a, double b, double c, double d) const {
    return a + 2.0 * b + 3.0 * c + 4.0 * d;
  }
};

/// Complex input and output: `a * z` written with `.real()` / `.imag()`.
struct ScaleComplex {
  OPENPFC_HD std::complex<double> operator()(const std::complex<double> &z,
                                             double a) const {
    return {a * z.real(), a * z.imag()};
  }
};

/// Complex input, real output.
struct Abs2 {
  OPENPFC_HD double operator()(const std::complex<double> &z) const {
    return z.real() * z.real() + z.imag() * z.imag();
  }
};

} // namespace pfc::test
