// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file transform_functors.inc
 * @brief Device instantiations of the transform test functors (stamped into
 *        transform_functors.cu / .hip). One TU per consumer target.
 */

#include <complex>

#include "transform_functors.hpp"
#include <openpfc/runtime/gpu/transform_gpu.hpp>

using ComplexD = std::complex<double>;

OPENPFC_INSTANTIATE_TRANSFORM(pfc::test::Square, double, double)
OPENPFC_INSTANTIATE_TRANSFORM(pfc::test::Halve, float, float)
OPENPFC_INSTANTIATE_TRANSFORM(pfc::test::Reaction, double, double, double, double)
OPENPFC_INSTANTIATE_TRANSFORM(pfc::test::Weighted4, double, double, double, double,
                              double)
OPENPFC_INSTANTIATE_TRANSFORM(pfc::test::ScaleComplex, ComplexD, ComplexD, double)
OPENPFC_INSTANTIATE_TRANSFORM(pfc::test::Abs2, double, ComplexD)
