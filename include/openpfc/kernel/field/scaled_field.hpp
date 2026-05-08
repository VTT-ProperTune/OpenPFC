// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file scaled_field.hpp
 * @brief Lightweight proxy returned by `operator*(double, field-like)`.
 *
 * @details
 * `pfc::field::ScaledField` is a tiny non-owning view that pairs a scalar
 * `alpha` with a contiguous `double` buffer. It exists solely so that
 * expressions like
 *
 *     u += dt * du;
 *
 * can be written in user code (compact-driver style) and dispatched to a
 * single axpy in `pfc::field::LocalField::operator+=`. The underlying
 * field types (`LocalField<double>`, `pfc::sim::DuField<G, Eval>`, ...)
 * each provide a `friend operator*(double, ...)` returning this proxy.
 *
 * Lifetime: the proxy is intended to be a transient temporary on the same
 * statement (`u += dt * du;`). It captures a raw pointer to the source
 * buffer and is **not** safe to store or pass across statements where the
 * source could be moved or resized.
 */

#include <cstddef>

namespace pfc::field {

/**
 * @brief View of a scaled contiguous `double` buffer (`alpha * data[0..size)`).
 *
 * Produced by `operator*(double, const LocalField<double>&)` and the
 * matching overload on `pfc::sim::DuField<G, Eval>`. Consumed by
 * `LocalField::operator+=(ScaledField)`.
 */
struct ScaledField {
  double alpha{0.0};
  const double *data{nullptr};
  std::size_t size{0};
};

} // namespace pfc::field
