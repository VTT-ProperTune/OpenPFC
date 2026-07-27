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
 * single axpy in field operators. The underlying
 * field types (`pfc::data::Field<double>`, `PaddedBrick<double>`,
 * `pfc::sim::DuField<G, Eval>`, …) each provide a matching
 * `operator*(double, …)` returning this proxy.
 *
 * Lifetime: the proxy is intended to be a transient temporary on the same
 * statement (`u += dt * du;`).
 */

#include <cstddef>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>

namespace pfc::field {

/**
 * @brief View of a scaled contiguous `double` buffer.
 *
 * Produced by `operator*(double, const pfc::data::Field<double>&)` and
 * overloads on other field types. Consumed by field operators that
 * support scaled field addition.
 */
struct ScaledField {
    double alpha{0.0};
    FieldView<double> field_;

    const double* data() const noexcept { return field_.data(); }
    std::size_t size() const noexcept { return field_.size(); }
};

/**
 * @brief Build a `ScaledField` proxy from a scalar and a `pfc::data::Field<double>`.
 *
 * Enables compact-driver expressions such as
 *
 *     u += dt * du;
 *
 * The proxy is intended for use as a transient on a single statement.
 */
inline ScaledField operator*(double alpha, const pfc::data::Field<double> &f) noexcept {
  FieldView<double> view(
      f.data(),
      f.size(),
      f.local_size(),
      f.spacing(),
      f.origin()
  );
  return ScaledField{alpha, std::move(view)};
}

} // namespace pfc::field
