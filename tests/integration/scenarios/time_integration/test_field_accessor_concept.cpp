// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <openpfc/kernel/field/field_accessor_concept.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

using pfc::field::FieldAccessor;

void test_data_field_satisfies_concept() {
  static_assert(FieldAccessor<pfc::data::Field<double>>);
  static_assert(FieldAccessor<pfc::data::Field<float>>);
}

// std::vector<double> is deliberately NOT tested here: it has both size()
// and data()/data() const returning pointers convertible to void*/const
// void*, so it genuinely satisfies FieldAccessor as specified (this concept
// only checks storage access, not anything field-specific) -- asserting
// otherwise would be asserting something false about the concept's actual,
// correct behavior, not testing a real rejection case.
void test_non_field_types_rejected() {
  static_assert(!FieldAccessor<int>);
  // Verify that the new pfc::data::Field types satisfy the concept
  static_assert(FieldAccessor<pfc::data::Field<double>>);
  static_assert(FieldAccessor<pfc::data::Field<float>>);
}
