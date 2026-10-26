// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/field/padded_face_flux.hpp>

TEST_CASE("padded face flux validates geometry and preserves orientation",
          "[field][padded_face_flux]") {
  const auto domain = pfc::domain::with_spacing({3, 2, 2}, {0.5, 0.7, 0.9});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> a(domain, box, 1), p(domain, box, 2);
  for (int k = -2; k < 4; ++k)
    for (int j = -2; j < 4; ++j)
      for (int i = -2; i < 5; ++i) p(i, j, k) = 3.0 * i - 2.0 * j + k;
  for (int k = -1; k < 3; ++k)
    for (int j = -1; j < 3; ++j)
      for (int i = -1; i < 4; ++i) a(i, j, k) = 2.0;
  const auto flux = pfc::field::fd::padded_face_flux(a, p);
  CHECK(flux(0, -1, 0, 0) == 12.0);
  CHECK(flux(1, 0, -1, 0) == -4.0 / 0.7);
  CHECK(flux(2, 0, 0, -1) == 2.0 / 0.9);
  pfc::data::Field<double> unpadded(domain, box, 0);
  CHECK_THROWS_AS(pfc::field::fd::padded_face_flux(a, unpadded),
                  std::invalid_argument);
  const auto other = pfc::domain::with_spacing({3, 2, 2}, {1, 1, 1});
  pfc::data::Field<double> mismatch(other, box, 1);
  CHECK_THROWS_AS(pfc::field::fd::padded_face_flux(a, mismatch),
                  std::invalid_argument);
}
