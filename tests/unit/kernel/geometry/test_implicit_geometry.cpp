// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/geometry/cut_cell.hpp>

#include <cmath>

using Catch::Approx;
using pfc::Domain;
using pfc::geometry::absd;
using pfc::geometry::Box;
using pfc::geometry::CellKind;
using pfc::geometry::classify_box;
using pfc::geometry::classify_general;
using pfc::geometry::classify_layout;
using pfc::geometry::Family;
using pfc::geometry::hex_tet_corner;
using pfc::geometry::identity_rotation;
using pfc::geometry::Implicit;
using pfc::geometry::integrate_box;
using pfc::geometry::is_signed_distance;
using pfc::geometry::layout_from_domain;
using pfc::geometry::make_axis_aligned_ellipsoid;
using pfc::geometry::make_ellipse;
using pfc::geometry::make_ellipsoid;
using pfc::geometry::make_plane;
using pfc::geometry::make_sphere;
using pfc::geometry::measure_box;
using pfc::geometry::measure_domain;
using pfc::geometry::numerical_gradient;
using pfc::geometry::Policy;
using pfc::geometry::poly_box;
using pfc::geometry::poly_volume;
using pfc::geometry::rotation_about;
using pfc::geometry::sample_corners;
using pfc::geometry::Side;
using pfc::geometry::side_of;
using pfc::geometry::Vec3;
using pfc::geometry::wrap_coordinate;

static constexpr double kPi = 3.14159265358979323846;

TEST_CASE("box polyhedron has unit volume", "[geometry][poly]") {
  const double volume = poly_volume(poly_box(Box{{0, 0, 0}, {1, 1, 1}}));
  REQUIRE(volume == Approx(1.0).epsilon(1e-12));
  double sum = 0;
  for (int t = 0; t < 6; ++t) {
    Vec3 p[4];
    for (int i = 0; i < 4; ++i) {
      const int id = hex_tet_corner(t, i);
      p[i] = {(id & 1) ? 1.0 : 0.0, (id & 2) ? 1.0 : 0.0, (id & 4) ? 1.0 : 0.0};
    }
    sum += absd(pfc::geometry::triple(p[1] - p[0], p[2] - p[0], p[3] - p[0])) / 6.0;
  }
  REQUIRE(sum == Approx(1.0).epsilon(1e-12));
}

TEST_CASE("plane and sphere are signed distances; ellipsoid polynomial is not",
          "[geometry][implicit]") {
  const Implicit plane = make_plane({0.2, -0.4, 1.5}, {0, 2, 0});
  REQUIRE(is_signed_distance(plane.family));
  REQUIRE(plane.value({0.2, -0.4, 1.5}) == Approx(0).margin(1e-14));
  REQUIRE(plane.value({0.2, 0.6, 1.5}) == Approx(1).epsilon(1e-12));
  const Implicit sphere = make_sphere({1, 2, 3}, 1.5);
  REQUIRE(sphere.value({1, 2, 3}) == Approx(-1.5));
  REQUIRE(sphere.value({2.5, 2, 3}) == Approx(0).margin(1e-12));
  const Implicit ellipsoid = make_axis_aligned_ellipsoid({0, 0, 0}, {2, 1, 0.5});
  REQUIRE(ellipsoid.family == Family::Ellipsoid);
  REQUIRE_FALSE(is_signed_distance(ellipsoid.family));
  REQUIRE(ellipsoid.value({2, 0, 0}) == Approx(0).margin(1e-12));
  REQUIRE(ellipsoid.value({0, 0, 0}) == Approx(-1));
  // The polynomial at a point one semi-axis beyond the tip is not distance 1.
  REQUIRE(ellipsoid.value({4, 0, 0}) == Approx(3));
}

TEST_CASE("rotated ellipsoid uses the principal-axis transformation",
          "[geometry][implicit]") {
  const auto rotation = rotation_about({0, 0, 1}, kPi / 2);
  const Implicit shape = make_ellipsoid({1, -2, 4}, {3, 1, 2}, rotation);
  // Local +x maps to world +y, so the long axis ends at center+(0,3,0).
  REQUIRE(shape.value({1, 1, 4}) == Approx(0).margin(1e-10));
  REQUIRE(shape.value({2, -2, 4}) == Approx(0).margin(1e-10));
  const Vec3 probe{1.4, -1.2, 4.3};
  const Vec3 numeric =
      numerical_gradient([&](Vec3 x) { return shape.value(x); }, probe, 1e-6);
  const Vec3 exact = shape.gradient(probe);
  REQUIRE(numeric.x == Approx(exact.x).margin(1e-6));
  REQUIRE(numeric.y == Approx(exact.y).margin(1e-6));
  REQUIRE(numeric.z == Approx(exact.z).margin(1e-6));
  const auto centre = shape.unit_normal(shape.point, {});
  REQUIRE(centre.degenerate == 1);
}

TEST_CASE("reflection is rejected and a 90 degree axis rotation is orthonormal",
          "[geometry][rotation]") {
  const double reflect[9] = {1, 0, 0, 0, 1, 0, 0, 0, -1};
  REQUIRE_THROWS_AS(pfc::geometry::rotation_from_matrix(reflect),
                    std::invalid_argument);
  const auto rotation = rotation_about({0, 1, 0}, kPi / 2);
  const Vec3 image = rotation.apply({1, 0, 0});
  REQUIRE(image.x == Approx(0).margin(1e-12));
  REQUIRE(image.y == Approx(0).margin(1e-12));
  REQUIRE(image.z == Approx(-1).margin(1e-12));
}

TEST_CASE("corner signs do not prove that a sphere misses a cell",
          "[geometry][classify]") {
  const Implicit sphere = make_sphere({0.5, 0.5, 0.5}, 0.1);
  const Box cell{{0, 0, 0}, {1, 1, 1}};
  const auto corners = sample_corners(sphere, cell, {});
  REQUIRE(corners.inside == 0);
  REQUIRE(corners.outside == 8);
  const auto verdict = classify_box(sphere, cell, {});
  REQUIRE(verdict.kind == CellKind::Cut);
  const auto general = classify_general([&](Vec3 x) { return sphere.value(x); },
                                        [&](Box) { return 1.0; }, cell, {}, 0);
  REQUIRE(general.kind == CellKind::Cut);
  REQUIRE(general.unresolved == 1);
  const auto proven = classify_general([&](Vec3 x) { return sphere.value(x); },
                                       [&](Box) { return 1.0; }, cell, {}, 6);
  REQUIRE(proven.kind == CellKind::Cut);
  REQUIRE(proven.unresolved == 0);
}

TEST_CASE("translated anisotropic cells classify a tangent sphere",
          "[geometry][classify]") {
  Domain domain;
  domain.size = {5, 4, 3};
  domain.spacing = {0.4, 0.5, 0.25};
  domain.origin = {-1.3, 0.2, 4.0};
  domain.periodic = {false, false, false};
  const auto layout = layout_from_domain(domain);
  REQUIRE(layout.count[0] == 4);
  REQUIRE(layout.count[1] == 3);
  REQUIRE(layout.count[2] == 2);
  const auto cell = layout.box(1, 0, 1);
  // External tangent: center just outside the max-x face, distance = radius.
  const Implicit sphere =
      make_sphere({cell.hi.x + 0.2, 0.5 * (cell.lo.y + cell.hi.y),
                   0.5 * (cell.lo.z + cell.hi.z)},
                  0.2);
  const auto verdict = classify_box(sphere, cell, {});
  REQUIRE(verdict.kind == CellKind::Cut);
  REQUIRE(verdict.tangent == 1);
  const auto far = classify_box(sphere, layout.box(0, 0, 0), {});
  REQUIRE(far.kind == CellKind::Exterior);
}

TEST_CASE("periodic seam cell ends on the image of the origin",
          "[geometry][classify]") {
  Domain domain;
  domain.size = {8, 8, 8};
  domain.spacing = {0.5, 0.5, 0.5};
  domain.origin = {1, 2, 3};
  domain.periodic = {true, true, true};
  const auto layout = layout_from_domain(domain);
  REQUIRE(layout.count[0] == 8);
  const auto seam = layout.box(7, 0, 0);
  REQUIRE(seam.hi.x == Approx(domain.origin[0] + 8 * domain.spacing[0]));
  const int periodic[3] = {1, 1, 1};
  const Vec3 image =
      wrap_coordinate(seam.hi, layout.origin, layout.period(), periodic);
  REQUIRE(image.x == Approx(layout.origin.x).margin(1e-12));
  // φ is not tiled. A sphere on the seam cuts the last cell and not the first.
  const Implicit sphere =
      make_sphere({seam.lo.x + 0.05, seam.lo.y + 0.25, seam.lo.z + 0.25}, 0.1);
  REQUIRE(classify_box(sphere, seam, {}).kind == CellKind::Cut);
  REQUIRE(classify_box(sphere, layout.box(0, 0, 0), {}).kind == CellKind::Exterior);
}

TEST_CASE("near-zero samples and a degenerate sphere gradient",
          "[geometry][implicit]") {
  const Implicit sphere = make_sphere({0, 0, 0}, 1);
  Policy policy;
  policy.length_absolute = 1e-8;
  policy.length_relative = 0;
  REQUIRE(side_of(sphere, {1 + 1e-12, 0, 0}, policy, 1) == Side::OnInterface);
  REQUIRE(side_of(sphere, {1 + 1e-6, 0, 0}, policy, 1) == Side::Outside);
  REQUIRE(sphere.unit_normal({0, 0, 0}, policy).degenerate == 1);
  REQUIRE(sphere.unit_normal({1, 0, 0}, policy).degenerate == 0);
}

TEST_CASE("plane cut conserves volume and matches analytic moments",
          "[geometry][cut]") {
  const Box box{{0, 0, 0}, {1, 1, 1}};
  Policy policy;
  policy.moment_degree = 2;
  const auto half =
      integrate_box(make_plane({0, 0, 0.5}, {0, 0, 1}), box, 3, policy, 1);
  REQUIRE(half.measure.inside_volume == Approx(0.5).epsilon(1e-12));
  REQUIRE(half.measure.outside_volume == Approx(0.5).epsilon(1e-12));
  REQUIRE(half.measure.inside_volume + half.measure.outside_volume ==
          Approx(half.measure.cell_volume).epsilon(1e-12));
  REQUIRE(half.measure.interface_area == Approx(1).epsilon(1e-12));
  REQUIRE(half.complete == 1);
  double w = 0, moment_z = 0, moment_z2 = 0;
  for (int i = 0; i < half.inside_count; ++i) {
    REQUIRE(half.inside_w[i] > 0);
    w += half.inside_w[i];
    moment_z += half.inside_w[i] * half.inside_at[i].z;
    moment_z2 += half.inside_w[i] * half.inside_at[i].z * half.inside_at[i].z;
  }
  REQUIRE(w == Approx(0.5).epsilon(1e-12));
  REQUIRE(moment_z == Approx(0.125).epsilon(1e-10));
  REQUIRE(moment_z2 == Approx(1.0 / 24.0).epsilon(1e-10));
  const Implicit diagonal = make_plane({0.5, 0.5, 0.5}, {1, 1, 1});
  const auto mid = measure_box(diagonal, box, 3, {});
  REQUIRE(mid.inside_volume == Approx(0.5).epsilon(1e-10));
}

TEST_CASE("diagonal plane volume matches the corner tetrahedron",
          "[geometry][cut]") {
  const Box box{{0, 0, 0}, {1, 1, 1}};
  // x+y+z <= 1 has volume 1/6. The unit normal does not change the set.
  const auto cut =
      measure_box(make_plane({1.0 / 3, 1.0 / 3, 1.0 / 3}, {1, 1, 1}), box, 3, {});
  REQUIRE(cut.inside_volume == Approx(1.0 / 6.0).epsilon(1e-10));
  REQUIRE(cut.outside_volume == Approx(5.0 / 6.0).epsilon(1e-10));
}

TEST_CASE("a contained sphere and ellipsoid recover analytic volume",
          "[geometry][cut]") {
  const Box box{{0, 0, 0}, {1, 1, 1}};
  const Implicit sphere = make_sphere({0.5, 0.5, 0.5}, 0.2);
  const auto sphere_cut = measure_box(sphere, box, 3, {});
  REQUIRE(sphere_cut.inside_volume == Approx(sphere.body_volume()).epsilon(1e-12));
  REQUIRE(sphere_cut.interface_area == Approx(sphere.body_area()).epsilon(1e-12));
  REQUIRE(sphere_cut.outside_volume ==
          Approx(1 - sphere.body_volume()).epsilon(1e-12));
  const auto rotation = rotation_about({1, 1, 1}, 0.7);
  const Implicit ellipsoid =
      make_ellipsoid({0.5, 0.5, 0.5}, {0.15, 0.1, 0.08}, rotation);
  const auto ell = measure_box(ellipsoid, box, 3, {});
  REQUIRE(ell.inside_volume == Approx(ellipsoid.body_volume()).epsilon(1e-12));
  REQUIRE(ell.inside_volume > 0);
  REQUIRE_FALSE(ell.sliver_inside);
}

TEST_CASE("curved cut volume converges under subdivision and grid refinement",
          "[geometry][cut]") {
  const Implicit sphere = make_sphere({0.5, 0.5, 0.5}, 0.3);
  Policy coarse;
  coarse.max_depth = 2;
  Policy fine;
  fine.max_depth = 4;
  // The sphere is not contained in a single octant of depth 0, so the
  // root cell is cut and the shortcut does not apply. Force that by
  // measuring a cell the sphere only partly covers.
  const Box partial{{0.4, 0.4, 0.4}, {0.8, 0.8, 0.8}};
  Policy shallow;
  shallow.max_depth = 1;
  const double a = measure_box(sphere, partial, 3, coarse).inside_volume;
  const double b = measure_box(sphere, partial, 3, fine).inside_volume;
  const double shallow_volume =
      measure_box(sphere, partial, 3, shallow).inside_volume;
  REQUIRE(a > 0);
  REQUIRE(b > 0);
  REQUIRE(absd(b - a) <= absd(shallow_volume - b) + 1e-6);
  Domain domain;
  domain.size = {9, 9, 9};
  domain.spacing = {0.125, 0.125, 0.125};
  domain.origin = {0, 0, 0};
  domain.periodic = {false, false, false};
  Policy grid;
  grid.max_depth = 3;
  const auto sum = measure_domain(sphere, layout_from_domain(domain), grid);
  const double exact = sphere.body_volume();
  REQUIRE(sum.inside_volume == Approx(exact).epsilon(0.02));
  REQUIRE(sum.inside_volume + sum.outside_volume ==
          Approx(sum.cell_volume).epsilon(1e-8));
}

TEST_CASE("tiny positive cut fractions are kept", "[geometry][cut]") {
  const Box box{{0, 0, 0}, {1, 1, 1}};
  Policy policy;
  policy.sliver_fraction = 1e-4;
  const auto sliver =
      measure_box(make_plane({0, 0, 1e-6}, {0, 0, 1}), box, 3, policy);
  REQUIRE(sliver.inside_volume == Approx(1e-6).epsilon(1e-6));
  REQUIRE(sliver.inside_volume > 0);
  REQUIRE(sliver.sliver_inside == 1);
}

TEST_CASE("ellipse in the plane is a polynomial, not a distance",
          "[geometry][implicit]") {
  const auto rotation = rotation_about({0, 0, 1}, 0.4);
  const Implicit ellipse = make_ellipse({0, 0, 0}, {2, 1, 0}, rotation);
  REQUIRE_FALSE(is_signed_distance(ellipse.family));
  REQUIRE(ellipse.value({0, 0, 0}) < 0);
  REQUIRE_THROWS_AS(
      make_ellipse({0, 0, 0}, {2, 1, 0}, rotation_about({1, 0, 0}, 0.2)),
      std::invalid_argument);
}

TEST_CASE("ellipsoid range matches a dense sample of the minimum",
          "[geometry][implicit]") {
  const auto rotation = rotation_about({0.2, -0.4, 0.9}, 1.1);
  const Implicit shape = make_ellipsoid({0.3, -0.2, 0.4}, {1.4, 0.7, 0.5}, rotation);
  const Box box{{-0.2, -1.0, 0.0}, {1.1, 0.4, 1.3}};
  pfc::geometry::Range range;
  REQUIRE(shape.range_on_box(box.lo, box.hi, range));
  double sample_min = 1e300;
  const int n = 24;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const Vec3 p{box.lo.x + (box.hi.x - box.lo.x) * i / (n - 1.0),
                     box.lo.y + (box.hi.y - box.lo.y) * j / (n - 1.0),
                     box.lo.z + (box.hi.z - box.lo.z) * k / (n - 1.0)};
        sample_min = std::min(sample_min, shape.value(p));
      }
  REQUIRE(range.min_phi <= sample_min + 1e-8);
  REQUIRE(sample_min < range.min_phi + 0.05);
}

TEST_CASE("cut list is compact and deterministic", "[geometry][classify]") {
  Domain domain =
      pfc::domain::with_spacing({8, 8, 8}, {0.25, 0.2, 0.3}, {false, false, false});
  const auto layout = layout_from_domain(domain);
  const auto list = classify_layout(make_sphere({0.7, 0.5, 0.4}, 0.35), layout, {});
  REQUIRE(list.interior + list.exterior + list.cut == layout.size());
  REQUIRE(list.cut == static_cast<int>(list.index.size()));
  for (std::size_t i = 1; i < list.index.size(); ++i)
    REQUIRE(list.index[i] > list.index[i - 1]);
  REQUIRE(list.cut > 0);
}

TEST_CASE("a flat domain classifies a circle in the plane", "[geometry][classify]") {
  Domain domain =
      pfc::domain::with_spacing({9, 9, 1}, {0.125, 0.125, 1}, {false, false, false});
  const auto layout = layout_from_domain(domain);
  REQUIRE(layout.dimension() == 2);
  REQUIRE(layout.count[2] == 1);
  const Implicit circle = pfc::geometry::make_circle({0.5, 0.5, 0}, 0.2);
  const auto cell = layout.box(4, 4, 0);
  const auto corners = sample_corners(circle, cell, {});
  const auto verdict = classify_box(circle, cell, {});
  // The circle can sit inside the centre cell while every corner is outside.
  if (corners.inside == 0 && corners.outside > 0)
    REQUIRE(verdict.kind == CellKind::Cut);
  const auto list = classify_layout(circle, layout, {});
  REQUIRE(list.interior + list.exterior + list.cut == layout.size());
  REQUIRE(list.cut > 0);
}
