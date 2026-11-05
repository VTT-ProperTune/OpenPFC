// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/field/grain_interface.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using pfc::field::grain_interface::assign_periodic_voronoi;
using pfc::field::grain_interface::assign_periodic_voronoi_min_image;
using pfc::field::grain_interface::BandSite;
using pfc::field::grain_interface::build_grain_interface;
using pfc::field::grain_interface::classify_junctions;
using pfc::field::grain_interface::compact_storage_bytes;
using pfc::field::grain_interface::dense_level_set_bytes;
using pfc::field::grain_interface::grain_cell_count;
using pfc::field::grain_interface::GrainGrid;
using pfc::field::grain_interface::GrainInterface;
using pfc::field::grain_interface::GrainSeed;
using pfc::field::grain_interface::JunctionCensus;

namespace {

void report(const std::string &label, double a, double b, double c, double d) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "GRAIN " << label << " " << a << " " << b << " " << c << " " << d << '\n';
  std::cout << out.str();
}

GrainGrid cube(int n) {
  GrainGrid grid;
  grid.nx = n;
  grid.ny = n;
  grid.nz = n;
  grid.hx = 1.0 / static_cast<double>(n);
  grid.hy = grid.hx;
  grid.hz = grid.hx;
  return grid;
}

std::size_t at(const GrainGrid &grid, int i, int j, int k) {
  return static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
}

const BandSite *find_site(const GrainInterface &in, std::size_t index) {
  for (const BandSite &site : in.band) {
    if (static_cast<std::size_t>(site.index) == index) return &site;
  }
  return nullptr;
}

std::vector<GrainSeed> irregular_seeds(const GrainGrid &grid, int count) {
  const double lx = grid.nx * grid.hx;
  const double ly = grid.ny * grid.hy;
  const double lz = grid.nz * grid.hz;
  std::vector<GrainSeed> seeds(static_cast<std::size_t>(count));
  for (int s = 0; s < count; ++s) {
    double x = std::fmod(0.17 + s * 0.6180339887498948, 1.0) * lx;
    double y = std::fmod(0.31 + s * 0.4142135623730950, 1.0) * ly;
    double z = std::fmod(0.47 + s * 0.7071067811865476, 1.0) * lz;
    if (!(x < lx)) x = 0.0;
    if (!(y < ly)) y = 0.0;
    if (!(z < lz)) z = 0.0;
    seeds[static_cast<std::size_t>(s)] = GrainSeed{x, y, z};
  }
  return seeds;
}

void require_compact(const GrainInterface &in, int n_grains) {
  const std::size_t n = in.grain_id.size();
  const std::size_t compact = compact_storage_bytes(in);
  const std::size_t dense = dense_level_set_bytes(n_grains, n);
  REQUIRE(compact == n * sizeof(int) + in.band.size() * sizeof(BandSite) +
                         in.cell_count.size() * sizeof(int) +
                         in.volume_fraction.size() * sizeof(double));
  REQUIRE(compact < dense);
  REQUIRE(in.band.size() > 0);
  REQUIRE(in.band.size() < n);
  int sum = 0;
  double fraction = 0.0;
  for (int s = 0; s < n_grains; ++s) {
    REQUIRE(in.cell_count[static_cast<std::size_t>(s)] > 0);
    sum += in.cell_count[static_cast<std::size_t>(s)];
    fraction += in.volume_fraction[static_cast<std::size_t>(s)];
  }
  REQUIRE(sum == static_cast<int>(n));
  REQUIRE(std::abs(fraction - 1.0) < 1e-12);
}

} // namespace

TEST_CASE("Periodic grain ids stay compact and match both Voronoi maps",
          "[unit][field][grain_interface]") {
  const GrainGrid grid = cube(32);
  const std::size_t n = grain_cell_count(grid);
  const std::vector<GrainSeed> slab{{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}};
  std::vector<int> images(n, -1);
  std::vector<int> wrapped(n, -1);
  assign_periodic_voronoi(grid, slab, images);
  assign_periodic_voronoi_min_image(grid, slab, wrapped);
  REQUIRE(images == wrapped);

  const GrainInterface built = build_grain_interface(grid, slab);
  REQUIRE(built.grain_id == images);
  REQUIRE(built.cell_count[0] == static_cast<int>(n / 2));
  REQUIRE(built.cell_count[1] == static_cast<int>(n / 2));
  REQUIRE(built.volume_fraction[0] == 0.5);
  REQUIRE(built.volume_fraction[1] == 0.5);
  const JunctionCensus slab_class = classify_junctions(built);
  REQUIRE(slab_class.faces == static_cast<int>(built.band.size()));
  REQUIRE(slab_class.triples == 0);
  REQUIRE(slab_class.higher == 0);
  require_compact(built, 2);

  const BandSite *low = find_site(built, at(grid, 0, 16, 16));
  const BandSite *high = find_site(built, at(grid, 31, 16, 16));
  const BandSite *mid = find_site(built, at(grid, 15, 16, 16));
  REQUIRE(low != nullptr);
  REQUIRE(high != nullptr);
  REQUIRE(mid != nullptr);
  REQUIRE(low->owner == 0);
  REQUIRE(low->other == 1);
  REQUIRE(low->n_ids == 2);
  REQUIRE(std::abs(low->normal_x + 1.0) < 1e-12);
  REQUIRE(std::abs(low->normal_y) < 1e-12);
  REQUIRE(std::abs(low->normal_z) < 1e-12);
  REQUIRE(std::abs(low->distance - 0.5 * grid.hx) < 1e-12);
  REQUIRE(high->owner == 1);
  REQUIRE(high->other == 0);
  REQUIRE(std::abs(high->normal_x - 1.0) < 1e-12);
  REQUIRE(mid->owner == 0);
  REQUIRE(std::abs(mid->normal_x - 1.0) < 1e-12);

  const GrainGrid odd = cube(5);
  const std::size_t n_odd = grain_cell_count(odd);
  std::vector<int> tie_a(n_odd, 3);
  std::vector<int> tie_b(n_odd, 4);
  assign_periodic_voronoi(odd, slab, tie_a);
  assign_periodic_voronoi_min_image(odd, slab, tie_b);
  REQUIRE(tie_a == tie_b);
  REQUIRE(tie_a[at(odd, 2, 2, 2)] == 0);

  std::vector<int> kept(8, 7);
  const std::vector<int> original = kept;
  REQUIRE_THROWS_AS(assign_periodic_voronoi(grid, {}, kept), std::invalid_argument);
  REQUIRE(kept == original);
  const std::vector<GrainSeed> outside{{1.0, 0.5, 0.5}};
  REQUIRE_THROWS_AS(assign_periodic_voronoi(grid, outside, images),
                    std::invalid_argument);
  REQUIRE(images[0] == built.grain_id[0]);
  const std::vector<GrainSeed> bad{{std::nan(""), 0.2, 0.2}};
  REQUIRE_THROWS_AS(assign_periodic_voronoi_min_image(grid, bad, wrapped),
                    std::domain_error);
  REQUIRE(wrapped == images);

  std::vector<GrainSeed> corners;
  for (double z : {0.25, 0.75}) {
    for (double y : {0.25, 0.75}) {
      for (double x : {0.25, 0.75}) corners.push_back(GrainSeed{x, y, z});
    }
  }
  const GrainInterface octet = build_grain_interface(grid, corners);
  std::vector<int> octet_wrap(n, -1);
  assign_periodic_voronoi_min_image(grid, corners, octet_wrap);
  REQUIRE(octet.grain_id == octet_wrap);
  for (int count : octet.cell_count) REQUIRE(count == static_cast<int>(n / 8));
  const JunctionCensus octet_class = classify_junctions(octet);
  REQUIRE(octet_class.faces > 0);
  REQUIRE(octet_class.triples + octet_class.higher > 0);
  require_compact(octet, 8);

  int min_count = 0;
  int max_count = 0;
  std::size_t band_16 = 0;
  for (int grains : {16, 64, 128}) {
    const std::vector<GrainSeed> seeds = irregular_seeds(grid, grains);
    std::vector<int> by_image(n, -1);
    std::vector<int> by_wrap(n, -1);
    assign_periodic_voronoi(grid, seeds, by_image);
    assign_periodic_voronoi_min_image(grid, seeds, by_wrap);
    REQUIRE(by_image == by_wrap);
    const GrainInterface poly = build_grain_interface(grid, seeds);
    REQUIRE(poly.grain_id == by_image);
    require_compact(poly, grains);
    const JunctionCensus census = classify_junctions(poly);
    REQUIRE(census.faces > 0);
    REQUIRE(census.triples > 0);
    int lo = poly.cell_count[0];
    int hi = poly.cell_count[0];
    for (int count : poly.cell_count) {
      lo = std::min(lo, count);
      hi = std::max(hi, count);
    }
    REQUIRE(hi > lo);
    for (const BandSite &site : poly.band) {
      if (site.n_ids != 2) continue;
      const double len =
          std::sqrt(site.normal_x * site.normal_x + site.normal_y * site.normal_y +
                    site.normal_z * site.normal_z);
      REQUIRE(std::abs(len - 1.0) < 1e-12);
      REQUIRE(site.distance > 0.0);
      REQUIRE(site.other != site.owner);
    }
    if (grains == 16) {
      min_count = lo;
      max_count = hi;
      band_16 = poly.band.size();
    }
    report(std::to_string(grains), static_cast<double>(lo), static_cast<double>(hi),
           static_cast<double>(poly.band.size()),
           static_cast<double>(compact_storage_bytes(poly)) /
               static_cast<double>(dense_level_set_bytes(grains, n)));
  }
  report("slab", static_cast<double>(built.band.size()),
         static_cast<double>(octet_class.triples),
         static_cast<double>(octet_class.higher),
         static_cast<double>(compact_storage_bytes(octet)) /
             static_cast<double>(dense_level_set_bytes(8, n)));
  REQUIRE(min_count > 0);
  REQUIRE(max_count > min_count);
  REQUIRE(band_16 > 0);
}
