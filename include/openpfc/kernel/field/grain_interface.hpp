// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file grain_interface.hpp
 * @brief Compact periodic grain identity and a narrow interface band.
 *
 * One integer field names the grain in every cell. Interface cells, their
 * oriented normals, and the junction class live in a band whose length
 * follows the boundary, not the grain count. This is not one signed-distance
 * field per grain, and it does not move a boundary.
 */

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace pfc::field::grain_interface {

/// Cell counts and spacing. Cell `i` sits at `(i + 1/2) hx`.
struct GrainGrid {
  int nx{0};
  int ny{0};
  int nz{0};
  double hx{0.0};
  double hy{0.0};
  double hz{0.0};
};

/// One seed inside the periodic box `[0, L)`.
struct GrainSeed {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

/// One interface cell. `n_ids` is 2 on a face, 3 on a triple line, and
/// greater than 3 at a higher junction. `distance` is a positive width on
/// this cell only.
struct BandSite {
  int index{0};
  int owner{0};
  int other{0};
  int n_ids{0};
  double normal_x{0.0};
  double normal_y{0.0};
  double normal_z{0.0};
  double distance{0.0};
};

static_assert(std::is_trivially_copyable_v<BandSite>);
static_assert(std::is_standard_layout_v<BandSite>);
static_assert(sizeof(BandSite) == 4 * sizeof(int) + 4 * sizeof(double));

struct GrainInterface {
  GrainGrid grid{};
  std::vector<int> grain_id;
  std::vector<BandSite> band;
  std::vector<int> cell_count;
  std::vector<double> volume_fraction;
};

struct JunctionCensus {
  int faces{0};
  int triples{0};
  int higher{0};
};

[[nodiscard]] inline std::size_t grain_cell_count(const GrainGrid &grid) {
  if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1) return 0;
  return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
         static_cast<std::size_t>(grid.nz);
}

/// Host bytes of the grain-id field, the band, and the per-grain totals.
[[nodiscard]] inline std::size_t compact_storage_bytes(const GrainInterface &in) {
  return in.grain_id.size() * sizeof(int) + in.band.size() * sizeof(BandSite) +
         in.cell_count.size() * sizeof(int) +
         in.volume_fraction.size() * sizeof(double);
}

/// Bytes of one dense signed-distance field for every grain.
[[nodiscard]] inline std::size_t dense_level_set_bytes(int n_grains,
                                                       std::size_t n_cells) {
  if (n_grains < 1 || n_cells == 0) return 0;
  return static_cast<std::size_t>(n_grains) * n_cells * sizeof(double);
}

/// Device payload: grain ids, band records, per-grain counts, and one
/// mismatch word.
[[nodiscard]] inline std::size_t
resident_device_bytes(std::size_t n_cells, std::size_t n_band, int n_grains) {
  if (n_grains < 1) return 0;
  return n_cells * sizeof(int) + n_band * sizeof(BandSite) +
         static_cast<std::size_t>(n_grains) * sizeof(int) + sizeof(int);
}

[[nodiscard]] inline JunctionCensus classify_junctions(const GrainInterface &in) {
  JunctionCensus out;
  for (const BandSite &site : in.band) {
    if (site.n_ids == 2)
      ++out.faces;
    else if (site.n_ids == 3)
      ++out.triples;
    else if (site.n_ids > 3)
      ++out.higher;
  }
  return out;
}

namespace detail {

inline void validate_grid(const GrainGrid &grid) {
  if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1)
    throw std::invalid_argument("grain grid needs a positive cell count");
  if (!(grid.hx > 0.0) || !(grid.hy > 0.0) || !(grid.hz > 0.0) ||
      !std::isfinite(grid.hx) || !std::isfinite(grid.hy) || !std::isfinite(grid.hz))
    throw std::invalid_argument("grain spacing must be positive");
}

inline void validate_seeds(const GrainGrid &grid, std::span<const GrainSeed> seeds) {
  validate_grid(grid);
  if (seeds.empty()) throw std::invalid_argument("grain seeds are empty");
  const double lx = grid.nx * grid.hx;
  const double ly = grid.ny * grid.hy;
  const double lz = grid.nz * grid.hz;
  for (const GrainSeed &seed : seeds) {
    if (!std::isfinite(seed.x) || !std::isfinite(seed.y) || !std::isfinite(seed.z))
      throw std::domain_error("nonfinite grain seed");
    if (!(seed.x >= 0.0) || !(seed.x < lx) || !(seed.y >= 0.0) || !(seed.y < ly) ||
        !(seed.z >= 0.0) || !(seed.z < lz))
      throw std::invalid_argument("grain seed lies outside the periodic box");
  }
}

[[nodiscard]] inline int wrap_coord(int i, int n) {
  if (i < 0) return i + n;
  if (i >= n) return i - n;
  return i;
}

[[nodiscard]] inline std::size_t linear_index(const GrainGrid &grid, int i, int j,
                                              int k) {
  return static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
}

[[nodiscard]] inline double image_distance2(double cx, double cy, double cz,
                                            const GrainSeed &seed, double lx,
                                            double ly, double lz) {
  double best = std::numeric_limits<double>::infinity();
  for (int ox = -1; ox <= 1; ++ox) {
    for (int oy = -1; oy <= 1; ++oy) {
      for (int oz = -1; oz <= 1; ++oz) {
        const double dx = cx - (seed.x + ox * lx);
        const double dy = cy - (seed.y + oy * ly);
        const double dz = cz - (seed.z + oz * lz);
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) best = d2;
      }
    }
  }
  return best;
}

[[nodiscard]] inline double min_image_distance2(double cx, double cy, double cz,
                                                const GrainSeed &seed, double lx,
                                                double ly, double lz) {
  const double dx = std::remainder(cx - seed.x, lx);
  const double dy = std::remainder(cy - seed.y, ly);
  const double dz = std::remainder(cz - seed.z, lz);
  return dx * dx + dy * dy + dz * dz;
}

template <typename Distance>
void assign_seeds(const GrainGrid &grid, std::span<const GrainSeed> seeds,
                  std::span<int> grain_id, Distance distance) {
  validate_seeds(grid, seeds);
  const std::size_t n = grain_cell_count(grid);
  if (grain_id.size() != n)
    throw std::invalid_argument("grain-id size does not match the grid");
  const double lx = grid.nx * grid.hx;
  const double ly = grid.ny * grid.hy;
  const double lz = grid.nz * grid.hz;
  const int n_seeds = static_cast<int>(seeds.size());
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const double cx = (static_cast<double>(i) + 0.5) * grid.hx;
        const double cy = (static_cast<double>(j) + 0.5) * grid.hy;
        const double cz = (static_cast<double>(k) + 0.5) * grid.hz;
        double best = std::numeric_limits<double>::infinity();
        int winner = 0;
        for (int s = 0; s < n_seeds; ++s) {
          const double d2 =
              distance(cx, cy, cz, seeds[static_cast<std::size_t>(s)], lx, ly, lz);
          if (d2 < best) {
            best = d2;
            winner = s;
          }
        }
        grain_id[linear_index(grid, i, j, k)] = winner;
      }
    }
  }
}

struct Neighborhood {
  int owner{0};
  int other{0};
  int n_ids{0};
  double normal_x{0.0};
  double normal_y{0.0};
  double normal_z{0.0};
  double distance{0.0};
};

[[nodiscard]] inline Neighborhood inspect_cell(const GrainGrid &grid,
                                               const int *grain_id, std::size_t p) {
  const int nxny = grid.nx * grid.ny;
  const int k = static_cast<int>(p / static_cast<std::size_t>(nxny));
  const int rem = static_cast<int>(p - static_cast<std::size_t>(k) * nxny);
  const int j = rem / grid.nx;
  const int i = rem - j * grid.nx;
  const int owner = grain_id[p];

  int seen[27];
  int nseen = 0;
  auto consider = [&](int id) {
    for (int t = 0; t < nseen; ++t) {
      if (seen[t] == id) return;
    }
    seen[nseen++] = id;
  };
  consider(owner);

  int tally_id[27];
  int tally[27];
  int ntally = 0;
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  double nearest = std::numeric_limits<double>::infinity();
  for (int dk = -1; dk <= 1; ++dk) {
    for (int dj = -1; dj <= 1; ++dj) {
      for (int di = -1; di <= 1; ++di) {
        if (di == 0 && dj == 0 && dk == 0) continue;
        const int ii = wrap_coord(i + di, grid.nx);
        const int jj = wrap_coord(j + dj, grid.ny);
        const int kk = wrap_coord(k + dk, grid.nz);
        const std::size_t q = linear_index(grid, ii, jj, kk);
        if (q == p) continue;
        const int id = grain_id[q];
        consider(id);
        if (id == owner) continue;
        int slot = -1;
        for (int t = 0; t < ntally; ++t) {
          if (tally_id[t] == id) slot = t;
        }
        if (slot < 0) {
          slot = ntally;
          tally_id[ntally] = id;
          tally[ntally] = 0;
          ++ntally;
        }
        ++tally[slot];
        const double dx = di * grid.hx;
        const double dy = dj * grid.hy;
        const double dz = dk * grid.hz;
        sx += dx;
        sy += dy;
        sz += dz;
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < nearest) nearest = dist;
      }
    }
  }

  int other = owner;
  int other_count = 0;
  for (int t = 0; t < ntally; ++t) {
    if (tally[t] > other_count || (tally[t] == other_count && tally_id[t] < other)) {
      other_count = tally[t];
      other = tally_id[t];
    }
  }
  const double len = std::sqrt(sx * sx + sy * sy + sz * sz);
  Neighborhood out;
  out.owner = owner;
  out.other = ntally == 0 ? owner : other;
  out.n_ids = nseen;
  if (len > 0.0) {
    out.normal_x = sx / len;
    out.normal_y = sy / len;
    out.normal_z = sz / len;
  }
  out.distance = ntally == 0 ? 0.0 : 0.5 * nearest;
  return out;
}

} // namespace detail

/// Nearest periodic seed by an explicit 27-image search. A tie keeps the
/// smaller seed index. The output is written only after the arguments pass.
inline void assign_periodic_voronoi(const GrainGrid &grid,
                                    std::span<const GrainSeed> seeds,
                                    std::span<int> grain_id) {
  detail::assign_seeds(grid, seeds, grain_id, detail::image_distance2);
}

/// Nearest periodic seed by the separated min-image formula. The tie rule
/// matches `assign_periodic_voronoi`.
inline void assign_periodic_voronoi_min_image(const GrainGrid &grid,
                                              std::span<const GrainSeed> seeds,
                                              std::span<int> grain_id) {
  detail::assign_seeds(grid, seeds, grain_id, detail::min_image_distance2);
}

/// Grain ids from the 27-image Voronoi map, plus the narrow band.
[[nodiscard]] inline GrainInterface
build_grain_interface(const GrainGrid &grid, std::span<const GrainSeed> seeds) {
  GrainInterface out;
  out.grid = grid;
  out.grain_id.assign(grain_cell_count(grid), 0);
  assign_periodic_voronoi(grid, seeds, out.grain_id);
  const int n_grains = static_cast<int>(seeds.size());
  out.cell_count.assign(static_cast<std::size_t>(n_grains), 0);
  for (int id : out.grain_id) ++out.cell_count[static_cast<std::size_t>(id)];
  const double n = static_cast<double>(out.grain_id.size());
  out.volume_fraction.resize(static_cast<std::size_t>(n_grains));
  for (int s = 0; s < n_grains; ++s) {
    out.volume_fraction[static_cast<std::size_t>(s)] =
        static_cast<double>(out.cell_count[static_cast<std::size_t>(s)]) / n;
  }
  out.band.reserve(out.grain_id.size() / 4);
  for (std::size_t p = 0; p < out.grain_id.size(); ++p) {
    const detail::Neighborhood nb =
        detail::inspect_cell(grid, out.grain_id.data(), p);
    if (nb.n_ids < 2) continue;
    BandSite site;
    site.index = static_cast<int>(p);
    site.owner = nb.owner;
    site.other = nb.other;
    site.n_ids = nb.n_ids;
    site.normal_x = nb.normal_x;
    site.normal_y = nb.normal_y;
    site.normal_z = nb.normal_z;
    site.distance = nb.distance;
    out.band.push_back(site);
  }
  return out;
}

} // namespace pfc::field::grain_interface
