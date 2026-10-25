// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file multigrid_plan.hpp
 * @brief Immutable, communication-free planning of rank-agglomerated Q1 GMG.
 *
 * A MultigridPlan describes geometry, active world ranks, rank groups,
 * owned-cell P/P^T routes and a configurable device-memory budget. It does
 * NOT allocate GPU fields, create MPI communicators or execute transfers.
 * The execution/workspace stage will consume this plan later.
 *
 * All rank IDs refer to the original fine-level communicator (0..P-1).
 * The process grids and boxes come from OpenPFC Decomposition/split_box,
 * not a second domain partitioner or a private MPI rank numbering scheme.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::solvers {

struct MultigridPlanOptions {
  /// Enforced on every axis of a level distributed across >1 MPI rank.
  int minimum_owned_extent = 16;
  /// Current Q1 Dirichlet row and P/P^T use exactly a two-cell halo.
  int halo_width = 2;
  /// Estimated halo-padded FP64 fields, not an allocation or measured peak.
  int estimated_padded_fields = 12;
  /// A single process continues down to 7^3 or 7x7x15 free nodes.
  int terminal_short_extent = 7;
  /// Reject any intermediate rank-grid contraction factor other than 1/2/3.
  bool permit_factor_three = true;
  /// Future MPI sender/receiver paths use int element counts per message.
  std::uint64_t maximum_route_elements =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max());
};

/**
 * The first, CPU-testable phase of an OpenPFC multigrid implementation.
 * This is currently a geometry policy for homogeneous-Dirichlet Q1 free
 * nodes (nx=ny=2^k-1, nz=nx or 2*nx+1). It is deliberately not a new
 * decomposition or a runtime implementation for other boundary conditions.
 */
class MultigridPlan {
public:
  struct Rank {
    int world_rank = -1; ///< Rank in original fine communicator.
    int level_rank = -1; ///< Compact rank in this level's communicator.
    Box3i owned;         ///< OpenPFC's unique owned free-node box.
    std::uint64_t owned_nodes = 0;
    std::uint64_t padded_nodes = 0;
    std::uint64_t estimated_device_bytes = 0;
  };

  struct Level {
    int index;
    Domain domain;
    Int3 process_grid;
    decomposition::Decomposition partition;
    std::vector<int> active_world_ranks; ///< x-fastest level rank -> world rank.
    std::vector<Rank> ranks;
    std::uint64_t global_nodes = 0;
    std::uint64_t total_estimated_device_bytes = 0;

    Level(int i, const decomposition::Decomposition &dec, std::vector<int> active)
        : index(i), domain(decomposition::domain(dec)),
          process_grid(decomposition::get_grid(dec)), partition(dec),
          active_world_ranks(std::move(active)) {}

    [[nodiscard]] int active_count() const noexcept {
      return static_cast<int>(active_world_ranks.size());
    }
  };

  struct Group {
    int coarse_world_rank = -1;        ///< Leader, an active world rank on coarse.
    std::vector<int> fine_world_ranks; ///< Members in fine-level rank order.
  };

  /**
   * Exactly one fine-owner/coarse-owner route per overlapping Q1 support
   * block. For R=P^T, fine sends ADDITIVE partial coarse contributions;
   * for P, coarse sends coefficients back to fine. Both use the same
   * coarse_overlap. Neither direction sends a replicated fine-grid field.
   *
   * A route need not stay within the coarse leader's Cartesian rank group:
   * one-cell interpolation support may cross that group's boundary.
   */
  struct Route {
    int fine_world_rank = -1;
    int coarse_world_rank = -1;
    Box3i coarse_overlap;
    Box3i fine_support_overlap;
    std::uint64_t values_per_direction = 0;
    bool crosses_group = false;
  };

  struct Transition {
    int fine_level = -1, coarse_level = -1;
    Int3 rank_contraction{1, 1, 1};
    std::vector<Group> groups;
    std::vector<Route> routes;
    std::uint64_t routed_values_per_direction = 0;
    std::uint64_t routed_bytes_per_direction = 0;
  };

  [[nodiscard]] static MultigridPlan build(const decomposition::Decomposition &fine,
                                           MultigridPlanOptions opt = {}) {
    validate_options(opt);
    const auto &d = decomposition::domain(fine);
    const auto n = domain::get_size(d);
    const auto per = domain::get_periodic(d);
    const auto is_power_two = [](std::int64_t a) {
      return a > 0 && (a & (a - 1)) == 0;
    };
    if (n[0] < 2 * opt.terminal_short_extent + 1 || n[1] != n[0] ||
        !(n[2] == n[0] || static_cast<std::int64_t>(n[2]) ==
                              2LL * static_cast<std::int64_t>(n[0]) + 1) ||
        !is_power_two(static_cast<std::int64_t>(n[0]) + 1) ||
        !is_power_two(static_cast<std::int64_t>(n[2]) + 1) || per[0] || per[1] ||
        per[2])
      throw std::invalid_argument(
          "MultigridPlan: expected nonperiodic cubic/2:1 2^k-1 Q1 free nodes");
    for (int a = 0; a < 3; ++a)
      if (!(d.spacing[a] > 0.0) || !std::isfinite(d.spacing[a]) ||
          !std::isfinite(d.origin[a]))
        throw std::invalid_argument("MultigridPlan: nonfinite spacing or origin");

    const int nranks = decomposition::get_num_domains(fine);
    if (nranks < 1) throw std::invalid_argument("MultigridPlan: no fine-level rank");
    std::vector<int> world_ranks;
    world_ranks.reserve(static_cast<std::size_t>(nranks));
    for (int r = 0; r < nranks; ++r) world_ranks.push_back(r);

    MultigridPlan out(opt, nranks);
    out.append_level(fine, std::move(world_ranks));
    validate_owned_width(out.m_levels.back(), opt);

    while (out.m_levels.back().domain.size[0] > opt.terminal_short_extent) {
      const Level &previous = out.m_levels.back();
      const auto prev_grid = previous.process_grid;
      const auto prev_domain = previous.domain;
      Int3 next_size{};
      Real3 next_spacing{}, next_origin{};
      for (int axis = 0; axis < 3; ++axis) {
        next_size[axis] = (prev_domain.size[axis] - 1) / 2;
        next_spacing[axis] = 2.0 * prev_domain.spacing[axis];
        next_origin[axis] = prev_domain.origin[axis] + prev_domain.spacing[axis];
        if (!std::isfinite(next_spacing[axis]) || !std::isfinite(next_origin[axis]))
          throw std::overflow_error(
              "MultigridPlan: coarse physical coordinates overflow");
      }
      const Int3 next_grid = choose_grid(prev_grid, next_size, opt);
      auto next_domain =
          domain::create(GridSize(next_size), PhysicalOrigin(next_origin),
                         GridSpacing(next_spacing), Bool3{false, false, false});
      const auto next_decomp = decomposition::create(next_domain, next_grid);
      std::vector<int> next_world = leaders(previous, next_grid);
      out.append_level(next_decomp, std::move(next_world));
      const auto &finer = out.m_levels[out.m_levels.size() - 2];
      const auto &coarser = out.m_levels.back();
      out.m_transitions.push_back(make_transition(finer, coarser, opt));
    }

    const auto &leaf = out.m_levels.back();
    if (leaf.active_count() != 1 || leaf.active_world_ranks.front() != 0 ||
        leaf.domain.size[0] != opt.terminal_short_extent ||
        !(leaf.domain.size[2] == opt.terminal_short_extent ||
          leaf.domain.size[2] == 2 * opt.terminal_short_extent + 1))
      throw std::invalid_argument(
          "MultigridPlan: cannot reach a one-rank bounded Q1 coarse level");
    return out;
  }

  [[nodiscard]] const MultigridPlanOptions &options() const noexcept {
    return m_options;
  }
  [[nodiscard]] const std::vector<Level> &levels() const noexcept {
    return m_levels;
  }
  [[nodiscard]] const std::vector<Transition> &transitions() const noexcept {
    return m_transitions;
  }
  [[nodiscard]] int world_size() const noexcept { return m_world_size; }

  /// Sum of field-budget estimates across every level active on a rank.
  /// This is a conservative planning parameter, NOT an actual device
  /// allocation, halo/MPI staging budget or measured GPU memory peak.
  [[nodiscard]] const std::vector<std::uint64_t> &
  estimated_device_bytes_by_world_rank() const noexcept {
    return m_device_bytes;
  }

private:
  explicit MultigridPlan(MultigridPlanOptions options, int world_size)
      : m_options(options), m_world_size(world_size),
        m_device_bytes(static_cast<std::size_t>(world_size), 0) {}

  static void validate_options(const MultigridPlanOptions &o) {
    if (o.minimum_owned_extent < 2 || o.halo_width != 2 ||
        o.estimated_padded_fields < 1 || o.terminal_short_extent != 7 ||
        o.maximum_route_elements == 0 ||
        o.maximum_route_elements >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("MultigridPlan: unsupported Q1 planning policy");
  }

  static std::uint64_t multiply(std::uint64_t a, std::uint64_t b, const char *what) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
      throw std::overflow_error(std::string("MultigridPlan: ") + what);
    return a * b;
  }
  static std::uint64_t add(std::uint64_t a, std::uint64_t b, const char *what) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
      throw std::overflow_error(std::string("MultigridPlan: ") + what);
    return a + b;
  }
  static std::uint64_t volume(const Int3 &n, int pad = 0) {
    std::uint64_t out = 1;
    for (int a = 0; a < 3; ++a) {
      if (n[a] < 1 || n[a] > std::numeric_limits<int>::max() - 2 * pad)
        throw std::invalid_argument("MultigridPlan: invalid box extent");
      out = multiply(out, static_cast<std::uint64_t>(n[a] + 2 * pad),
                     "box volume overflow");
    }
    return out;
  }
  static std::uint64_t volume(const Box3i &b) { return volume(b.size); }
  static Int3 coords(int rank, const Int3 &g) {
    return {rank % g[0], (rank / g[0]) % g[1], rank / (g[0] * g[1])};
  }
  static int rank_of(const Int3 &c, const Int3 &g) {
    return (c[2] * g[1] + c[1]) * g[0] + c[0];
  }
  static std::uint64_t grid_count(const Int3 &g) { return volume(g); }

  void append_level(const decomposition::Decomposition &dec,
                    std::vector<int> active) {
    const int nactive = decomposition::get_num_domains(dec);
    if (nactive != static_cast<int>(active.size()) ||
        grid_count(decomposition::get_grid(dec)) !=
            static_cast<std::uint64_t>(nactive))
      throw std::invalid_argument(
          "MultigridPlan: rank mapping disagrees with OpenPFC decomposition");
    Level level(static_cast<int>(m_levels.size()), dec, std::move(active));
    level.global_nodes = volume(level.domain.size);
    level.ranks.reserve(static_cast<std::size_t>(nactive));
    for (int local = 0; local < nactive; ++local) {
      const auto box = decomposition::local_box(dec, local);
      const int world = level.active_world_ranks[static_cast<std::size_t>(local)];
      if (world < 0 || world >= m_world_size)
        throw std::invalid_argument("MultigridPlan: rank identity out of range");
      Rank r;
      r.world_rank = world;
      r.level_rank = local;
      r.owned = box;
      r.owned_nodes = volume(box);
      r.padded_nodes = volume(box.size, m_options.halo_width);
      r.estimated_device_bytes = multiply(
          multiply(r.padded_nodes,
                   static_cast<std::uint64_t>(m_options.estimated_padded_fields),
                   "field-budget multiplication overflow"),
          sizeof(double), "estimated FP64 memory overflow");
      level.total_estimated_device_bytes =
          add(level.total_estimated_device_bytes, r.estimated_device_bytes,
              "per-level byte budget overflow");
      m_device_bytes[static_cast<std::size_t>(world)] =
          add(m_device_bytes[static_cast<std::size_t>(world)],
              r.estimated_device_bytes, "per-world-rank byte budget overflow");
      level.ranks.push_back(r);
    }
    m_levels.push_back(std::move(level));
  }

  static void validate_owned_width(const Level &level,
                                   const MultigridPlanOptions &opt) {
    for (const Rank &r : level.ranks)
      for (int a = 0; a < 3; ++a)
        if (r.owned.size[a] <
            (level.active_count() == 1 ? opt.halo_width : opt.minimum_owned_extent))
          throw std::invalid_argument(
              "MultigridPlan: active MPI brick below minimum owned extent");
  }

  static Int3 choose_grid(const Int3 &old, const Int3 &size,
                          const MultigridPlanOptions &opt) {
    std::array<std::vector<int>, 3> choices;
    for (int a = 0; a < 3; ++a) {
      choices[a].push_back(old[a]);
      if (old[a] % 2 == 0) choices[a].push_back(old[a] / 2);
      if (opt.permit_factor_three && old[a] % 3 == 0)
        choices[a].push_back(old[a] / 3);
      std::sort(choices[a].begin(), choices[a].end());
      choices[a].erase(std::unique(choices[a].begin(), choices[a].end()),
                       choices[a].end());
    }

    bool found = false;
    std::uint64_t max_active = 0;
    long double min_surface = std::numeric_limits<long double>::infinity();
    Int3 best{1, 1, 1};
    for (int gx : choices[0])
      for (int gy : choices[1])
        for (int gz : choices[2]) {
          const Int3 g{gx, gy, gz};
          const auto active = grid_count(g);
          bool valid = true;
          for (int a = 0; a < 3; ++a)
            if (g[a] > size[a] ||
                size[a] / g[a] <
                    (active == 1 ? opt.halo_width : opt.minimum_owned_extent))
              valid = false;
          if (!valid) continue;
          const long double x = static_cast<long double>(size[0]) / gx,
                            y = static_cast<long double>(size[1]) / gy,
                            z = static_cast<long double>(size[2]) / gz;
          const long double surface = x * y + y * z + z * x;
          if (!found || active > max_active ||
              (active == max_active && surface < min_surface)) {
            found = true;
            best = g;
            max_active = active;
            min_surface = surface;
          }
        }
    if (!found)
      throw std::invalid_argument(
          "MultigridPlan: no supported 1/2/3 rank contraction meets "
          "the minimum owned brick extent");
    return best;
  }

  static std::vector<int> leaders(const Level &fine, const Int3 &coarse_grid) {
    const Int3 parent = fine.process_grid;
    Int3 factor{};
    for (int a = 0; a < 3; ++a) {
      if (coarse_grid[a] < 1 || parent[a] % coarse_grid[a] != 0)
        throw std::invalid_argument("MultigridPlan: non-nested process grids");
      factor[a] = parent[a] / coarse_grid[a];
      if (factor[a] != 1 && factor[a] != 2 && factor[a] != 3)
        throw std::invalid_argument("MultigridPlan: unsupported rank contraction");
    }
    const int n = static_cast<int>(grid_count(coarse_grid));
    std::vector<int> active;
    active.reserve(static_cast<std::size_t>(n));
    for (int cr = 0; cr < n; ++cr) {
      const auto c = coords(cr, coarse_grid);
      const Int3 first{c[0] * factor[0], c[1] * factor[1], c[2] * factor[2]};
      active.push_back(
          fine.active_world_ranks[static_cast<std::size_t>(rank_of(first, parent))]);
    }
    return active;
  }

  static Transition make_transition(const Level &fine, const Level &coarse,
                                    const MultigridPlanOptions &opt) {
    Transition t;
    t.fine_level = fine.index;
    t.coarse_level = coarse.index;
    for (int a = 0; a < 3; ++a)
      t.rank_contraction[a] = fine.process_grid[a] / coarse.process_grid[a];

    // Nested Cartesian rank groups: future MPI_Comm_split participants.
    t.groups.resize(coarse.ranks.size());
    for (std::size_t c = 0; c < t.groups.size(); ++c)
      t.groups[c].coarse_world_rank = coarse.active_world_ranks[c];
    std::vector<int> grouped_owner(fine.ranks.size(), -1);
    for (const Rank &f : fine.ranks) {
      const auto fc = coords(f.level_rank, fine.process_grid);
      const Int3 cc{fc[0] / t.rank_contraction[0], fc[1] / t.rank_contraction[1],
                    fc[2] / t.rank_contraction[2]};
      const int c = rank_of(cc, coarse.process_grid);
      const int owner = coarse.active_world_ranks[static_cast<std::size_t>(c)];
      t.groups[static_cast<std::size_t>(c)].fine_world_ranks.push_back(f.world_rank);
      grouped_owner[static_cast<std::size_t>(f.level_rank)] = owner;
    }

    const auto coarse_size = domain::get_size(coarse.domain);
    const auto cgrid = coarse.process_grid;
    for (const Rank &f : fine.ranks) {
      Int3 support_lo{}, support_hi{};
      for (int axis = 0; axis < 3; ++axis) {
        // For physical free Q1 nodes: fine i couples to coarse j iff
        // 2*j <= i <= 2*j+2. These are the exact support extrema.
        support_lo[axis] = std::max(0, (f.owned.low[axis] + 1) / 2 - 1);
        support_hi[axis] = std::min(coarse_size[axis] - 1, f.owned.high[axis] / 2);
      }
      Int3 first{}, last{};
      for (int axis = 0; axis < 3; ++axis) {
        first[axis] = -1;
        last[axis] = -1;
        for (int i = 0; i < cgrid[axis]; ++i) {
          Int3 c{0, 0, 0};
          c[axis] = i;
          const int rank = rank_of(c, cgrid);
          const auto &box = coarse.ranks[static_cast<std::size_t>(rank)].owned;
          if (box.high[axis] < support_lo[axis] || box.low[axis] > support_hi[axis])
            continue;
          if (first[axis] < 0) first[axis] = i;
          last[axis] = i;
        }
        if (first[axis] < 0)
          throw std::logic_error(
              "MultigridPlan: no coarse owner for fine Q1 support");
      }

      for (int z = first[2]; z <= last[2]; ++z)
        for (int y = first[1]; y <= last[1]; ++y)
          for (int x = first[0]; x <= last[0]; ++x) {
            const int c = rank_of(Int3{x, y, z}, cgrid);
            const Rank &dst = coarse.ranks[static_cast<std::size_t>(c)];
            Int3 clo{}, chi{}, flo{}, fhi{};
            for (int axis = 0; axis < 3; ++axis) {
              clo[axis] = std::max(support_lo[axis], dst.owned.low[axis]);
              chi[axis] = std::min(support_hi[axis], dst.owned.high[axis]);
              // The fine contributors to this coarse block can live on
              // either side of the agglomeration group boundary.
              flo[axis] = std::max(f.owned.low[axis], 2 * clo[axis]);
              fhi[axis] = std::min(f.owned.high[axis], 2 * chi[axis] + 2);
              if (clo[axis] > chi[axis] || flo[axis] > fhi[axis])
                throw std::logic_error("MultigridPlan: empty Q1 transfer route");
            }
            Route route;
            route.fine_world_rank = f.world_rank;
            route.coarse_world_rank = dst.world_rank;
            route.coarse_overlap = Box3i::from_bounds(clo, chi);
            route.fine_support_overlap = Box3i::from_bounds(flo, fhi);
            route.values_per_direction = volume(route.coarse_overlap);
            if (route.values_per_direction > opt.maximum_route_elements)
              throw std::overflow_error(
                  "MultigridPlan: route exceeds MPI int-count capacity");
            route.crosses_group =
                route.coarse_world_rank !=
                grouped_owner[static_cast<std::size_t>(f.level_rank)];
            t.routed_values_per_direction =
                add(t.routed_values_per_direction, route.values_per_direction,
                    "aggregate transfer count overflow");
            t.routes.push_back(std::move(route));
          }
    }
    t.routed_bytes_per_direction =
        multiply(t.routed_values_per_direction, sizeof(double),
                 "aggregate route bytes overflow");
    return t;
  }

  MultigridPlanOptions m_options;
  int m_world_size;
  std::vector<Level> m_levels;
  std::vector<Transition> m_transitions;
  std::vector<std::uint64_t> m_device_bytes;
};

} // namespace pfc::solvers
