// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file multigrid_mpi_transfers.hpp
 * @brief Host Q1 P/P^T transport over MultigridPlan's rank-agglomerated levels.
 *
 * This is the CPU/MPI qualification layer. It consumes the exact rank
 * ownership and Q1 routes prepared by MultigridPlan, without defining a
 * second partitioner or assembling/replicating a global fine-grid field.
 * Active communicators are created once; no GPU or V-cycle is constructed.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/solvers/multigrid_plan.hpp>

namespace pfc::solvers {

/**
 * Rank-agglomerated, uniquely owned Q1 transfers on the host.
 *
 * Every world rank constructs this object collectively, even if it becomes
 * inactive on a coarse level. A level's active communicator contains precisely
 * the plan's world ranks, ordered by level_rank. Inactive ranks return without
 * participating in that level's transfers. All *active fine-level* ranks must
 * invoke restrict_pt/prolong_add in the same order.
 *
 * Inputs and outputs are contiguous x-fastest arrays of OWNED nodes; no halo
 * data or nonlocal fine-grid replicas are accepted. Communication is directly
 * between the plan's fine/coarse owners, not merely within leader groups:
 * trilinear interpolation can cross an agglomeration-group boundary.
 *
 * This first version deliberately allocates temporary route buffers per call.
 * Workspace ownership and device buffers belong to the subsequent step.
 */
class MultigridMPITransfers {
public:
  explicit MultigridMPITransfers(const MultigridPlan &plan, MPI_Comm world)
      : m_plan(plan), m_world(world) {
    if (world == MPI_COMM_NULL)
      throw std::invalid_argument("MultigridMPITransfers: null communicator");
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
      throw std::invalid_argument("MultigridMPITransfers: MPI not initialized");
    check(MPI_Comm_rank(world, &m_rank), "MPI_Comm_rank");
    int size = 0;
    check(MPI_Comm_size(world, &size), "MPI_Comm_size");
    if (size != plan.world_size())
      throw std::invalid_argument(
          "MultigridMPITransfers: communicator differs from plan world size");

    m_comms.reserve(plan.levels().size());
    for (const auto &level : plan.levels()) {
      const auto it = std::find(level.active_world_ranks.begin(),
                                level.active_world_ranks.end(), m_rank);
      const bool active = it != level.active_world_ranks.end();
      const int key = active ? static_cast<int>(
          std::distance(level.active_world_ranks.begin(), it)) : 0;
      MPI_Comm level_comm = MPI_COMM_NULL;
      check(MPI_Comm_split(world, active ? 0 : MPI_UNDEFINED, key,
                           &level_comm), "MPI_Comm_split");
      m_comms.push_back(level_comm);
    }
  }

  MultigridMPITransfers(const MultigridMPITransfers &) = delete;
  MultigridMPITransfers &operator=(const MultigridMPITransfers &) = delete;

  ~MultigridMPITransfers() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) return;
    // MPI_Comm_free is collective among members of each active communicator.
    for (auto &comm : m_comms)
      if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
  }

  [[nodiscard]] MPI_Comm level_comm(std::size_t level) const {
    if (level >= m_comms.size())
      throw std::out_of_range("MultigridMPITransfers: level out of range");
    return m_comms[level];
  }

  [[nodiscard]] int world_rank() const noexcept { return m_rank; }

  /**
   * Additive restriction R=P^T.
   *
   * Each active fine rank computes Q1 contributions from its unique owned
   * nodes. Each coarse owner accumulates all partial blocks from the planned
   * source ranks. Inactive coarse ranks return an empty output.
   */
  void restrict_pt(std::size_t transition,
                   const std::vector<double> &fine_owned,
                   std::vector<double> &coarse_owned) const {
    const auto &step = get_step(transition);
    const auto &fine = m_plan.levels()[transition];
    const auto &coarse = m_plan.levels()[transition + 1];
    const auto *fr = rank_record(fine, m_rank);
    const auto *cr = rank_record(coarse, m_rank);
    if (!fr) {
      coarse_owned.clear();
      return;
    }
    validate_sizes(transition, fine_owned.size(), fr->owned_nodes,
                   &fine_owned == &coarse_owned, "restriction fine input");

    coarse_owned.assign(cr ? checked_size(cr->owned_nodes) : 0, 0.0);
    std::vector<std::vector<double>> payload(step.routes.size());
    std::vector<MPI_Request> requests;
    requests.reserve(step.routes.size());

    // Post receives before sends. Each route is a distinct owner pair.
    for (std::size_t i = 0; i < step.routes.size(); ++i) {
      const auto &route = step.routes[i];
      if (route.coarse_world_rank != m_rank ||
          route.fine_world_rank == m_rank) continue;
      payload[i].resize(checked_size(route.values_per_direction));
      MPI_Request req = MPI_REQUEST_NULL;
      check(MPI_Irecv(payload[i].data(), checked_count(route),
                      MPI_DOUBLE, level_rank(fine, route.fine_world_rank),
                      kRestrictTag, m_comms[transition], &req),
            "MPI_Irecv restriction");
      requests.push_back(req);
    }

    for (std::size_t i = 0; i < step.routes.size(); ++i) {
      const auto &route = step.routes[i];
      if (route.fine_world_rank != m_rank) continue;
      payload[i].resize(checked_size(route.values_per_direction));
      for_each_index(route.coarse_overlap, [&](const Int3 &j) {
        payload[i][checked_index(route.coarse_overlap, j)] =
            partial_restrict(fr->owned, fine_owned, j);
      });
      if (route.coarse_world_rank == m_rank) {
        add_block(route.coarse_overlap, payload[i], cr->owned, coarse_owned);
      } else {
        MPI_Request req = MPI_REQUEST_NULL;
        check(MPI_Isend(payload[i].data(), checked_count(route),
                        MPI_DOUBLE, level_rank(fine, route.coarse_world_rank),
                        kRestrictTag, m_comms[transition], &req),
              "MPI_Isend restriction");
        requests.push_back(req);
      }
    }

    wait(requests);
    for (std::size_t i = 0; i < step.routes.size(); ++i) {
      const auto &route = step.routes[i];
      if (route.coarse_world_rank == m_rank &&
          route.fine_world_rank != m_rank)
        add_block(route.coarse_overlap, payload[i], cr->owned, coarse_owned);
    }
  }

  /**
   * Prolongation P from uniquely owned coarse coefficients to fine nodes.
   * Reverse the exact same planned Q1 routes. A fine owner adds each coarse
   * coefficient's trilinear support exactly once, including cross-group data.
   */
  void prolong(std::size_t transition,
               const std::vector<double> &coarse_owned,
               std::vector<double> &fine_owned) const {
    const auto &step = get_step(transition);
    const auto &fine = m_plan.levels()[transition];
    const auto &coarse = m_plan.levels()[transition + 1];
    const auto *fr = rank_record(fine, m_rank);
    const auto *cr = rank_record(coarse, m_rank);
    if (!fr) {
      fine_owned.clear();
      return;
    }
    validate_sizes(transition, coarse_owned.size(),
                   cr ? cr->owned_nodes : 0,
                   &coarse_owned == &fine_owned, "prolongation coarse input");

    fine_owned.assign(checked_size(fr->owned_nodes), 0.0);
    std::vector<std::vector<double>> payload(step.routes.size());
    std::vector<MPI_Request> requests;
    requests.reserve(step.routes.size());

    for (std::size_t i = 0; i < step.routes.size(); ++i) {
      const auto &route = step.routes[i];
      if (route.fine_world_rank != m_rank ||
          route.coarse_world_rank == m_rank) continue;
      payload[i].resize(checked_size(route.values_per_direction));
      MPI_Request req = MPI_REQUEST_NULL;
      check(MPI_Irecv(payload[i].data(), checked_count(route),
                      MPI_DOUBLE, level_rank(fine, route.coarse_world_rank),
                      kProlongTag, m_comms[transition], &req),
            "MPI_Irecv prolongation");
      requests.push_back(req);
    }

    for (std::size_t i = 0; i < step.routes.size(); ++i) {
      const auto &route = step.routes[i];
      if (route.coarse_world_rank != m_rank) continue;
      payload[i].resize(checked_size(route.values_per_direction));
      for_each_index(route.coarse_overlap, [&](const Int3 &j) {
        payload[i][checked_index(route.coarse_overlap, j)] =
            coarse_owned[checked_index(cr->owned, j)];
      });
      if (route.fine_world_rank != m_rank) {
        MPI_Request req = MPI_REQUEST_NULL;
        check(MPI_Isend(payload[i].data(), checked_count(route),
                        MPI_DOUBLE, level_rank(fine, route.fine_world_rank),
                        kProlongTag, m_comms[transition], &req),
              "MPI_Isend prolongation");
        requests.push_back(req);
      }
    }

    wait(requests);
    for (std::size_t i = 0; i < step.routes.size(); ++i) {
      const auto &route = step.routes[i];
      if (route.fine_world_rank == m_rank)
        add_prolong_block(route.coarse_overlap, payload[i],
                          fr->owned, fine_owned);
    }
  }

private:
  static constexpr int kRestrictTag = 9521;
  static constexpr int kProlongTag = 9522;

  static void check(int status, const char *operation) {
    if (status != MPI_SUCCESS)
      throw std::runtime_error(std::string("MultigridMPITransfers: ") +
                               operation + " failed");
  }

  static std::size_t checked_size(std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("MultigridMPITransfers: size_t overflow");
    return static_cast<std::size_t>(value);
  }

  static int checked_count(const MultigridPlan::Route &route) {
    if (route.values_per_direction >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("MultigridMPITransfers: MPI count overflow");
    return static_cast<int>(route.values_per_direction);
  }

  static std::size_t checked_index(const Box3i &box, const Int3 &point) {
    return static_cast<std::size_t>(box.to_linear(point));
  }

  static const MultigridPlan::Rank *
  rank_record(const MultigridPlan::Level &level, int world_rank) {
    const auto it = std::find(level.active_world_ranks.begin(),
                              level.active_world_ranks.end(), world_rank);
    if (it == level.active_world_ranks.end()) return nullptr;
    return &level.ranks[static_cast<std::size_t>(
        std::distance(level.active_world_ranks.begin(), it))];
  }

  static int level_rank(const MultigridPlan::Level &level, int world_rank) {
    const auto *r = rank_record(level, world_rank);
    if (!r)
      throw std::logic_error("MultigridMPITransfers: inactive route endpoint");
    return r->level_rank;
  }

  const MultigridPlan::Transition &get_step(std::size_t index) const {
    if (index >= m_plan.transitions().size())
      throw std::out_of_range("MultigridMPITransfers: step out of range");
    return m_plan.transitions()[index];
  }

  // All active ranks agree on input validity before posting point-to-point
  // operations. Otherwise a local validation exception could strand peers.
  void validate_sizes(std::size_t level, std::size_t actual,
                      std::uint64_t expected, bool aliased,
                      const char *what) const {
    const int invalid =
        (aliased || actual != checked_size(expected)) ? 1 : 0;
    int any_invalid = 0;
    check(MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX,
                        m_comms[level]), "MPI_Allreduce input validation");
    if (any_invalid)
      throw std::invalid_argument(
          std::string("MultigridMPITransfers: invalid ") + what);
  }

  static double weight(int i, int j) {
    const auto displacement = static_cast<std::int64_t>(i) -
                              (2LL * static_cast<std::int64_t>(j) + 1);
    if (displacement == 0) return 1.0;
    if (displacement == -1 || displacement == 1) return 0.5;
    return 0.0;
  }

  static double partial_restrict(const Box3i &owned,
                                 const std::vector<double> &values,
                                 const Int3 &j) {
    Int3 lo{}, hi{};
    for (int axis = 0; axis < 3; ++axis) {
      const auto centre = 2LL * static_cast<std::int64_t>(j[axis]) + 1;
      lo[axis] = static_cast<int>(std::max<std::int64_t>(
          owned.low[axis], centre - 1));
      hi[axis] = static_cast<int>(std::min<std::int64_t>(
          owned.high[axis], centre + 1));
    }
    double sum = 0.0;
    if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) return sum;
    for (int z = lo[2]; z <= hi[2]; ++z)
      for (int y = lo[1]; y <= hi[1]; ++y)
        for (int x = lo[0]; x <= hi[0]; ++x) {
          const Int3 i{x, y, z};
          sum += weight(x, j[0]) * weight(y, j[1]) *
                 weight(z, j[2]) * values[checked_index(owned, i)];
        }
    return sum;
  }

  static void add_block(const Box3i &block,
                        const std::vector<double> &source,
                        const Box3i &owned,
                        std::vector<double> &destination) {
    for_each_index(block, [&](const Int3 &at) {
      destination[checked_index(owned, at)] +=
          source[checked_index(block, at)];
    });
  }

  static void add_prolong_block(const Box3i &coarse_block,
                                const std::vector<double> &coefficients,
                                const Box3i &fine_owned,
                                std::vector<double> &result) {
    for_each_index(coarse_block, [&](const Int3 &j) {
      const double c = coefficients[checked_index(coarse_block, j)];
      Int3 lo{}, hi{};
      for (int a = 0; a < 3; ++a) {
        const auto centre = 2LL * static_cast<std::int64_t>(j[a]) + 1;
        lo[a] = static_cast<int>(std::max<std::int64_t>(
            fine_owned.low[a], centre - 1));
        hi[a] = static_cast<int>(std::min<std::int64_t>(
            fine_owned.high[a], centre + 1));
      }
      if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) return;
      for (int z = lo[2]; z <= hi[2]; ++z)
        for (int y = lo[1]; y <= hi[1]; ++y)
          for (int x = lo[0]; x <= hi[0]; ++x) {
            const Int3 i{x, y, z};
            result[checked_index(fine_owned, i)] += c *
                weight(x, j[0]) * weight(y, j[1]) * weight(z, j[2]);
          }
    });
  }

  static void wait(std::vector<MPI_Request> &requests) {
    if (requests.empty()) return;
    if (requests.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("MultigridMPITransfers: too many requests");
    check(MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                      MPI_STATUSES_IGNORE), "MPI_Waitall");
  }

  const MultigridPlan &m_plan;
  MPI_Comm m_world = MPI_COMM_NULL;
  int m_rank = -1;
  std::vector<MPI_Comm> m_comms;
};

} // namespace pfc::solvers
