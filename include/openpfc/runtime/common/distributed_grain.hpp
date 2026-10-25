// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * @file distributed_grain.hpp
 * @brief Collective compact grain observation, planning and publication.
 */
#pragma once
#include <climits>
#include <mpi.h>
#include <openpfc/kernel/grain/distributed.hpp>
#include <openpfc/runtime/common/grain_remapping.hpp>
#include <optional>
#include <unordered_map>

namespace pfc::grain::distributed {
enum class Status : unsigned {
  Success,
  Deferred,
  InvalidInput,
  UnsafeCadence,
  Unseeded,
  UnknownIdentity,
  MissingSupport,
  CapacityOverflow,
  Infeasible,
  SearchLimit,
  StageFailure
};
struct Binding {
  Id component = 0, uid = 0;
};
struct Plan {
  Status status = Status::InvalidInput;
  Snapshot snapshot;
  Slot slots = 0;
  std::vector<Transfer> global_moves;
  std::vector<Grain> local_grains;
  std::vector<Transfer> local_moves;
  std::vector<Binding> bindings;
  std::size_t exchanged_bytes = 0;
};
namespace detail {
inline bool supported(const remapping::Options &options) {
  return options.diagnostics == nullptr &&
         (options.method == remapping::Method::Incremental ||
          options.method == remapping::Method::GlobalSaturation ||
          options.method == remapping::Method::GlobalLargestFirst ||
          options.method == remapping::Method::CompleteOracle);
}
inline void mpi(int result) {
  if (result != MPI_SUCCESS)
    throw std::runtime_error("distributed grain MPI failure");
}
inline Status vote(Status status, MPI_Comm comm) {
  auto input = static_cast<unsigned>(status), output = input;
  mpi(MPI_Allreduce(&input, &output, 1, MPI_UNSIGNED, MPI_MAX, comm));
  return static_cast<Status>(output);
}
// Every local allocation/validation failure is voted before the next collective.
inline std::vector<std::vector<Id>> gather(std::span<const Id> words, MPI_Comm comm,
                                           std::size_t max_words) {
  int ranks = 0;
  mpi(MPI_Comm_size(comm, &ranks));
  std::vector<int> counts, offsets;
  auto status = Status::Success;
  try {
    counts.resize(ranks);
    offsets.resize(ranks);
  } catch (...) {
    status = Status::CapacityOverflow;
  }
  if (words.size() > std::size_t(INT_MAX)) status = Status::CapacityOverflow;
  if (vote(status, comm) != Status::Success)
    throw std::length_error("distributed grain collective buffer capacity");
  const int count = static_cast<int>(words.size());
  mpi(MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm));
  std::size_t total = 0;
  for (int r = 0; r < ranks; ++r) {
    offsets[r] = static_cast<int>(total);
    total += std::size_t(counts[r]);
    if (total > max_words || total > std::size_t(INT_MAX))
      status = Status::CapacityOverflow;
  }
  std::vector<Id> all;
  std::vector<std::vector<Id>> output;
  try {
    if (status == Status::Success) {
      all.resize(total);
      output.resize(ranks);
      for (int r = 0; r < ranks; ++r) output[r].resize(counts[r]);
    }
  } catch (...) {
    status = Status::CapacityOverflow;
  }
  if (vote(status, comm) != Status::Success)
    throw std::length_error("distributed grain summary limit exceeded");
  mpi(MPI_Allgatherv(words.data(), count, MPI_UINT64_T, all.data(), counts.data(),
                     offsets.data(), MPI_UINT64_T, comm));
  for (int r = 0; r < ranks; ++r)
    std::copy_n(all.begin() + offsets[r], counts[r], output[r].begin());
  return output;
}
inline std::vector<Id> encode(const Observation &o, Slot slots,
                              std::span<const Grain> grains,
                              const remapping::Options &options) {
  validate(o.partition);
  const auto &p = o.partition;
  std::vector<Id> w{p.global.nx,
                    p.global.ny,
                    p.global.nz,
                    p.global.periodic_x,
                    p.global.periodic_y,
                    p.global.periodic_z,
                    Id(p.global.connectivity),
                    p.lower[0],
                    p.lower[1],
                    p.lower[2],
                    p.extent[0],
                    p.extent[1],
                    p.extent[2],
                    slots,
                    options.contact_radius,
                    options.contact_capacity,
                    Id(options.method),
                    options.limits.attempts,
                    options.limits.depth,
                    options.componentwise,
                    options.check_now,
                    options.epoch,
                    grains.size()};
  for (auto g : grains) w.insert(w.end(), {g.id, g.slot, g.active});
  w.push_back(o.components.size());
  for (auto c : o.components) w.insert(w.end(), {c.key, c.slot, c.seed, c.samples});
  w.push_back(o.boundary.size());
  for (auto b : o.boundary) w.insert(w.end(), {b.cell, b.component, b.slot});
  w.push_back(o.contacts.size());
  for (auto e : o.contacts) w.insert(w.end(), {e.first, e.second});
  return w;
}
struct Reader {
  std::span<const Id> words;
  std::size_t cursor = 0;
  Id next() {
    if (cursor == words.size())
      throw std::invalid_argument("truncated grain summary");
    return words[cursor++];
  }
};
inline Observation decode(std::span<const Id> words) {
  Reader rd{words};
  Observation o;
  o.partition.global = {std::size_t(rd.next()), std::size_t(rd.next()),
                        std::size_t(rd.next()), bool(rd.next()),
                        bool(rd.next()),        bool(rd.next()),
                        Connectivity(rd.next())};
  for (auto &x : o.partition.lower) x = rd.next();
  for (auto &x : o.partition.extent) x = rd.next();
  for (int i = 0; i < 9; ++i) rd.next();
  const auto grains = rd.next();
  for (std::size_t i = 0; i < grains * 3; ++i) rd.next();
  auto count = rd.next();
  for (std::size_t i = 0; i < count; ++i)
    o.components.push_back({rd.next(), Slot(rd.next()), rd.next(), rd.next()});
  count = rd.next();
  for (std::size_t i = 0; i < count; ++i)
    o.boundary.push_back({rd.next(), rd.next(), Slot(rd.next())});
  count = rd.next();
  for (std::size_t i = 0; i < count; ++i)
    o.contacts.push_back({rd.next(), rd.next()});
  if (rd.cursor != words.size())
    throw std::invalid_argument("trailing grain summary");
  validate(o.partition);
  return o;
}
// Pure validation: throws before graph construction; contains no collectives.
inline void validate_partition_cover(std::span<const Observation> obs,
                                     Grid3D global) {
  std::size_t volume = 0;
  for (std::size_t r = 0; r < obs.size(); ++r) {
    volume += cell_count(obs[r].partition.local());
    for (std::size_t s = 0; s < r; ++s) {
      bool overlap = true;
      for (int d = 0; d < 3; ++d)
        overlap &= obs[r].partition.lower[d] <
                       obs[s].partition.lower[d] + obs[s].partition.extent[d] &&
                   obs[s].partition.lower[d] <
                       obs[r].partition.lower[d] + obs[r].partition.extent[d];
      if (overlap) throw std::invalid_argument("overlapping grain partitions");
    }
  }
  if (volume != cell_count(global))
    throw std::invalid_argument("grain partitions do not cover global volume");
}

// Component identity is (rank, local key), never the reusable slot number.
struct ComponentTopology {
  using Key = std::pair<std::size_t, Id>;
  std::map<Key, std::size_t> indices;
  std::vector<Component> components;
  std::vector<std::size_t> parent;
  std::size_t root(std::size_t i) const {
    while (i != parent[i]) i = parent[i];
    return i;
  }
  std::size_t index(std::size_t rank, Id key) const {
    auto it = indices.find({rank, key});
    if (it == indices.end())
      throw std::invalid_argument("unknown boundary component");
    return it->second;
  }
};

inline ComponentTopology index_components(std::span<const Observation> obs,
                                          Slot slots) {
  ComponentTopology topology;
  for (std::size_t r = 0; r < obs.size(); ++r)
    for (auto c : obs[r].components) {
      if (!c.key || c.slot >= slots || !c.samples ||
          !topology.indices
               .emplace(ComponentTopology::Key{r, c.key}, topology.components.size())
               .second)
        throw std::invalid_argument("invalid component summary");
      topology.components.push_back(c);
    }
  topology.parent.resize(topology.components.size());
  std::iota(topology.parent.begin(), topology.parent.end(), 0);
  return topology;
}

// Reconcile same-slot neighbours and collect contacts without MPI or UID
// publication. Every boundary sample must belong to the observed owned shell.
inline std::set<std::pair<std::size_t, std::size_t>>
collect_component_contacts(std::span<const Observation> obs, Grid3D global,
                           std::size_t contact_radius, ComponentTopology &topology) {
  std::unordered_map<Id, std::vector<std::pair<std::size_t, std::size_t>>> shell;
  for (std::size_t r = 0; r < obs.size(); ++r) {
    std::set<std::pair<Id, Slot>> seen;
    for (auto b : obs[r].boundary) {
      const auto component = topology.index(r, b.component);
      if (b.cell >= cell_count(global) ||
          b.slot != topology.components[component].slot ||
          !seen.emplace(b.cell, b.slot).second)
        throw std::invalid_argument("invalid or duplicate boundary sample");
      const auto &p = obs[r].partition;
      const std::array<std::size_t, 3> x{b.cell % global.nx,
                                         (b.cell / global.nx) % global.ny,
                                         b.cell / (global.nx * global.ny)};
      std::size_t cell = 0;
      for (int d = 0; d < 3; ++d)
        if (x[d] < p.lower[d] || x[d] >= p.lower[d] + p.extent[d])
          throw std::invalid_argument("boundary sample outside owned box");
      cell = x[0] - p.lower[0] +
             p.extent[0] * (x[1] - p.lower[1] + p.extent[1] * (x[2] - p.lower[2]));
      if (!boundary(p, cell, contact_radius))
        throw std::invalid_argument("boundary sample outside shell");
      shell[b.cell].push_back({r, component});
    }
  }
  std::set<std::pair<std::size_t, std::size_t>> edges;
  for (std::size_t r = 0; r < obs.size(); ++r)
    for (auto e : obs[r].contacts) {
      auto a = topology.index(r, e.first), b = topology.index(r, e.second);
      if (a != b) edges.emplace(std::min(a, b), std::max(a, b));
    }
  const int radius = static_cast<int>(contact_radius);
  for (const auto &[cell, entries] : shell)
    for (int dz = -radius; dz <= radius; ++dz)
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
          if (!pfc::grain::detail::stencil(global, dx, dy, dz, radius)) continue;
          std::size_t other;
          if (!pfc::grain::detail::offset(global, cell, dx, dy, dz, other)) continue;
          const auto found = shell.find(other);
          if (found == shell.end()) continue;
          for (auto [r, a] : entries)
            for (auto [s, b] : found->second) {
              if (a == b) continue;
              edges.emplace(std::min(a, b), std::max(a, b));
              if (topology.components[a].slot == topology.components[b].slot &&
                  pfc::grain::detail::stencil(global, dx, dy, dz, 1) &&
                  std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1) {
                auto ra = topology.root(a), rb = topology.root(b);
                topology.parent[std::max(ra, rb)] = std::min(ra, rb);
              }
            }
        }
  return edges;
}

} // namespace detail

/// Collective calls must occur in identical order. Only compact shells,
/// component metadata and contacts are gathered. No amplitudes/dense labels.
/// Resource/validation failures reject on all ranks before publication.
inline Plan prepare(const Observation &local, std::span<const Grain> grains,
                    Slot slots, const remapping::Options &options, MPI_Comm comm,
                    std::size_t max_words = 16 * 1024 * 1024) {
  Plan plan;
  auto status = Status::Success;
  std::vector<Id> encoded;
  try {
    if (!detail::supported(options) || !slots || slots == unassigned ||
        !options.contact_radius || options.contact_radius > std::size_t(INT_MAX / 2))
      throw std::invalid_argument("invalid distributed grain palette/radius");
    if (pfc::grain::detail::prepare_transfer(slots, grains, {}).status !=
        TransferStatus::Success)
      throw std::invalid_argument("invalid grain registry");
    encoded = detail::encode(local, slots, grains, options);
  } catch (...) {
    status = Status::InvalidInput;
  }
  plan.status = detail::vote(status, comm);
  if (plan.status != Status::Success) return plan;
  std::vector<std::vector<Id>> packets;
  try {
    packets = detail::gather(encoded, comm, max_words);
  } catch (const std::length_error &) {
    plan.status = Status::CapacityOverflow;
    return plan;
  }
  int rank = 0;
  detail::mpi(MPI_Comm_rank(comm, &rank));
  try {
    const auto prepare_local_plan = [&] {
      std::vector<Observation> obs;
      const auto header = 23 + grains.size() * 3;
      for (const auto &w : packets) {
        // Geometry, numerical policy and complete global registry must agree.
        if (w.size() < header) throw std::invalid_argument("short grain header");
        for (std::size_t i = 0; i < header; ++i)
          if ((i < 7 || i >= 13) && w[i] != encoded[i])
            throw std::invalid_argument("inconsistent global grain metadata");
        plan.exchanged_bytes += w.size() * sizeof(Id);
        obs.push_back(detail::decode(w));
      }
      if (!options.check_now) {
        plan.status = Status::Deferred;
        return;
      }
      // All following stages are rank-local; the final vote is unchanged.
      detail::validate_partition_cover(obs, local.partition.global);
      auto topology = detail::index_components(obs, slots);
      const auto edges = detail::collect_component_contacts(
          obs, local.partition.global, options.contact_radius, topology);
      const auto &components = topology.components;
      const auto root = [&](std::size_t i) { return topology.root(i); };
      const auto index = [&](std::size_t r, Id key) {
        return topology.index(r, key);
      };
      std::vector<Id> uids(components.size(), 0);
      std::map<Id, std::uint64_t> counts;
      const auto resolve_identity_and_support = [&] {
        for (std::size_t i = 0; i < components.size(); ++i) {
          auto &id = uids[root(i)];
          if (components[i].seed && id && id != components[i].seed) {
            plan.status = Status::UnsafeCadence;
            break;
          }
          if (components[i].seed) id = components[i].seed;
        }
        if (plan.status != Status::Success) return;
        for (std::size_t i = 0; i < components.size(); ++i) {
          const Id uid = uids[root(i)];
          if (!uid) {
            plan.status = Status::Unseeded;
            break;
          }
          auto g = std::lower_bound(grains.begin(), grains.end(), uid,
                                    [](Grain g, Id id) { return g.id < id; });
          if (g == grains.end() || g->id != uid || !g->active) {
            plan.status = Status::UnknownIdentity;
            break;
          }
          if (g->slot != components[i].slot)
            throw std::invalid_argument("UID slot mismatch");
          if (counts[uid] > UINT64_MAX - components[i].samples)
            throw std::overflow_error("grain support overflow");
          counts[uid] += components[i].samples;
        }
        if (plan.status != Status::Success) return;
      };
      resolve_identity_and_support();
      if (plan.status != Status::Success) return;
      const auto decide_and_prepare_transfer = [&] {
        std::vector<Id> vertices;
        std::vector<Contact> contacts;
        for (auto g : grains)
          if (g.active) {
            if (!counts.contains(g.id)) {
              plan.status = Status::MissingSupport;
              break;
            }
            vertices.push_back(g.id);
          }
        if (plan.status != Status::Success) return;
        for (auto [a, b] : edges)
          if (uids[root(a)] != uids[root(b)])
            contacts.push_back({uids[root(a)], uids[root(b)]});
        auto graph = make_contact_graph(vertices, contacts);
        if (graph.edges.size() > options.contact_capacity) {
          plan.status = Status::CapacityOverflow;
          return;
        }
        std::vector<double> weights;
        for (auto id : graph.vertices) weights.push_back(double(counts[id]));
        auto decision =
            remapping::detail::decide(graph, grains, slots, weights, options);
        if (decision.status != remapping::Status::Success) {
          plan.status = decision.status == remapping::Status::Infeasible
                            ? Status::Infeasible
                            : Status::SearchLimit;
          return;
        }
        auto prepared =
            pfc::grain::detail::prepare_transfer(slots, grains, decision.moves);
        plan.snapshot = {options.epoch, std::move(prepared.grains),
                         std::move(graph)};
        plan.slots = slots;
        plan.global_moves = decision.moves;
      };
      decide_and_prepare_transfer();
      if (plan.status != Status::Success) return;
      // Bind local component keys only after a complete global decision.
      const auto bind_local_components = [&] {
        std::set<Id> present;
        for (auto c : local.components) {
          auto uid = uids[root(index(rank, c.key))];
          plan.bindings.push_back({c.key, uid});
          present.insert(uid);
        }
        for (auto g : grains)
          if (present.contains(g.id)) plan.local_grains.push_back(g);
        for (auto m : plan.global_moves)
          if (present.contains(m.id)) plan.local_moves.push_back(m);
      };
      bind_local_components();
    };
    prepare_local_plan();
  } catch (const std::bad_alloc &) {
    plan.status = Status::CapacityOverflow;
  } catch (...) {
    plan.status = Status::InvalidInput;
  }
  plan.status = detail::vote(plan.status, comm);
  if (plan.status != Status::Success) {
    plan.snapshot = {};
    plan.local_grains.clear();
    plan.local_moves.clear();
    plan.bindings.clear();
  }
  return plan;
}

/// The producer overload votes rank-local tracking/allocation exceptions before
/// any rank enters summary collectives. Prefer it to constructing observations
/// outside the collective call when producers can fail on only one rank.
template <class Producer>
  requires std::invocable<Producer>
Plan prepare(Producer producer, std::span<const Grain> grains, Slot slots,
             const remapping::Options &options, MPI_Comm comm,
             std::size_t max_words = 16 * 1024 * 1024) {
  Observation observed;
  auto status = Status::Success;
  try {
    observed = producer();
  } catch (const std::domain_error &) {
    status = Status::UnsafeCadence;
  } catch (const std::bad_alloc &) {
    status = Status::CapacityOverflow;
  } catch (const std::length_error &) {
    status = Status::CapacityOverflow;
  } catch (...) {
    status = Status::InvalidInput;
  }
  status = detail::vote(status, comm);
  if (status != Status::Success) {
    Plan plan;
    plan.status = status;
    return plan;
  }
  return prepare(observed, grains, slots, options, comm, max_words);
}

template <class Result> struct Staged {
  Status status = Status::InvalidInput;
  std::optional<Result> transaction;
};
/// Stage callbacks must leave originals untouched and return TransferResult.
/// All rank-local failures and exceptions are voted before returning any
/// publishable owning buffers. Process loss/MPI failure is not recoverable.
template <class Callback>
auto stage(const Plan &plan, MPI_Comm comm, Callback callback,
           std::size_t max_words = 16 * 1024 * 1024) {
  using Result = std::invoke_result_t<Callback, std::span<const Grain>,
                                      std::span<const Transfer>>;
  Staged<Result> out;
  out.status = detail::vote(plan.status, comm);
  if (out.status != Status::Success) return out;
  std::vector<Id> contract;
  try {
    pfc::grain::validate(plan.snapshot, plan.slots);
    auto local = pfc::grain::detail::prepare_transfer(plan.slots, plan.local_grains,
                                                      plan.local_moves);
    if (local.status != TransferStatus::Success)
      throw std::invalid_argument("invalid local transfer contract");
    std::vector<Transfer> expected_moves;
    for (auto m : plan.global_moves) {
      auto g = std::lower_bound(plan.local_grains.begin(), plan.local_grains.end(),
                                m.id, [](Grain g, Id id) { return g.id < id; });
      if (g != plan.local_grains.end() && g->id == m.id) expected_moves.push_back(m);
    }
    if (expected_moves.size() != plan.local_moves.size())
      throw std::invalid_argument("inconsistent local move count");
    for (std::size_t i = 0; i < expected_moves.size(); ++i) {
      auto a = expected_moves[i], b = plan.local_moves[i];
      if (a.id != b.id || a.source != b.source || a.destination != b.destination)
        throw std::invalid_argument("local moves disagree with global plan");
    }
    for (auto g : local.grains) {
      auto expected =
          std::lower_bound(plan.snapshot.grains.begin(), plan.snapshot.grains.end(),
                           g.id, [](Grain g, Id id) { return g.id < id; });
      if (expected == plan.snapshot.grains.end() || expected->id != g.id ||
          expected->slot != g.slot || expected->active != g.active)
        throw std::invalid_argument("local registry disagrees with global plan");
    }
    for (auto e : plan.snapshot.graph.edges) {
      auto first =
          std::lower_bound(plan.snapshot.grains.begin(), plan.snapshot.grains.end(),
                           e.first, [](Grain g, Id id) { return g.id < id; });
      auto second =
          std::lower_bound(plan.snapshot.grains.begin(), plan.snapshot.grains.end(),
                           e.second, [](Grain g, Id id) { return g.id < id; });
      if (first->slot == second->slot)
        throw std::invalid_argument("global plan retains contact conflict");
    }
    contract = {plan.slots, plan.snapshot.epoch, plan.snapshot.grains.size()};
    for (auto g : plan.snapshot.grains)
      contract.insert(contract.end(), {g.id, g.slot, g.active});
    contract.push_back(plan.snapshot.graph.edges.size());
    for (auto e : plan.snapshot.graph.edges)
      contract.insert(contract.end(), {e.first, e.second});
    for (auto m : plan.global_moves)
      contract.insert(contract.end(), {m.id, m.source, m.destination});
  } catch (...) {
    out.status = Status::InvalidInput;
  }
  out.status = detail::vote(out.status, comm);
  if (out.status != Status::Success) return out;
  std::vector<std::vector<Id>> contracts;
  try {
    contracts = detail::gather(contract, comm, max_words);
  } catch (const std::length_error &) {
    out.status = Status::CapacityOverflow;
    return out;
  }
  for (const auto &other : contracts)
    if (other != contract) out.status = Status::InvalidInput;
  if (out.status != Status::Success) return out;
  try {
    out.transaction.emplace(callback(plan.local_grains, plan.local_moves));
    if (out.transaction->status != TransferStatus::Success)
      out.status = Status::StageFailure;
  } catch (...) {
    out.status = Status::StageFailure;
  }
  out.status = detail::vote(out.status, comm);
  if (out.status != Status::Success) out.transaction.reset();
  return out;
}
} // namespace pfc::grain::distributed
