// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <iostream>
#include <map>
#include <openpfc/kernel/grain/signed_transfer.hpp>
#include <openpfc/runtime/common/distributed_bounds.hpp>
#include <openpfc/runtime/common/distributed_grain.hpp>
#include <openpfc/runtime/cpu/detail/grain_tracking.hpp>

#include "fixture.hpp"
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    require(ranks == 1 || ranks == 2 || ranks == 4 || ranks == 8,
            "supported fixture rank count");
    // Pure stage regressions are independent of MPI rank count.
    {
      const Grid3D global{4, 1, 1, false, false, false, Connectivity::Six};
      std::vector<dg::Observation> obs(2);
      obs[0].partition = {global, {0, 0, 0}, {2, 1, 1}};
      obs[1].partition = {global, {2, 0, 0}, {2, 1, 1}};
      dg::detail::validate_partition_cover(obs, global);
      auto invalid = obs;
      invalid[1].partition.lower[0] = 1;
      bool rejected = false;
      try {
        dg::detail::validate_partition_cover(invalid, global);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "partition validation rejects overlap before contacts");
      invalid = obs;
      invalid.pop_back();
      rejected = false;
      try {
        dg::detail::validate_partition_cover(invalid, global);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "partition validation rejects missing global coverage");
      obs[0].components = {{1, 0, 11, 1}};
      obs[1].components = {{1, 0, 11, 1}};
      obs[0].boundary = {{1, 1, 0}};
      obs[1].boundary = {{2, 1, 0}};
      auto topology = dg::detail::index_components(obs, 2);
      const auto contacts =
          dg::detail::collect_component_contacts(obs, global, 1, topology);
      require(contacts.size() == 1, "contact stage finds the cross-rank seam");
      require(topology.root(0) == topology.root(1),
              "same-slot seam joins components independently of their local keys");
      obs[1].components[0].slot = 1;
      obs[1].boundary[0].slot = 1;
      topology = dg::detail::index_components(obs, 2);
      dg::detail::collect_component_contacts(obs, global, 1, topology);
      require(topology.root(0) != topology.root(1),
              "contact alone never merges different slots");
      obs[1].boundary[0].cell = 0;
      rejected = false;
      try {
        dg::detail::collect_component_contacts(obs, global, 1, topology);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "contact validation rejects non-owned shell cells");
    }
    for (auto connectivity : {Connectivity::Six, Connectivity::TwentySix}) {
      Fixture f(rank, ranks, connectivity);
      remapping::Options options;
      options.contact_radius = 3;
      options.epoch = 17;
      auto producer = [&] {
        return dg::observe(f.part, f.slots, f.occupied, f.seeds,
                           options.contact_radius);
      };
      auto observation = producer();
      auto plan = dg::prepare(producer, f.grains, f.slots, options, MPI_COMM_WORLD);
      require(plan.status == dg::Status::Success,
              "distributed seeded component reconciliation");
      auto oracle = tracking::reference::contact_graph(f.part.global, f.slots,
                                                       f.global_labels, 3);
      require(plan.snapshot.graph.vertices == oracle.vertices &&
                  plan.snapshot.graph.edges == oracle.edges,
              "independent global contact oracle");
      require(!plan.global_moves.empty(), "actual remapping move required");
      std::map<Id, Id> bindings;
      for (auto b : plan.bindings) bindings[b.component] = b.uid;
      std::vector<Id> resolved(observation.local_components.size());
      for (std::size_t i = 0; i < resolved.size(); ++i)
        if (observation.local_components[i])
          resolved[i] = bindings.at(observation.local_components[i]);
      require(resolved == f.labels,
              "global UID propagated into every local support sample");
      auto before = f.values;
      auto old_labels = f.labels;
      auto apply = [&](std::span<const Grain> grains,
                       std::span<const Transfer> moves) {
        return transfer_signed(f.part.local(), f.slots, f.values, resolved, grains,
                               moves);
      };
      auto staged = dg::stage(plan, MPI_COMM_WORLD, apply);
      require(staged.status == dg::Status::Success && staged.transaction.has_value(),
              "collective signed transaction");
      require(f.values == before && f.labels == old_labels,
              "original inputs unchanged before publication");
      auto cells = cell_count(f.part.local());
      for (std::size_t cell = 0; cell < cells; ++cell)
        for (Slot s = 0; s < f.slots; ++s) {
          auto i = cells * s + cell;
          if (!f.labels[i]) continue;
          auto g = std::lower_bound(plan.snapshot.grains.begin(),
                                    plan.snapshot.grains.end(), f.labels[i],
                                    [](Grain g, Id id) { return g.id < id; });
          auto j = cells * g->slot + cell;
          require(staged.transaction->labels[j] == f.labels[i] &&
                      staged.transaction->values[j] == f.values[i],
                  "UID/signed amplitude survives slot change exactly");
        }
      auto bound_plan = dg::prepare_bounds(
          [&] { return dg::observe_bounds(f.part, f.slots, f.labels, f.grains); },
          f.grains, f.slots, options, MPI_COMM_WORLD);
      require(bound_plan.status == dg::Status::Success,
              "conservative global support bounds plan");
      require(bound_plan.snapshot.graph.vertices == oracle.vertices,
              "bound graph preserves all globally present UID vertices");
      for (auto edge : oracle.edges)
        require(std::binary_search(bound_plan.snapshot.graph.edges.begin(),
                                   bound_plan.snapshot.graph.edges.end(), edge),
                "bound graph contains every exact contact across "
                "periodic/decomposition boundaries");
      if (connectivity == Connectivity::TwentySix)
        require(bound_plan.snapshot.graph.edges.size() > oracle.edges.size(),
                "disconnected periodic islands admit documented conservative false "
                "contacts");
      auto bound_transfer =
          dg::stage(bound_plan, MPI_COMM_WORLD, [&](auto grains, auto moves) {
            return transfer_signed(f.part.local(), f.slots, f.values, f.labels,
                                   grains, moves);
          });
      require(bound_transfer.status == dg::Status::Success,
              "conservative bounds signed transfer");
      for (std::size_t cell = 0; cell < cells; ++cell)
        for (Slot slot = 0; slot < f.slots; ++slot) {
          auto at = cells * slot + cell;
          if (!f.labels[at]) continue;
          auto g = std::lower_bound(bound_plan.snapshot.grains.begin(),
                                    bound_plan.snapshot.grains.end(), f.labels[at],
                                    [](Grain g, Id id) { return g.id < id; });
          auto destination = cells * g->slot + cell;
          require(bound_transfer.transaction->labels[destination] == f.labels[at] &&
                      bound_transfer.transaction->values[destination] ==
                          f.values[at],
                  "UID/amplitude exact under bound-driven slot changes");
        }
      auto wrong_bound = dg::prepare_bounds(
          [&] {
            auto b = dg::observe_bounds(f.part, f.slots, f.labels, f.grains);
            if (!b.bounds.empty()) b.bounds.front().uid = 999;
            return b;
          },
          f.grains, f.slots, options, MPI_COMM_WORLD);
      require(wrong_bound.status == dg::Status::UnknownIdentity,
              "bounds do not silently allocate unknown UIDs");
      // A single rank's failure rejects every staged result, including peers.
      auto failed = dg::stage(plan, MPI_COMM_WORLD, [&](auto grains, auto moves) {
        auto transaction = apply(grains, moves);
        if (rank == 0) transaction.status = TransferStatus::InvalidSupport;
        return transaction;
      });
      require(failed.status == dg::Status::StageFailure && !failed.transaction,
              "collective rollback after one-rank failure");
      require(f.values == before && f.labels == old_labels,
              "failed collective leaves all originals untouched");
      require(dg::stage(plan, MPI_COMM_WORLD, apply, 1).status ==
                  dg::Status::CapacityOverflow,
              "collective stage capacity failure");
      auto tampered = plan;
      if (!tampered.local_moves.empty())
        tampered.local_moves.front().destination =
            (tampered.local_moves.front().destination + 1) % f.slots;
      require(dg::stage(tampered, MPI_COMM_WORLD, apply).status ==
                  dg::Status::InvalidInput,
              "mutable local moves cannot override global plan");
      auto inconsistent = plan;
      if (rank == 0) ++inconsistent.snapshot.epoch;
      if (ranks > 1) {
        auto rejected = dg::stage(inconsistent, MPI_COMM_WORLD, apply);
        require(rejected.status == dg::Status::InvalidInput && !rejected.transaction,
                "different per-rank plan rejected before staging");
      }
      auto disabled = options;
      disabled.check_now = false;
      require(dg::prepare(producer, f.grains, f.slots, disabled, MPI_COMM_WORLD)
                      .status == dg::Status::Deferred,
              "disabled remapping control");
      // Contact triangle 101/202/303 needs three slots; slot3 grain removed.
      Fixture small = f;
      small.slots = 2;
      small.grains.pop_back();
      small.values.resize(cells * 2);
      small.labels.resize(cells * 2);
      small.seeds.resize(cells * 2);
      small.occupied.resize(cells * 2);
      auto exact_options = options;
      exact_options.method = remapping::Method::CompleteOracle;
      auto infeasible = dg::prepare(
          [&] { return dg::observe(small.part, 2, small.occupied, small.seeds, 3); },
          small.grains, 2, exact_options, MPI_COMM_WORLD);
      require(infeasible.status == dg::Status::Infeasible,
              "infeasible palette control");
      auto unseeded = dg::prepare(
          [&] {
            auto o = producer();
            for (auto &c : o.components)
              if (c.seed == 101) c.seed = 0;
            return o;
          },
          f.grains, f.slots, options, MPI_COMM_WORLD);
      require(unseeded.status == dg::Status::Unseeded,
              "unseeded global component rejected without UID invention");
      auto unknown = dg::prepare(
          [&] {
            auto o = producer();
            for (auto &c : o.components)
              if (c.seed == 101) c.seed = 999;
            return o;
          },
          f.grains, f.slots, options, MPI_COMM_WORLD);
      require(unknown.status == dg::Status::UnknownIdentity,
              "unregistered global component UID rejected");
      auto unsafe = f;
      const std::array<std::size_t, 3> touch{
          8, connectivity == Connectivity::Six ? 9u : 7u,
          connectivity == Connectivity::Six ? 9u : 7u};
      bool owns = true;
      for (int d = 0; d < 3; ++d)
        owns &= touch[d] >= f.part.lower[d] &&
                touch[d] < f.part.lower[d] + f.part.extent[d];
      if (owns) {
        auto cell =
            touch[0] - f.part.lower[0] +
            f.part.extent[0] * (touch[1] - f.part.lower[1] +
                                f.part.extent[1] * (touch[2] - f.part.lower[2]));
        unsafe.occupied[cell] = 1;
        unsafe.seeds[cell] = 202;
      }
      auto unsafe_plan = dg::prepare(
          [&] {
            return dg::observe(unsafe.part, unsafe.slots, unsafe.occupied,
                               unsafe.seeds, 3);
          },
          unsafe.grains, unsafe.slots, options, MPI_COMM_WORLD);
      require(unsafe_plan.status == dg::Status::UnsafeCadence,
              "same-slot touching UID components reject unsafe cadence");
      auto unsupported = options;
      unsupported.method = remapping::Method(99);
      require(dg::prepare(producer, f.grains, f.slots, unsupported, MPI_COMM_WORLD)
                      .status == dg::Status::InvalidInput,
              "unsupported coloring policy fails explicitly");
      require(dg::prepare_bounds(
                  [&] {
                    return dg::observe_bounds(f.part, f.slots, f.labels, f.grains);
                  },
                  f.grains, f.slots, unsupported, MPI_COMM_WORLD)
                      .status == dg::Status::InvalidInput,
              "unsupported bounds policy fails explicitly");
      auto bad_registry = f.grains;
      if (rank == 0) bad_registry[0].slot = 2;
      auto bad =
          dg::prepare(producer, bad_registry, f.slots, options, MPI_COMM_WORLD);
      require(bad.status == dg::Status::InvalidInput,
              "registry mismatch or source slot rejected collectively");
      require(dg::prepare(producer, f.grains, f.slots, options, MPI_COMM_WORLD, 1)
                      .status == dg::Status::CapacityOverflow,
              "bounded summary resource failure");
      if (rank == 0)
        std::cout << "distributed grain connectivity=" << int(connectivity)
                  << " ranks=" << ranks << " contacts=" << oracle.edges.size()
                  << " moves=" << plan.global_moves.size()
                  << " summary bytes=" << plan.exchanged_bytes << " verified\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}
