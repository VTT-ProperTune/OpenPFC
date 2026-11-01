// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/decomposition/owned_block_gather.hpp>

#include <cstdio>
#include <vector>

int main() {
  int ready = 0;
  MPI_Initialized(&ready);
  if (ready == 0) {
    MPI_Init(nullptr, nullptr);
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int status = 0;
  if (size != 1 || rank != 0) {
    std::fprintf(stderr, "owned-block consumer expects one rank\n");
    status = 2;
  } else {
    const pfc::Domain domain = pfc::domain::create(pfc::Int3{2, 2, 1});
    const pfc::Decomposition decomposition =
        pfc::decomposition::create(domain, pfc::Int3{1, 1, 1});
    const pfc::Box3i box = pfc::decomposition::local_box(decomposition, 0);
    std::vector<double> values(static_cast<std::size_t>(box.count()));
    for (int j = 0; j < box.size[1]; ++j) {
      for (int i = 0; i < box.size[0]; ++i) {
        values[static_cast<std::size_t>(i + box.size[0] * j)] =
            10000.0 * (box.low[0] + i) + 100.0 * (box.low[1] + j);
      }
    }
    const pfc::decomposition::OwnedBlockView view{box, values};
    const auto gathered = pfc::decomposition::gather_owned_blocks(
        decomposition, MPI_COMM_WORLD, 0,
        std::span<const pfc::decomposition::OwnedBlockView>(&view, 1));
    std::vector<double> restored(values.size(), -1.0);
    pfc::decomposition::OwnedBlockBuffer buffer{box, restored};
    pfc::decomposition::scatter_owned_blocks(
        decomposition, MPI_COMM_WORLD, 0, gathered,
        std::span<pfc::decomposition::OwnedBlockBuffer>(&buffer, 1));
    bool ok = gathered.size() == 1 && gathered[0].size() == values.size() &&
              restored == values && gathered[0] == values;
    if (ok) {
      std::printf("OpenPFC owned-block consumer OK\n");
    } else {
      std::fprintf(stderr, "owned-block consumer mismatch\n");
      status = 1;
    }
  }

  int done = 0;
  MPI_Finalized(&done);
  if (ready == 0 && done == 0) {
    MPI_Finalize();
  }
  return status;
}
