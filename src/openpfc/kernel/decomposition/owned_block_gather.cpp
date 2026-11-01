// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/decomposition/owned_block_gather.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <string>

namespace pfc::decomposition {
namespace {

constexpr const char *kNull = "owned-block transfer: communicator is null";
constexpr const char *kRoot =
    "owned-block transfer: root is outside the communicator";
constexpr const char *kRanks =
    "owned-block transfer: rank count does not match the decomposition";
constexpr const char *kBox =
    "owned-block transfer: owned box does not match the decomposition";
constexpr const char *kLength =
    "owned-block transfer: block length does not match its box";
constexpr const char *kFields =
    "owned-block transfer: field count disagrees across ranks";
constexpr const char *kMeta = "owned-block transfer: owned-box metadata disagrees";
constexpr const char *kRootDisagree =
    "owned-block transfer: root disagrees across ranks";
constexpr const char *kGlobalEmpty =
    "owned-block transfer: global buffer is not empty off the root";
constexpr const char *kGlobalCount =
    "owned-block transfer: global field count does not match the blocks";
constexpr const char *kGlobalLength =
    "owned-block transfer: global buffer length does not match the domain";
constexpr const char *kMpiCount = "owned-block transfer: MPI count exceeds INT_MAX";
constexpr const char *kPeer = "owned-block transfer rejected on a peer rank";

enum class Fault : int { Ok = 0, Argument = 1, Overflow = 2 };

struct FaultState {
  Fault code = Fault::Ok;
  std::string message;

  void argument(const char *text) {
    if (code == Fault::Ok) {
      code = Fault::Argument;
      message = text;
    }
  }

  void overflow(const char *text) {
    code = Fault::Overflow;
    message = text;
  }
};

struct Draft {
  FaultState fault;
  MPI_Comm comm = MPI_COMM_NULL;
  const Decomposition *decomposition = nullptr;
  int rank = 0;
  int nprocs = 0;
  int root = 0;
  int nfields = 0;
  std::array<int, 3> domain_size{};
};

struct Layout {
  int rank = 0;
  int nprocs = 0;
  int root = 0;
  int nfields = 0;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  std::size_t global_count = 0;
  int total = 0;
  std::vector<int> counts;
  std::vector<int> displs;
  std::vector<int> meta;
};

void raise_if_fault(MPI_Comm comm, const FaultState &local) {
  int code = static_cast<int>(local.code);
  int reduced = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&code, &reduced, 1, MPI_INT, MPI_MAX, comm),
      "MPI_Allreduce owned-block transfer");
  if (reduced == static_cast<int>(Fault::Ok)) {
    return;
  }
  if (!local.message.empty()) {
    if (local.code == Fault::Overflow) {
      throw std::overflow_error(local.message);
    }
    throw std::invalid_argument(local.message);
  }
  if (reduced == static_cast<int>(Fault::Overflow)) {
    throw std::overflow_error(kPeer);
  }
  throw std::invalid_argument(kPeer);
}

Draft start_transfer(const Decomposition &decomposition, MPI_Comm comm, int root,
                     const std::vector<Box3i> &boxes,
                     const std::vector<std::size_t> &lengths) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument(kNull);
  }
  Draft draft;
  draft.comm = comm;
  draft.decomposition = &decomposition;
  draft.root = root;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &draft.rank),
                               "MPI_Comm_rank owned-block transfer");
  pfc::mpi::throw_on_mpi_error(MPI_Comm_size(comm, &draft.nprocs),
                               "MPI_Comm_size owned-block transfer");
  if (boxes.size() > static_cast<std::size_t>(INT_MAX)) {
    draft.fault.overflow(kMpiCount);
  } else {
    draft.nfields = static_cast<int>(boxes.size());
  }
  if (root < 0 || root >= draft.nprocs) {
    draft.fault.argument(kRoot);
  }
  if (draft.nprocs != get_num_domains(decomposition)) {
    draft.fault.argument(kRanks);
    return draft;
  }
  const Box3i owned = local_box(decomposition, draft.rank);
  draft.domain_size = get_global_domain(decomposition).size;
  if (boxes.size() != lengths.size()) {
    draft.fault.argument(kLength);
    return draft;
  }
  for (std::size_t field = 0; field < boxes.size(); ++field) {
    if (boxes[field] != owned) {
      draft.fault.argument(kBox);
      continue;
    }
    try {
      const std::size_t count =
          pfc::mpi::checked_local_extent_product(owned.size, "owned-block transfer");
      if (lengths[field] != count) {
        draft.fault.argument(kLength);
      }
      if (count > static_cast<std::size_t>(INT_MAX)) {
        draft.fault.overflow(kMpiCount);
      }
    } catch (const std::overflow_error &) {
      draft.fault.overflow(kMpiCount);
    } catch (const std::invalid_argument &) {
      draft.fault.argument(kBox);
    }
  }
  try {
    (void)pfc::mpi::checked_local_extent_product(draft.domain_size,
                                                 "owned-block transfer");
  } catch (const std::overflow_error &) {
    draft.fault.overflow(kMpiCount);
  } catch (const std::invalid_argument &) {
    draft.fault.argument(kMeta);
  }
  return draft;
}

void note_scatter_global(Draft &draft,
                         const std::vector<std::vector<double>> &global) {
  if (draft.fault.code != Fault::Ok) {
    return;
  }
  if (draft.root < 0 || draft.root >= draft.nprocs) {
    return;
  }
  if (draft.rank == draft.root) {
    if (global.size() != static_cast<std::size_t>(draft.nfields)) {
      draft.fault.argument(kGlobalCount);
      return;
    }
    std::size_t count = 0;
    try {
      count = pfc::mpi::checked_local_extent_product(draft.domain_size,
                                                     "owned-block transfer");
    } catch (const std::overflow_error &) {
      draft.fault.overflow(kMpiCount);
      return;
    } catch (const std::invalid_argument &) {
      draft.fault.argument(kMeta);
      return;
    }
    for (const auto &field : global) {
      if (field.size() != count) {
        draft.fault.argument(kGlobalLength);
        return;
      }
    }
  } else if (!global.empty()) {
    draft.fault.argument(kGlobalEmpty);
  }
}

bool meta_matches(const int *meta, const Box3i &expected, int nx, int ny, int nz) {
  const int lx = meta[0];
  const int ly = meta[1];
  const int lz = meta[2];
  const int ox = meta[3];
  const int oy = meta[4];
  const int oz = meta[5];
  if (lx < 1 || ly < 1 || lz < 1 || ox < 0 || oy < 0 || oz < 0) {
    return false;
  }
  const long long hx = static_cast<long long>(ox) + lx - 1;
  const long long hy = static_cast<long long>(oy) + ly - 1;
  const long long hz = static_cast<long long>(oz) + lz - 1;
  if (hx > INT_MAX || hy > INT_MAX || hz > INT_MAX) {
    return false;
  }
  if (static_cast<long long>(ox) + lx > nx || static_cast<long long>(oy) + ly > ny ||
      static_cast<long long>(oz) + lz > nz) {
    return false;
  }
  const Box3i got =
      Box3i::from_bounds({ox, oy, oz}, {static_cast<int>(hx), static_cast<int>(hy),
                                        static_cast<int>(hz)});
  return got == expected;
}

Layout finish_layout(Draft &draft) {
  raise_if_fault(draft.comm, draft.fault);

  const int nx = draft.domain_size[0];
  const int ny = draft.domain_size[1];
  const int nz = draft.domain_size[2];
  const int header[5] = {draft.nfields, draft.root, nx, ny, nz};
  std::vector<int> all_header(static_cast<std::size_t>(draft.nprocs) * 5);
  pfc::mpi::throw_on_mpi_error(
      MPI_Allgather(header, 5, MPI_INT, all_header.data(), 5, MPI_INT, draft.comm),
      "MPI_Allgather owned-block transfer header");

  bool fields_differ = false;
  bool root_differs = false;
  bool domain_differs = false;
  for (int rank = 0; rank < draft.nprocs; ++rank) {
    const int *item = all_header.data() + static_cast<std::size_t>(rank) * 5;
    if (item[0] != draft.nfields) {
      fields_differ = true;
    }
    if (item[1] != draft.root) {
      root_differs = true;
    }
    if (item[2] != nx || item[3] != ny || item[4] != nz) {
      domain_differs = true;
    }
  }
  if (fields_differ) {
    draft.fault.argument(kFields);
  } else if (root_differs) {
    draft.fault.argument(kRootDisagree);
  } else if (domain_differs) {
    draft.fault.argument(kMeta);
  }
  raise_if_fault(draft.comm, draft.fault);

  const Box3i owned = local_box(*draft.decomposition, draft.rank);
  const int meta[6] = {owned.size[0], owned.size[1], owned.size[2],
                       owned.low[0],  owned.low[1],  owned.low[2]};
  Layout layout;
  layout.rank = draft.rank;
  layout.nprocs = draft.nprocs;
  layout.root = draft.root;
  layout.nfields = draft.nfields;
  layout.nx = nx;
  layout.ny = ny;
  layout.nz = nz;
  layout.meta.resize(static_cast<std::size_t>(draft.nprocs) * 6);
  pfc::mpi::throw_on_mpi_error(
      MPI_Allgather(meta, 6, MPI_INT, layout.meta.data(), 6, MPI_INT, draft.comm),
      "MPI_Allgather owned-block transfer boxes");
  for (int rank = 0; rank < draft.nprocs; ++rank) {
    const int *item = layout.meta.data() + static_cast<std::size_t>(rank) * 6;
    const Box3i expected = local_box(*draft.decomposition, rank);
    if (!meta_matches(item, expected, nx, ny, nz)) {
      draft.fault.argument(kMeta);
    }
  }
  raise_if_fault(draft.comm, draft.fault);

  layout.global_count = pfc::mpi::checked_local_extent_product(
      draft.domain_size, "owned-block transfer");
  layout.counts.resize(static_cast<std::size_t>(draft.nprocs));
  layout.displs.resize(static_cast<std::size_t>(draft.nprocs));
  int total = 0;
  for (int rank = 0; rank < draft.nprocs; ++rank) {
    const int *item = layout.meta.data() + static_cast<std::size_t>(rank) * 6;
    const std::array<int, 3> extent{item[0], item[1], item[2]};
    const std::size_t cells =
        pfc::mpi::checked_local_extent_product(extent, "owned-block transfer");
    const int count = pfc::mpi::ensure_mpi_int_count(cells, "owned-block transfer");
    if (count > INT_MAX - total) {
      throw std::overflow_error(kMpiCount);
    }
    layout.displs[static_cast<std::size_t>(rank)] = total;
    layout.counts[static_cast<std::size_t>(rank)] = count;
    total += count;
  }
  if (static_cast<std::size_t>(total) != layout.global_count) {
    throw std::invalid_argument(kMeta);
  }
  layout.total = total;
  return layout;
}

void place_brick(const double *src, int lx, int ly, int lz, int ox, int oy, int oz,
                 int nx, int ny, double *dst) {
  const auto snx = static_cast<std::size_t>(nx);
  const auto sny = static_cast<std::size_t>(ny);
  const auto slx = static_cast<std::size_t>(lx);
  const auto sly = static_cast<std::size_t>(ly);
  for (int k = 0; k < lz; ++k) {
    for (int j = 0; j < ly; ++j) {
      const std::size_t src_at =
          (static_cast<std::size_t>(j) + static_cast<std::size_t>(k) * sly) * slx;
      const std::size_t dst_at =
          static_cast<std::size_t>(ox) + (static_cast<std::size_t>(oy + j) +
                                          static_cast<std::size_t>(oz + k) * sny) *
                                             snx;
      std::copy_n(src + src_at, slx, dst + dst_at);
    }
  }
}

void extract_brick(const double *src, int lx, int ly, int lz, int ox, int oy, int oz,
                   int nx, int ny, double *dst) {
  const auto snx = static_cast<std::size_t>(nx);
  const auto sny = static_cast<std::size_t>(ny);
  const auto slx = static_cast<std::size_t>(lx);
  const auto sly = static_cast<std::size_t>(ly);
  for (int k = 0; k < lz; ++k) {
    for (int j = 0; j < ly; ++j) {
      const std::size_t dst_at =
          (static_cast<std::size_t>(j) + static_cast<std::size_t>(k) * sly) * slx;
      const std::size_t src_at =
          static_cast<std::size_t>(ox) + (static_cast<std::size_t>(oy + j) +
                                          static_cast<std::size_t>(oz + k) * sny) *
                                             snx;
      std::copy_n(src + src_at, slx, dst + dst_at);
    }
  }
}

std::vector<Box3i> boxes_of(std::span<const OwnedBlockView> blocks) {
  std::vector<Box3i> boxes;
  boxes.reserve(blocks.size());
  for (const OwnedBlockView &block : blocks) {
    boxes.push_back(block.box);
  }
  return boxes;
}

std::vector<std::size_t> lengths_of(std::span<const OwnedBlockView> blocks) {
  std::vector<std::size_t> lengths;
  lengths.reserve(blocks.size());
  for (const OwnedBlockView &block : blocks) {
    lengths.push_back(block.values.size());
  }
  return lengths;
}

std::vector<Box3i> boxes_of(std::span<const OwnedBlockBuffer> blocks) {
  std::vector<Box3i> boxes;
  boxes.reserve(blocks.size());
  for (const OwnedBlockBuffer &block : blocks) {
    boxes.push_back(block.box);
  }
  return boxes;
}

std::vector<std::size_t> lengths_of(std::span<const OwnedBlockBuffer> blocks) {
  std::vector<std::size_t> lengths;
  lengths.reserve(blocks.size());
  for (const OwnedBlockBuffer &block : blocks) {
    lengths.push_back(block.values.size());
  }
  return lengths;
}

} // namespace

std::vector<std::vector<double>>
gather_owned_blocks(const Decomposition &decomposition, MPI_Comm comm, int root,
                    std::span<const OwnedBlockView> blocks) {
  const std::vector<Box3i> boxes = boxes_of(blocks);
  const std::vector<std::size_t> lengths = lengths_of(blocks);
  Draft draft = start_transfer(decomposition, comm, root, boxes, lengths);
  Layout layout = finish_layout(draft);
  if (layout.nfields == 0) {
    return {};
  }

  std::vector<std::vector<double>> gathered;
  if (layout.rank == layout.root) {
    gathered.resize(static_cast<std::size_t>(layout.nfields));
  }
  std::vector<double> packed;
  if (layout.rank == layout.root) {
    packed.resize(static_cast<std::size_t>(layout.total));
  }
  for (int field = 0; field < layout.nfields; ++field) {
    const int count = layout.counts[static_cast<std::size_t>(layout.rank)];
    pfc::mpi::throw_on_mpi_error(
        MPI_Gatherv(blocks[static_cast<std::size_t>(field)].values.data(), count,
                    MPI_DOUBLE, layout.rank == layout.root ? packed.data() : nullptr,
                    layout.counts.data(), layout.displs.data(), MPI_DOUBLE,
                    layout.root, comm),
        "MPI_Gatherv owned-block transfer");
    if (layout.rank != layout.root) {
      continue;
    }
    gathered[static_cast<std::size_t>(field)].assign(layout.global_count, 0.0);
    for (int rank = 0; rank < layout.nprocs; ++rank) {
      const int *meta = layout.meta.data() + static_cast<std::size_t>(rank) * 6;
      place_brick(packed.data() + layout.displs[static_cast<std::size_t>(rank)],
                  meta[0], meta[1], meta[2], meta[3], meta[4], meta[5], layout.nx,
                  layout.ny, gathered[static_cast<std::size_t>(field)].data());
    }
  }
  if (layout.rank != layout.root) {
    return {};
  }
  return gathered;
}

void scatter_owned_blocks(const Decomposition &decomposition, MPI_Comm comm,
                          int root, const std::vector<std::vector<double>> &global,
                          std::span<OwnedBlockBuffer> blocks) {
  const std::vector<Box3i> boxes = boxes_of(blocks);
  const std::vector<std::size_t> lengths = lengths_of(blocks);
  Draft draft = start_transfer(decomposition, comm, root, boxes, lengths);
  note_scatter_global(draft, global);
  Layout layout = finish_layout(draft);
  if (layout.nfields == 0) {
    return;
  }

  std::vector<double> packed;
  if (layout.rank == layout.root) {
    packed.resize(static_cast<std::size_t>(layout.total));
  }
  for (int field = 0; field < layout.nfields; ++field) {
    if (layout.rank == layout.root) {
      for (int rank = 0; rank < layout.nprocs; ++rank) {
        const int *meta = layout.meta.data() + static_cast<std::size_t>(rank) * 6;
        extract_brick(global[static_cast<std::size_t>(field)].data(), meta[0],
                      meta[1], meta[2], meta[3], meta[4], meta[5], layout.nx,
                      layout.ny,
                      packed.data() + layout.displs[static_cast<std::size_t>(rank)]);
      }
    }
    const int count = layout.counts[static_cast<std::size_t>(layout.rank)];
    pfc::mpi::throw_on_mpi_error(
        MPI_Scatterv(layout.rank == layout.root ? packed.data() : nullptr,
                     layout.counts.data(), layout.displs.data(), MPI_DOUBLE,
                     blocks[static_cast<std::size_t>(field)].values.data(), count,
                     MPI_DOUBLE, layout.root, comm),
        "MPI_Scatterv owned-block transfer");
  }
}

} // namespace pfc::decomposition
