// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/decomposition/padded_sparse_halos.hpp>
#include <openpfc/kernel/decomposition/stage_preparation.hpp>
#include <openpfc/runtime/gpu/comm_sparse_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/reflection_ghosts_gpu.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {
int sample(int global_x, int j, int k, int nx, int slot) {
  const int x = std::clamp(global_x, 0, nx - 1);
  const int y = (j % 3 + 3) % 3, z = (k % 2 + 2) % 2;
  return 1000000 + 10000 * slot + x + 10 * y + 100 * z;
}
template <class Field> void initialize(Field &field, int slot) {
  using T = typename Field::value_type;
  field.with_host_view([&](T *data, std::size_t length) {
    std::fill(data, data + length, T(-999));
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 3; ++j)
        data[field.idx(0, j, k)] =
            sample(field.box().low[0], j, k, field.global_size()[0], slot);
  });
}
template <class Field> void check_faces(const Field &field, int slot) {
  field.with_host_read([&](const auto *data, std::size_t) {
    const int low = field.box().low[0], nx = field.global_size()[0];
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 3; ++j)
        for (int i = -1; i <= 1; ++i)
          CHECK(data[field.idx(i, j, k)] == sample(low + i, j, k, nx, slot));
    for (int k = 0; k < 2; ++k)
      for (int j : {-1, 3})
        CHECK(data[field.idx(0, j, k)] == sample(low, j, k, nx, slot));
    for (int k : {-1, 2})
      for (int j = 0; j < 3; ++j)
        CHECK(data[field.idx(0, j, k)] == sample(low, j, k, nx, slot));
  });
}
} // namespace

TEST_CASE("runtime device field sets prepare integer labels and reflection walls",
          "[MPI][device_stage_fields]") {
  int devices = 0;
#if defined(OPENPFC_GPU_API_HIP)
  using Space = pfc::HIPSpace;
  const bool available = hipGetDeviceCount(&devices) == hipSuccess && devices > 0;
#else
  using Space = pfc::CUDASpace;
  const bool available = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
#endif
  int all_available = available ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &all_available, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all_available) SKIP("No allocated device");
  using Tag = pfc::memory_space_to_backend_t<Space>;
  pfc::core::DataBuffer<Tag, int> dense(3);
  pfc::core::SparseVector<Tag, int> selected(std::vector<std::size_t>{2, 0},
                                             std::vector<int>{-16777219, 16777217});
  pfc::core::scatter(selected, dense.data(), dense.size());
  std::vector<int> copied(3);
  dense.copy_to_host(copied.data(), copied.size());
  CHECK(copied[0] == 16777217);
  CHECK(copied[2] == -16777219);
  CHECK_THROWS_AS(pfc::core::gather(selected, dense.data(), 2), std::runtime_error);
  CHECK_THROWS_AS(pfc::core::scatter(selected, dense.data(), 2), std::runtime_error);
  dense.copy_to_host(copied.data(), copied.size());
  CHECK(copied[0] == 16777217);
  CHECK(copied[2] == -16777219);
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const auto domain =
      pfc::domain::with_spacing({ranks, 3, 2}, {1, 1, 1}, {false, true, true});
  const pfc::decomposition::Decomposition decomp(domain, {ranks, 1, 1});
  const auto box = pfc::decomposition::local_box(decomp, rank);
  using Field = pfc::data::Field<double, Space>;
  using Exchange = pfc::comm::SparseExchange<Space, double>;
  const int slots = 2 * ranks + 3; // runtime count beyond fixed four-field packs
  std::vector<std::unique_ptr<Field>> fields;
  std::vector<std::unique_ptr<Exchange>> exchanges;
  pfc::communication::StagePreparationService<> preparation;
  std::vector<std::string> names;
  names.reserve(slots + 1);
  const auto walls = pfc::communication::reflection_walls<double>(
      pfc::communication::ReflectionParity::Even,
      pfc::communication::BoundaryCentering::CellFace);
  for (int slot = 0; slot < slots; ++slot) {
    fields.push_back(std::make_unique<Field>(domain, box, 1));
    initialize(*fields.back(), slot);
    auto halos =
        pfc::halo::make_padded_sparse_halos(*fields.back(), decomp, rank, 64 * slot);
    exchanges.push_back(std::make_unique<Exchange>(*fields.back(), std::move(halos),
                                                   rank, MPI_COMM_WORLD));
    names.push_back("phase-" + std::to_string(slot));
    preparation.bind(names.back(), *exchanges.back());
    preparation.set_reflection(names.back(), walls, *fields.back());
  }
  pfc::data::Field<int, Space> labels(domain, box, 1);
  initialize(labels, slots);
  auto label_halos =
      pfc::halo::make_padded_sparse_halos(labels, decomp, rank, 64 * slots);
  pfc::comm::SparseExchange<Space, int> label_exchange(
      labels, std::move(label_halos), rank, MPI_COMM_WORLD);
  names.push_back("labels");
  preparation.bind(names.back(), label_exchange);
  preparation.set_reflection(names.back(),
                             pfc::communication::reflection_walls<int>(
                                 pfc::communication::ReflectionParity::Even,
                                 pfc::communication::BoundaryCentering::CellFace),
                             labels);
  std::vector<std::string_view> requested;
  for (const auto &name : names) requested.push_back(name);
  pfc::communication::StagePreparationRequirements requirements;
  requirements.needs_halo_exchange = requirements.needs_boundary_update = true;
  preparation.prepare(requirements, requested);
  for (int slot = 0; slot < slots; ++slot) check_faces(*fields[slot], slot);
  check_faces(labels, slots);
  const std::string_view missing[] = {"unbound"};
  CHECK_THROWS_AS(preparation.prepare(requirements, missing), std::invalid_argument);

  // Re-prepare after an owned state change; neither transport nor reflection
  // may depend on the old host mirror or the first stage's halo content.
  for (int slot = 0; slot < slots; ++slot) initialize(*fields[slot], slot + slots);
  initialize(labels, 2 * slots);
  requirements.ordering = pfc::communication::BoundaryHaloOrder::HaloThenBoundary;
  preparation.prepare(requirements, requested);
  for (int slot = 0; slot < slots; ++slot) check_faces(*fields[slot], slot + slots);
  check_faces(labels, 2 * slots);
}
