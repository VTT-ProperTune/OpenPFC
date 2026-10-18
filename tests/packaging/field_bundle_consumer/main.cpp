// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

#if defined(CHECK_CUDA) || defined(CHECK_HIP)
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#endif
#include <openpfc/kernel/checkpoint/field_bundle.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

#if defined(CHECK_CUDA)
using Space = pfc::CUDASpace;
#elif defined(CHECK_HIP)
using Space = pfc::HIPSpace;
#else
using Space = pfc::HostSpace;
#endif
using Field = pfc::data::Field<double, Space>;
using Named = pfc::checkpoint::NamedField<Space>;

void require(bool passed, const char *message) {
  if (!passed) throw std::runtime_error(message);
}

template <class F> auto actual_storage(F &field) {
  if constexpr (F::is_host_space)
    return field.vec();
  else
    return field.buffer().to_host();
}

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int argc, char **argv, int rank, int ranks) {
        require(argc == 2, "Pass a new bundle directory");
        require(ranks <= 7, "At most seven x slabs");
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
        int count = 0;
#if defined(CHECK_CUDA)
        require(cudaGetDeviceCount(&count) == cudaSuccess,
                "Cannot query CUDA devices");
#else
    require(hipGetDeviceCount(&count)==hipSuccess,"Cannot query HIP devices");
#endif
        MPI_Comm node = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                            &node);
        int local_size = 0;
        MPI_Comm_size(node, &local_size);
        MPI_Comm_free(&node);
        require(count >= local_size,
                "Need a distinct visible allocated device per local rank");
        pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);
#endif
        auto domain = pfc::domain::create({7, 5, 3});
        const std::filesystem::path path = argv[1];
        if (rank == 0) {
          Field whole(domain, pfc::Box3i::from_bounds({0, 0, 0}, {6, 4, 2}), 2);
          whole.with_host_view([&](double *data, std::size_t size) {
            std::fill(data, data + size, -17.);
            for (int k = 0; k < 3; ++k)
              for (int j = 0; j < 5; ++j)
                for (int i = 0; i < 7; ++i)
                  data[whole.idx(i, j, k)] = 100 * i + 10 * j + k + .125;
          });
          whole.sync_to_device();
          auto out = pfc::checkpoint::publish_field_bundle(
              path, std::vector<Named>{{"state", &whole}},
              pfc::checkpoint::FieldBundleState{.accepted_time = .5,
                                                .accepted_increment = 7,
                                                .method_identity =
                                                    "consumer/relaxation"},
              MPI_COMM_SELF);
          require(out.ok, "Cannot publish bundle");
          require(!whole.residency().device_needs_refresh(),
                  "Packing invalidated device");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const int base = 7 / ranks, extra = 7 % ranks;
        const int lo = rank * base + std::min(rank, extra),
                  nx = base + (rank < extra);
        Field part(domain, pfc::Box3i::from_bounds({lo, 0, 0}, {lo + nx - 1, 4, 2}),
                   1);
        part.with_host_view([](double *data, std::size_t size) {
          std::fill(data, data + size, -99.);
        });
        part.sync_to_device();
        const auto state = pfc::checkpoint::load_field_bundle(
            path, std::vector<Named>{{"state", &part}}, MPI_COMM_WORLD);
        bool match = state.accepted_time == .5 && state.accepted_increment == 7 &&
                     state.method_identity == "consumer/relaxation";
        match = match && !part.residency().device_needs_refresh();
        const auto actual = actual_storage(part);
        for (int k = -1; k < 4; ++k)
          for (int j = -1; j < 6; ++j)
            for (int i = -1; i < nx + 1; ++i) {
              const bool owned =
                  i >= 0 && i < nx && j >= 0 && j < 5 && k >= 0 && k < 3;
              const auto index =
                  (i + 1) + (j + 1) * (nx + 2) + (k + 1) * (nx + 2) * 7;
              const double expected =
                  owned ? 100 * (lo + i) + 10 * j + k + .125 : -99.;
              match = match && actual[index] == expected;
            }
        int passed = match;
        MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        require(passed == 1, "Owned values, halos or accepted state differ");
        if (rank == 0)
          std::printf("field bundle consumer passed on %d ranks\n", ranks);
        return 0;
      });
}
