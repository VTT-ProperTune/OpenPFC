// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/runtime/gpu/observable_reduce_gpu.hpp>

#include <algorithm>
#include <limits>

namespace {
using namespace pfc::sim::gpu;
struct StatisticsMap {
  OPENPFC_HD ReductionValues<5> operator()(double value, int, int, int) const {
    const bool valid = ::isfinite(value);
    return {{valid ? value : 0, valid ? value : INFINITY, valid ? value : -INFINITY,
             valid ? 0.0 : 1.0, valid ? value * value : 0}};
  }
};
} // namespace

TEST_CASE("fused device observables exclude padding and combine across ranks",
          "[MPI][device_observables]") {
  int count = 0;
#if defined(OPENPFC_GPU_API_HIP)
  const auto status = hipGetDeviceCount(&count);
  using Space = pfc::HIPSpace;
#else
  const auto status = cudaGetDeviceCount(&count);
  using Space = pfc::CUDASpace;
#endif
  if (!pfc::mpi::all_ranks_succeeded(status == pfc::gpuSuccess && count > 0,
                                     MPI_COMM_WORLD))
    SKIP("No allocated device");
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const auto domain = pfc::domain::create({512, 2, 2});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double, Space> field(domain, box, 2);
  field.with_host_view([&](double *data, std::size_t length) {
    std::fill(data, data + length, 1e30); // Poison halos: must not enter any sum.
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 512; ++i)
          data[field.idx(i, j, k)] =
              rank == 0 && i == 0 && j == 0 && k == 0
                  ? std::numeric_limits<double>::quiet_NaN()
                  : double(rank * 10000 + i + 1000 * j + 2000 * k);
  });
  field.sync_to_device();
  const ReductionOps<5> ops{{ReductionOp::Sum, ReductionOp::Min, ReductionOp::Max,
                             ReductionOp::Sum, ReductionOp::Sum}};
  pfc::diagnostics::Traffic traffic;
  pfc::diagnostics::Scope observe(&traffic);
  const auto got = reduce_owned_multi(field, StatisticsMap{}, ops, MPI_COMM_WORLD);
  double sum = 0, squares = 0, minimum = INFINITY, maximum = -INFINITY;
  for (int r = 0; r < ranks; ++r)
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 512; ++i) {
          if (r == 0 && i == 0 && j == 0 && k == 0) continue;
          const double x = r * 10000 + i + 1000 * j + 2000 * k;
          sum += x;
          squares += x * x;
          minimum = std::min(minimum, x);
          maximum = std::max(maximum, x);
        }
  CHECK(got[0] == sum);
  CHECK(got[1] == minimum);
  CHECK(got[2] == maximum);
  CHECK(got[3] == 1);
  CHECK(got[4] == squares);
  const auto d2h = static_cast<int>(pfc::diagnostics::Direction::DeviceToHost);
  CHECK(traffic.copies.bytes[d2h] < field.size() * sizeof(double));

  // Empty owned domains contribute the neutral elements, even if all are empty.
  pfc::data::Field<double, Space> empty;
  const auto neutral =
      reduce_owned_multi(empty, StatisticsMap{}, ops, MPI_COMM_WORLD);
  CHECK(neutral[0] == 0);
  CHECK(neutral[1] == INFINITY);
  CHECK(neutral[2] == -INFINITY);
  CHECK(neutral[3] == 0);
  CHECK(neutral[4] == 0);
  auto &local = rank == 0 ? field : empty;
  const auto mixed = reduce_owned_multi(local, StatisticsMap{}, ops, MPI_COMM_WORLD);
  CHECK(mixed[0] == 3595264);
  CHECK(mixed[1] == 1);
  CHECK(mixed[2] == 3511);
  CHECK(mixed[3] == 1);

  CHECK(pfc::mpi::all_ranks_succeeded(true, MPI_COMM_WORLD));
  CHECK_FALSE(pfc::mpi::all_ranks_succeeded(rank != ranks - 1, MPI_COMM_WORLD));
  CHECK_NOTHROW(pfc::mpi::require_all_ranks(true, MPI_COMM_WORLD, "accepted"));
  CHECK_THROWS_AS(
      pfc::mpi::require_all_ranks(rank != ranks - 1, MPI_COMM_WORLD, "rejected"),
      std::runtime_error);
}
