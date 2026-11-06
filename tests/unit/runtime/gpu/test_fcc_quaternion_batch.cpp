// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fcc_quaternion_batch.cpp
 * @brief Device FCC batch reads a borrowed 4-component quaternion brick.
 *
 * `FccPoints::borrow_quaternion_fields` and `FccPoints::evaluate` are the
 * shipped path. The nine-double `set_orientations` path stays. Matching
 * owned boxes do not move orientation data across ranks.
 */

#include "test_helpers.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>
#include <openpfc/mechanics/quaternion_orientation.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/fcc_points.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include <mpi.h>

using namespace pfc::finite_strain;

#if defined(OPENPFC_TEST_FCC_HIP)
using Tag = pfc::backend::HIPTag;
using Space = pfc::HIPSpace;
#else
using Tag = pfc::backend::CUDATag;
using Space = pfc::CUDASpace;
#endif

using Points = gpu::FccPoints<Tag>;
using Field = pfc::data::Field<double, Space>;
using Buffer = pfc::core::DataBuffer<Tag, double>;

namespace {

using Quat = std::array<double, 4>;

struct GridEnv {
  std::string previous;
  bool had_previous = false;
  explicit GridEnv(int nproc) {
    if (const char *value = std::getenv("OPENPFC_FD_PROC_GRID")) {
      previous = value;
      had_previous = true;
    }
    const char *grid = "1,1,1";
    if (nproc == 2) grid = "1,1,2";
    if (nproc == 4) grid = "1,1,4";
    REQUIRE(setenv("OPENPFC_FD_PROC_GRID", grid, 1) == 0);
  }
  ~GridEnv() {
    if (had_previous)
      (void)setenv("OPENPFC_FD_PROC_GRID", previous.c_str(), 1);
    else
      (void)unsetenv("OPENPFC_FD_PROC_GRID");
  }
};

Quat axis_angle(double x, double y, double z, double angle) {
  const double norm = std::sqrt(x * x + y * y + z * z);
  const double v = std::sin(angle / 2.0) / norm;
  return {std::cos(angle / 2.0), x * v, y * v, z * v};
}

Quat negate(Quat q) {
  for (double &c : q) c = -c;
  return q;
}

std::vector<Quat> cubic_symmetries() {
  std::vector<Quat> quaternions{{1, 0, 0, 0}};
  for (const auto &axis : std::array<std::array<double, 3>, 3>{
           std::array<double, 3>{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}) {
    for (double k : {0.5, 1.0, 1.5}) {
      quaternions.push_back(
          axis_angle(axis[0], axis[1], axis[2], k * std::numbers::pi));
    }
  }
  for (const auto &axis : std::array<std::array<double, 3>, 4>{
           std::array<double, 3>{1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {-1, 1, 1}}) {
    for (double k : {2.0 / 3.0, 4.0 / 3.0}) {
      quaternions.push_back(
          axis_angle(axis[0], axis[1], axis[2], k * std::numbers::pi));
    }
  }
  for (const auto &axis :
       std::array<std::array<double, 3>, 6>{std::array<double, 3>{1, 1, 0},
                                            {1, -1, 0},
                                            {1, 0, 1},
                                            {1, 0, -1},
                                            {0, 1, 1},
                                            {0, 1, -1}}) {
    quaternions.push_back(axis_angle(axis[0], axis[1], axis[2], std::numbers::pi));
  }
  return quaternions;
}

bool device_present() {
#if defined(OPENPFC_TEST_FCC_HIP)
  return pfc::gpu::test::is_hip_available();
#else
  return pfc::gpu::test::is_cuda_available();
#endif
}

int device_count() {
  int count = 0;
#if defined(OPENPFC_TEST_FCC_HIP)
  if (hipGetDeviceCount(&count) != hipSuccess) return 0;
#else
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
#endif
  return count;
}

struct Owned {
  pfc::Domain domain;
  pfc::Box3i box;
  int rank = 0;
  int nproc = 1;
};

Owned owned_brick() {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE((nproc == 1 || nproc == 2 || nproc == 4));
  GridEnv grid(nproc);
  const auto domain = pfc::domain::create(pfc::Int3{8, 4, 8});
  const auto decomp = pfc::decomposition::create(domain, nproc);
  return {domain, pfc::decomposition::local_box(decomp, rank), rank, nproc};
}

bool to_crystal(const Quat &q, Tensor2 &orientation) {
  const double data[4] = {q[0], q[1], q[2], q[3]};
  return quaternion_sample_to_crystal(data, orientation);
}

Quat quaternion_at(const std::vector<Quat> &symmetries, std::size_t point) {
  const auto slot = point % 26;
  if (slot < 24) {
    Quat q = symmetries[slot];
    if ((point / 26) % 2 == 1) q = negate(q);
    return q;
  }
  Quat q = axis_angle(1.0, 2.0, -3.0, 1.13);
  if (slot == 25) q = negate(q);
  return q;
}

Tensor2 deformation() {
  Tensor2 value = identity2();
  value(0, 0) = 1.003;
  value(0, 1) = 0.002;
  value(1, 2) = 0.001;
  return value;
}

void upload(Field &field, const std::vector<double> &owned_values, double halo) {
  field.with_host_view([&](double *data, std::size_t n) {
    std::fill(data, data + n, halo);
    const auto nloc = field.local_size();
    std::size_t p = 0;
    for (int k = 0; k < nloc[2]; ++k) {
      for (int j = 0; j < nloc[1]; ++j) {
        for (int i = 0; i < nloc[0]; ++i, ++p) {
          data[field.idx(i, j, k)] = owned_values[p];
        }
      }
    }
  });
  field.sync_to_device();
}

double component(const std::vector<double> &packed, int c, std::size_t n,
                 std::size_t p) {
  return packed[static_cast<std::size_t>(c) * n + p];
}

void consider(double left, double right, double tol, double &peak, int &fail) {
  const double gap = std::abs(left - right);
  peak = std::max(peak, gap);
  if (!(gap < tol)) fail = 1;
}

} // namespace

TEST_CASE("Borrowed quaternion FCC batch matches the nine-double path",
          "[constitutive][fcc][quaternion][device]") {
  if (!device_present()) SKIP("GPU unavailable");
  const Owned owned = owned_brick();
  const auto symmetries = cubic_symmetries();
  REQUIRE(symmetries.size() == 24);
  const std::size_t count = static_cast<std::size_t>(owned.box.count());
  REQUIRE(count >= 26);

  std::vector<Quat> quats(count);
  std::vector<Tensor2> matrices(count);
  for (std::size_t p = 0; p < count; ++p) {
    quats[p] = quaternion_at(symmetries, p);
    REQUIRE(to_crystal(quats[p], matrices[p]));
    REQUIRE(crystal::proper_rotation(matrices[p]));
  }

  constexpr int halo = 1;
  std::array<Field, 4> fields{
      Field(owned.domain, owned.box, halo), Field(owned.domain, owned.box, halo),
      Field(owned.domain, owned.box, halo), Field(owned.domain, owned.box, halo)};
  for (int c = 0; c < 4; ++c) {
    std::vector<double> channel(count);
    for (std::size_t p = 0; p < count; ++p) channel[p] = quats[p][c];
    upload(fields[c], channel, 7.0);
  }
  const std::size_t field_bytes = 4 * fields[0].size() * sizeof(double);

  const FccSlipParameters parameters{2,   1,    0.5, 0.001, 5, 2.25,
                                     0.1, 0.02, 0.2, 0.05,  1, 1.4};
  const Tensor2 F = deformation();
  const auto state = fcc_identity_state(parameters);
  const auto aligned = try_evaluate_fcc(parameters, identity2(), state, F);
  REQUIRE(aligned.status == FccStatus::Success);
  const auto &aligned_value = aligned.value;

  std::vector<double> packed_f(9 * count);
  std::vector<Tensor2> orientations(count);
  for (std::size_t p = 0; p < count; ++p) {
    orientations[p] = matrices[p];
    for (int c = 0; c < 9; ++c) {
      packed_f[c * count + p] = F(c / 3, c % 3);
    }
  }
  Buffer deformation_buffer(9 * count);
  deformation_buffer.copy_from_host(packed_f);

  Points matrix(count, parameters);
  matrix.set_orientations(orientations);
  REQUIRE(matrix.orientation.size() == 9 * count);
  const auto matrix_bytes = matrix.storage_bytes();
  REQUIRE(matrix.evaluate(deformation_buffer).status == FccStatus::Success);
  const auto matrix_piola = matrix.piola.to_host();
  const auto matrix_tangent = matrix.tangent.to_host();

  Points borrowed(count, parameters);
  auto owner = std::make_shared<int>(1);
  std::array<const Field *, 4> views{&fields[0], &fields[1], &fields[2], &fields[3]};
  borrowed.borrow_quaternion_fields(views, owned.domain, owned.box, owner);
  owner.reset();
  REQUIRE(borrowed.orientation.size() == 0);
  const auto quaternion_bytes = borrowed.storage_bytes();
  REQUIRE(matrix_bytes - quaternion_bytes == 9 * count * sizeof(double));
  REQUIRE(borrowed.evaluate(deformation_buffer).status == FccStatus::Success);
  const auto quaternion_piola = borrowed.piola.to_host();
  const auto quaternion_tangent = borrowed.tangent.to_host();

  double direct_peak = 0.0;
  double host_peak = 0.0;
  double symmetry_peak = 0.0;
  int direct_fail = 0;
  int host_fail = 0;
  int symmetry_fail = 0;
  int saw_cubic = 0;
  int saw_negative = 0;
  int saw_random = 0;
  for (std::size_t p = 0; p < count; ++p) {
    const auto host = try_evaluate_fcc(parameters, matrices[p], state, F);
    if (host.status != FccStatus::Success) host_fail = 1;
    for (int c = 0; c < 9; ++c) {
      consider(component(quaternion_piola, c, count, p),
               component(matrix_piola, c, count, p), 2e-10, direct_peak,
               direct_fail);
      if (host.status == FccStatus::Success) {
        consider(component(quaternion_piola, c, count, p),
                 host.value.update.piola(c / 3, c % 3), 2e-10, host_peak, host_fail);
      }
    }
    for (int c = 0; c < 81; ++c) {
      const int i = c / 27;
      const int j = (c / 9) % 3;
      const int k = (c / 3) % 3;
      const int l = c % 3;
      consider(component(quaternion_tangent, c, count, p),
               component(matrix_tangent, c, count, p), 2e-8, direct_peak,
               direct_fail);
      if (host.status == FccStatus::Success) {
        consider(component(quaternion_tangent, c, count, p),
                 host.value.tangent(i, j, k, l), 2e-8, host_peak, host_fail);
      }
    }
    if (p % 26 < 24) {
      saw_cubic = 1;
      if ((p / 26) % 2 == 1) saw_negative = 1;
      for (int c = 0; c < 9; ++c) {
        consider(component(quaternion_piola, c, count, p),
                 aligned_value.update.piola(c / 3, c % 3), 5e-9, symmetry_peak,
                 symmetry_fail);
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              const int c = ((i * 3 + j) * 3 + k) * 3 + l;
              consider(component(quaternion_tangent, c, count, p),
                       aligned_value.tangent(i, j, k, l), 5e-8, symmetry_peak,
                       symmetry_fail);
            }
          }
        }
      }
    } else {
      saw_random = 1;
    }
  }

  REQUIRE(saw_cubic == 1);
  REQUIRE(saw_negative == 1);
  REQUIRE(saw_random == 1);

  constexpr int repeats = 8;
  const auto quat_start = std::chrono::steady_clock::now();
  for (int step = 0; step < repeats; ++step) {
    REQUIRE(borrowed.evaluate(deformation_buffer).status == FccStatus::Success);
  }
  const auto quat_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - quat_start)
                           .count();
  const auto matrix_start = std::chrono::steady_clock::now();
  for (int step = 0; step < repeats; ++step) {
    REQUIRE(matrix.evaluate(deformation_buffer).status == FccStatus::Success);
  }
  const auto matrix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - matrix_start)
                             .count();
  const auto copies =
      static_cast<std::size_t>(1 + repeats) * sizeof(unsigned long long);
  REQUIRE(borrowed.scalar_transfer_bytes == copies);
  REQUIRE(matrix.scalar_transfer_bytes == copies);
  REQUIRE(quat_ns > 0);
  REQUIRE(matrix_ns > 0);
  REQUIRE(borrowed.integrations == count * static_cast<std::size_t>(1 + repeats));

  int local_flags[5] = {direct_fail, host_fail, symmetry_fail, saw_cubic,
                        saw_random};
  int global_flags[5] = {};
  double local_peaks[3] = {direct_peak, host_peak, symmetry_peak};
  double global_peaks[3] = {};
  MPI_Allreduce(local_flags, global_flags, 5, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(local_peaks, global_peaks, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  WARN("rank=" << owned.rank << " nproc=" << owned.nproc
               << " devices=" << device_count() << " points=" << count
               << " matrix_bytes=" << matrix_bytes << " quaternion_bytes="
               << quaternion_bytes << " field_bytes=" << field_bytes
               << " status_copy_bytes=" << borrowed.scalar_transfer_bytes
               << " mpi_orientation_bytes=0"
               << " matrix_ns=" << matrix_ns << " quaternion_ns=" << quat_ns
               << " direct_peak=" << global_peaks[0] << " host_peak="
               << global_peaks[1] << " symmetry_peak=" << global_peaks[2]);
  REQUIRE(global_flags[0] == 0);
  REQUIRE(global_flags[1] == 0);
  REQUIRE(global_flags[2] == 0);
  REQUIRE(global_flags[3] == 1);
  REQUIRE(global_flags[4] == 1);
  REQUIRE(global_peaks[0] < 2e-8);
  REQUIRE(global_peaks[1] < 2e-8);
  REQUIRE(global_peaks[2] < 5e-8);
  REQUIRE(field_bytes == 4 * fields[0].size() * sizeof(double));
}

TEST_CASE("Quaternion FCC borrow rejects a bad source and keeps history",
          "[constitutive][fcc][quaternion][device]") {
  if (!device_present()) SKIP("GPU unavailable");
  const Owned owned = owned_brick();
  const std::size_t count = static_cast<std::size_t>(owned.box.count());
  constexpr int halo = 1;
  std::array<Field, 4> fields{
      Field(owned.domain, owned.box, halo), Field(owned.domain, owned.box, halo),
      Field(owned.domain, owned.box, halo), Field(owned.domain, owned.box, halo)};
  const Quat q = axis_angle(0.0, 0.0, 1.0, std::numbers::pi / 2.0);
  for (int c = 0; c < 4; ++c) {
    upload(fields[c], std::vector<double>(count, q[c]), 7.0);
  }
  const FccSlipParameters parameters{2,   1,    0.5, 0.001, 5, 2.25,
                                     0.1, 0.02, 0.2, 0.05,  1, 1.4};
  const Tensor2 F = deformation();
  std::vector<double> packed_f(9 * count, 0.0);
  for (std::size_t p = 0; p < count; ++p) {
    for (int c = 0; c < 9; ++c) packed_f[c * count + p] = F(c / 3, c % 3);
  }
  Buffer deformation_buffer(9 * count);
  deformation_buffer.copy_from_host(packed_f);
  std::array<const Field *, 4> views{&fields[0], &fields[1], &fields[2], &fields[3]};
  auto owner = std::make_shared<int>(1);

  Points missing(count, parameters);
  REQUIRE_THROWS_AS(missing.borrow_quaternion_fields(views, owned.domain, owned.box,
                                                     std::shared_ptr<const void>{}),
                    std::invalid_argument);
  auto shifted = pfc::domain::create(pfc::GridSize(pfc::Int3{8, 4, 8}),
                                     pfc::PhysicalOrigin({0.5, 0.5, 0.5}),
                                     pfc::GridSpacing({1.0, 1.0, 1.0}));
  REQUIRE_THROWS_AS(
      missing.borrow_quaternion_fields(views, shifted, owned.box, owner),
      std::invalid_argument);
  auto other = owned.box;
  other.low[0] += 1;
  other.high[0] += 1;
  REQUIRE_THROWS_AS(
      missing.borrow_quaternion_fields(views, owned.domain, other, owner),
      std::invalid_argument);
  Field wide(owned.domain, owned.box, halo + 1);
  auto mismatched = views;
  mismatched[1] = &wide;
  REQUIRE_THROWS_AS(
      missing.borrow_quaternion_fields(mismatched, owned.domain, owned.box, owner),
      std::invalid_argument);
  auto absent = views;
  absent[0] = nullptr;
  REQUIRE_THROWS_AS(
      missing.borrow_quaternion_fields(absent, owned.domain, owned.box, owner),
      std::invalid_argument);
  REQUIRE_THROWS_AS(missing.evaluate(deformation_buffer), std::logic_error);

  Points points(count, parameters);
  points.borrow_quaternion_fields(views, owned.domain, owned.box, owner);
  REQUIRE(points.evaluate(deformation_buffer).status == FccStatus::Success);
  points.accept_increment();
  const auto committed = points.committed.to_host();

  upload(fields[0], std::vector<double>(count, 2.0), 7.0);
  const auto refused = points.evaluate(deformation_buffer);
  REQUIRE(refused.status == FccStatus::Orientation);
  REQUIRE(refused.point == 0);
  REQUIRE_THROWS_AS(points.accept_increment(), std::logic_error);
  REQUIRE(points.committed.to_host() == committed);

  std::vector<double> nan_channel(count, q[0]);
  nan_channel[0] = std::numeric_limits<double>::quiet_NaN();
  upload(fields[0], nan_channel, 7.0);
  const auto nonfinite = points.evaluate(deformation_buffer);
  REQUIRE(nonfinite.status == FccStatus::Orientation);
  REQUIRE(nonfinite.point == 0);
  REQUIRE(points.committed.to_host() == committed);
}

int main(int argc, char **argv) {
  int started = 0;
  MPI_Initialized(&started);
  if (started == 0) MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int devices = device_count();
  if (devices > 0) {
#if defined(OPENPFC_TEST_FCC_HIP)
    hipSetDevice(rank % devices);
#else
    cudaSetDevice(rank % devices);
#endif
  }
  const int code = Catch::Session().run(argc, argv);
  int finished = 0;
  MPI_Finalized(&finished);
  if (finished == 0) MPI_Finalize();
  return code;
}
