// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_device_halo_element_types.cpp
 * @brief Device `HaloExchange` of `float` and `std::uint32_t` fields.
 *
 * Every halo cell must equal the periodic owner's value for Faces (6 faces)
 * and Full (all 26 directions), on the contiguous, host-staged and subarray
 * transports. `std::uint32_t` values above 2^24 and `0xFFFFFFFF` must arrive
 * exactly. Covers 1, 2 (`2x1x1`) and 4 (`2x2x1`) ranks.
 */

#include <catch2/catch_all.hpp>
#include <mpi.h>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>

namespace {

#if defined(OpenPFC_ENABLE_CUDA)
using Space = pfc::CUDASpace;
constexpr const char *kForcePacked = "OPENPFC_CUDA_FORCE_PACKED_HALO";
constexpr const char *kSubarray = "OPENPFC_CUDA_USE_SUBARRAY_HALO";
bool device_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
#else
using Space = pfc::HIPSpace;
constexpr const char *kForcePacked = "OPENPFC_HIP_FORCE_PACKED_HALO";
constexpr const char *kSubarray = "OPENPFC_HIP_USE_SUBARRAY_HALO";
bool device_available() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}
#endif

enum class Transport { Default, Packed, Subarray };

/// Selects a transport for exchangers constructed in this scope.
class TransportScope {
public:
  explicit TransportScope(Transport t) {
    ::unsetenv(kForcePacked);
    ::unsetenv(kSubarray);
    if (t == Transport::Packed) {
      ::setenv(kForcePacked, "1", 1);
    } else if (t == Transport::Subarray) {
      ::setenv(kSubarray, "1", 1);
    }
  }
  ~TransportScope() {
    ::unsetenv(kForcePacked);
    ::unsetenv(kSubarray);
  }
  TransportScope(const TransportScope &) = delete;
  TransportScope &operator=(const TransportScope &) = delete;
};

int wrap(int g, int n) { return ((g % n) + n) % n; }

/// Distinct per field and cell. Floats stay exact; labels exceed 2^24 and the
/// global origin of field 0 carries the largest label.
template <typename T> T cell_value(int field, int gx, int gy, int gz) {
  if constexpr (std::is_same_v<T, std::uint32_t>) {
    if (field == 0 && gx == 0 && gy == 0 && gz == 0) {
      return 0xFFFFFFFFu;
    }
    const auto lin = static_cast<std::uint32_t>(gx + 16 * gy + 256 * gz);
    return 0x01000001u + 2654435761u * (lin + 1u) +
           static_cast<std::uint32_t>(field);
  } else {
    return static_cast<T>(gx + 16 * gy + 256 * gz) + static_cast<T>(0.25 * field);
  }
}

template <typename T>
pfc::data::Field<T, Space> make_field(const pfc::decomposition::Decomposition &d,
                                      int rank, int hw, int field) {
  pfc::data::Field<T, Space> u(pfc::decomposition::domain(d),
                               pfc::decomposition::local_box(d, rank), hw);
  u.with_host_view([&](T *data, std::size_t) {
    const auto n = u.size3();
    for (int k = -hw; k < n[2] + hw; ++k)
      for (int j = -hw; j < n[1] + hw; ++j)
        for (int i = -hw; i < n[0] + hw; ++i) {
          const bool owned =
              i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
          const auto g = u.global(i, j, k);
          data[u.idx(i, j, k)] =
              owned ? cell_value<T>(field, g[0], g[1], g[2]) : T{};
        }
  });
  return u;
}

/// Faces checks the cells outside the owned box along exactly one axis; Full
/// checks every padded cell.
template <typename T>
bool halo_matches(pfc::data::Field<T, Space> &u, int field, bool full) {
  bool ok = true;
  u.with_host_view([&](T *data, std::size_t) {
    const auto n = u.size3();
    const auto gsz = u.global_size();
    const int hw = u.storage_halo();
    for (int k = -hw; k < n[2] + hw; ++k)
      for (int j = -hw; j < n[1] + hw; ++j)
        for (int i = -hw; i < n[0] + hw; ++i) {
          const int outside =
              (i < 0 || i >= n[0]) + (j < 0 || j >= n[1]) + (k < 0 || k >= n[2]);
          if (!full && outside > 1) {
            continue;
          }
          const auto g = u.global(i, j, k);
          ok &= data[u.idx(i, j, k)] == cell_value<T>(field, wrap(g[0], gsz[0]),
                                                      wrap(g[1], gsz[1]),
                                                      wrap(g[2], gsz[2]));
        }
  });
  return ok;
}

template <typename T>
void check_exchange(const pfc::decomposition::Decomposition &decomp, int rank,
                    int hw) {
  for (const bool full : {false, true}) {
    for (const auto transport :
         {Transport::Default, Transport::Packed, Transport::Subarray}) {
      CAPTURE(full, static_cast<int>(transport), hw);
      TransportScope scope(transport);
      auto u = make_field<T>(decomp, rank, hw, 0);
      auto v = make_field<T>(decomp, rank, hw, 1);
      pfc::comm::HaloExchangeOptions opt;
      opt.connectivity = full ? pfc::comm::HaloConnectivity::Full
                              : pfc::comm::HaloConnectivity::Faces;
      pfc::comm::HaloExchange<Space, T> halo({&u, &v}, decomp, rank, MPI_COMM_WORLD,
                                             opt);
      halo.exchange();
      REQUIRE(halo_matches(u, 0, full));
      REQUIRE(halo_matches(v, 1, full));
    }
  }
}

template <typename T> void check_current_communicator() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 && size != 2 && size != 4) {
    return;
  }
  if (!device_available()) {
    SKIP("No device available on this host");
  }
  auto domain = pfc::domain::create({8, 6, 4});
  const auto decomp = size == 1   ? pfc::decomposition::create(domain, 1)
                      : size == 2 ? pfc::decomposition::create(domain, {2, 1, 1})
                                  : pfc::decomposition::create(domain, {2, 2, 1});
  check_exchange<T>(decomp, rank, 1);
  check_exchange<T>(decomp, rank, 2);
}

} // namespace

TEST_CASE("Device HaloExchange transports float fields",
          "[MPI][gpu][padded_halo][full_halo][element_types]") {
  check_current_communicator<float>();
}

TEST_CASE("Device HaloExchange transports exact uint32 labels",
          "[MPI][gpu][padded_halo][full_halo][element_types]") {
  check_current_communicator<std::uint32_t>();
}

TEST_CASE("Device HaloExchange double path is unchanged by element types",
          "[MPI][gpu][padded_halo][full_halo][element_types]") {
  check_current_communicator<double>();
}

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
