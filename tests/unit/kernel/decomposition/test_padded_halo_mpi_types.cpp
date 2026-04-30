// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>

namespace {

// Linear index into a (nx_pad, ny_pad, nz_pad) padded buffer (x fastest).
inline std::size_t lin(int i, int j, int k, int nxp, int nyp) {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(nxp) +
         static_cast<std::size_t>(k) * static_cast<std::size_t>(nxp) *
             static_cast<std::size_t>(nyp);
}

// Self-loopback: send via direction `d` from `sendbuf`, recv via the
// matching direction (same dir; we test that send/recv subarrays
// describe consistent slabs by sending into a separate `recvbuf` and
// then comparing). We use two distinct buffers (sendbuf, recvbuf) so
// we can confirm what was *read* from sendbuf and where it landed in
// recvbuf, separately from the addressing.
inline void self_sendrecv(MPI_Datatype send_t, double *sendbuf, MPI_Datatype recv_t,
                          double *recvbuf) {
  const int self = 0;
  MPI_Sendrecv(sendbuf, 1, send_t, self, 99, recvbuf, 1, recv_t, self, 99,
               MPI_COMM_SELF, MPI_STATUS_IGNORE);
}

} // namespace

TEST_CASE("create_padded_face_types_6: +X/-X subarrays map the right slabs",
          "[halo][padded_mpi_types]") {
  const int nx = 4, ny = 3, nz = 2;
  const int hw = 1;
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const int nzp = nz + 2 * hw;
  const std::size_t N = static_cast<std::size_t>(nxp) *
                        static_cast<std::size_t>(nyp) *
                        static_cast<std::size_t>(nzp);

  std::vector<double> sendbuf(N, 0.0);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        sendbuf[lin(hw + i, hw + j, hw + k, nxp, nyp)] =
            100.0 + 10.0 * i + j + 0.1 * k;

  auto faces = pfc::halo::create_padded_face_types_6(nx, ny, nz, hw, MPI_DOUBLE);

  std::vector<double> recv(N, -1.0);
  self_sendrecv(faces[0].send_type.get(), sendbuf.data(), faces[0].recv_type.get(),
                recv.data());
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      const std::size_t dst = lin(nx + hw, hw + j, hw + k, nxp, nyp);
      const std::size_t src = lin(hw + (nx - 1), hw + j, hw + k, nxp, nyp);
      REQUIRE(recv[dst] == sendbuf[src]);
    }
  }

  std::fill(recv.begin(), recv.end(), -1.0);
  self_sendrecv(faces[1].send_type.get(), sendbuf.data(), faces[1].recv_type.get(),
                recv.data());
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      const std::size_t dst = lin(0, hw + j, hw + k, nxp, nyp);
      const std::size_t src = lin(hw, hw + j, hw + k, nxp, nyp);
      REQUIRE(recv[dst] == sendbuf[src]);
    }
  }
}

TEST_CASE("create_padded_face_types_6: +Y/-Y subarrays map the right slabs",
          "[halo][padded_mpi_types]") {
  const int nx = 3, ny = 4, nz = 2;
  const int hw = 1;
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const int nzp = nz + 2 * hw;
  const std::size_t N = static_cast<std::size_t>(nxp * nyp * nzp);

  std::vector<double> sendbuf(N, 0.0);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        sendbuf[lin(hw + i, hw + j, hw + k, nxp, nyp)] =
            1000.0 + 100.0 * j + 10.0 * i + k;

  auto faces = pfc::halo::create_padded_face_types_6(nx, ny, nz, hw, MPI_DOUBLE);

  std::vector<double> recv(N, -1.0);
  self_sendrecv(faces[2].send_type.get(), sendbuf.data(), faces[2].recv_type.get(),
                recv.data());
  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx; ++i) {
      const std::size_t dst = lin(hw + i, ny + hw, hw + k, nxp, nyp);
      const std::size_t src = lin(hw + i, hw + (ny - 1), hw + k, nxp, nyp);
      REQUIRE(recv[dst] == sendbuf[src]);
    }
  }

  std::fill(recv.begin(), recv.end(), -1.0);
  self_sendrecv(faces[3].send_type.get(), sendbuf.data(), faces[3].recv_type.get(),
                recv.data());
  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx; ++i) {
      const std::size_t dst = lin(hw + i, 0, hw + k, nxp, nyp);
      const std::size_t src = lin(hw + i, hw, hw + k, nxp, nyp);
      REQUIRE(recv[dst] == sendbuf[src]);
    }
  }
}

TEST_CASE("create_padded_face_types_6: +Z/-Z subarrays map the right slabs",
          "[halo][padded_mpi_types]") {
  const int nx = 2, ny = 3, nz = 4;
  const int hw = 1;
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const int nzp = nz + 2 * hw;
  const std::size_t N = static_cast<std::size_t>(nxp * nyp * nzp);

  std::vector<double> sendbuf(N, 0.0);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        sendbuf[lin(hw + i, hw + j, hw + k, nxp, nyp)] =
            10000.0 + 1000.0 * k + 10.0 * j + i;

  auto faces = pfc::halo::create_padded_face_types_6(nx, ny, nz, hw, MPI_DOUBLE);

  std::vector<double> recv(N, -1.0);
  self_sendrecv(faces[4].send_type.get(), sendbuf.data(), faces[4].recv_type.get(),
                recv.data());
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const std::size_t dst = lin(hw + i, hw + j, nz + hw, nxp, nyp);
      const std::size_t src = lin(hw + i, hw + j, hw + (nz - 1), nxp, nyp);
      REQUIRE(recv[dst] == sendbuf[src]);
    }
  }

  std::fill(recv.begin(), recv.end(), -1.0);
  self_sendrecv(faces[5].send_type.get(), sendbuf.data(), faces[5].recv_type.get(),
                recv.data());
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const std::size_t dst = lin(hw + i, hw + j, 0, nxp, nyp);
      const std::size_t src = lin(hw + i, hw + j, hw, nxp, nyp);
      REQUIRE(recv[dst] == sendbuf[src]);
    }
  }
}

TEST_CASE(
    "create_padded_face_types_6: subarrays size = hw * orth_owned * orth2_owned",
    "[halo][padded_mpi_types]") {
  const int nx = 5, ny = 4, nz = 3;
  const int hw = 2;
  auto faces = pfc::halo::create_padded_face_types_6(nx, ny, nz, hw, MPI_DOUBLE);

  auto count = [](MPI_Datatype dt) {
    int size = 0;
    MPI_Type_size(dt, &size);
    return static_cast<std::size_t>(size) / sizeof(double);
  };

  REQUIRE(count(faces[0].send_type.get()) == static_cast<std::size_t>(hw * ny * nz));
  REQUIRE(count(faces[0].recv_type.get()) == static_cast<std::size_t>(hw * ny * nz));

  REQUIRE(count(faces[2].send_type.get()) == static_cast<std::size_t>(nx * hw * nz));

  REQUIRE(count(faces[4].send_type.get()) == static_cast<std::size_t>(nx * ny * hw));
}
