// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_exchange_wait_all.cpp
 * @brief Contract: exchange::wait_all nullifies active caller MPI_Request slots.
 */

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include <openpfc/kernel/decomposition/exchange.hpp>

TEST_CASE("exchange::wait_all nullifies active caller MPI_Request slots",
          "[exchange][wait_all]") {
  int rank = 0;
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

  SECTION("active Isend/Irecv slots become MPI_REQUEST_NULL") {
    int send_buf = 7;
    int recv_buf = 0;
    MPI_Request reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    REQUIRE(MPI_Irecv(&recv_buf, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &reqs[0]) ==
            MPI_SUCCESS);
    REQUIRE(MPI_Isend(&send_buf, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &reqs[1]) ==
            MPI_SUCCESS);

    REQUIRE(reqs[0] != MPI_REQUEST_NULL);
    REQUIRE(reqs[1] != MPI_REQUEST_NULL);

    pfc::exchange::wait_all(reqs, 3);

    REQUIRE(reqs[0] == MPI_REQUEST_NULL);
    REQUIRE(reqs[1] == MPI_REQUEST_NULL);
    REQUIRE(reqs[2] == MPI_REQUEST_NULL);
    REQUIRE(recv_buf == 7);
  }

  SECTION("all-null array is tolerated") {
    MPI_Request reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    REQUIRE_NOTHROW(pfc::exchange::wait_all(reqs, 3));
    REQUIRE(reqs[0] == MPI_REQUEST_NULL);
    REQUIRE(reqs[1] == MPI_REQUEST_NULL);
    REQUIRE(reqs[2] == MPI_REQUEST_NULL);
  }
}
