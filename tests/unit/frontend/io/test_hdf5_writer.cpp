// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#ifdef OPENPFC_HAS_HDF5
#include <hdf5.h>
#include <openpfc/frontend/io/hdf5_writer.hpp>
#endif
#include <openpfc/frontend/ui/results_writer_catalog.hpp>

TEST_CASE("builtin catalog hdf5 key matches OpenPFC_ENABLE_HDF5", "[ui][hdf5][io]") {
  auto cat = pfc::ui::make_builtin_results_writer_catalog();
#ifdef OPENPFC_HAS_HDF5
  REQUIRE(cat.has_type("hdf5"));
#else
  REQUIRE_FALSE(cat.has_type("hdf5"));
#endif
}

#ifdef OPENPFC_HAS_HDF5

TEST_CASE("HDF5Writer writes field dataset and XDMF sidecar", "[hdf5][io]") {
  const std::array<int, 3> global{4, 2, 2};
  const std::array<int, 3> local{4, 2, 2};
  const std::array<int, 3> offset{0, 0, 0};
  std::vector<double> data(4 * 2 * 2);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<double>(i) + 0.25;
  }

  const auto dir = std::filesystem::temp_directory_path() / "openpfc_hdf5_writer";
  std::filesystem::create_directories(dir);
  const std::string pattern = (dir / "field_%d.h5").string();
  const std::string h5 = (dir / "field_0.h5").string();
  const std::string xdmf = (dir / "field_0.xdmf").string();

  {
    pfc::HDF5Writer writer(pattern, MPI_COMM_SELF);
    writer.set_domain(global, local, offset);
    REQUIRE_NOTHROW(writer.write(0, data));
  }

  REQUIRE(std::filesystem::exists(h5));
  REQUIRE(std::filesystem::exists(xdmf));

  hid_t file = H5Fopen(h5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(file >= 0);
  hid_t dset = H5Dopen2(file, "field", H5P_DEFAULT);
  REQUIRE(dset >= 0);
  std::vector<double> back(data.size(), 0.0);
  REQUIRE(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  back.data()) >= 0);
  H5Dclose(dset);
  H5Fclose(file);

  REQUIRE(back == data);

  std::vector<std::complex<double>> z(data.size());
  pfc::HDF5Writer writer2(pattern, MPI_COMM_SELF);
  writer2.set_domain(global, local, offset);
  REQUIRE_THROWS_AS(writer2.write(0, z), std::invalid_argument);

  std::filesystem::remove(h5);
  std::filesystem::remove(xdmf);
}

TEST_CASE("HDF5Writer parallel x-split round-trips /field", "[hdf5][io][MPI]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 2) {
    return;
  }

  const std::array<int, 3> global{8, 2, 2};
  const std::array<int, 3> local{4, 2, 2};
  const std::array<int, 3> offset{rank * 4, 0, 0};
  std::vector<double> data(4 * 2 * 2);
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 4; ++i) {
        const int gi = offset[0] + i;
        data[static_cast<std::size_t>(i + j * 4 + k * 8)] =
            static_cast<double>(gi + 10 * j + 100 * k);
      }
    }
  }

  const auto dir =
      std::filesystem::temp_directory_path() / "openpfc_hdf5_writer_mpi";
  if (rank == 0) {
    std::filesystem::create_directories(dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const std::string pattern = (dir / "field_%d.h5").string();
  const std::string h5 = (dir / "field_0.h5").string();
  const std::string xdmf = (dir / "field_0.xdmf").string();

  {
    pfc::HDF5Writer writer(pattern, MPI_COMM_WORLD);
    writer.set_domain(global, local, offset);
    REQUIRE_NOTHROW(writer.write(0, data));
  }

  if (rank == 0) {
    REQUIRE(std::filesystem::exists(h5));
    REQUIRE(std::filesystem::exists(xdmf));
    hid_t file = H5Fopen(h5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    hid_t dset = H5Dopen2(file, "field", H5P_DEFAULT);
    REQUIRE(dset >= 0);
    std::vector<double> back(8 * 2 * 2, -1.0);
    REQUIRE(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    back.data()) >= 0);
    H5Dclose(dset);
    H5Fclose(file);
    for (int k = 0; k < 2; ++k) {
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 8; ++i) {
          const double expect = static_cast<double>(i + 10 * j + 100 * k);
          REQUIRE(back[static_cast<std::size_t>(i + j * 8 + k * 16)] == expect);
        }
      }
    }
    std::filesystem::remove(h5);
    std::filesystem::remove(xdmf);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

#endif
