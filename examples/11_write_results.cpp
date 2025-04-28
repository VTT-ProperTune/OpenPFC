// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "11_write_results.hpp"
#include <complex>
#include <mpi.h>
#include <openpfc/openpfc.hpp>
#include <string>
#include <vector>

using namespace pfc;

// In this example, we will write the results of a simulation to a file.
int main(int argc, char **argv) {
  MPI_Worker worker(argc, argv);
  World world({4, 3, 2});
  Decomposition decomp(world);
  DiscreteField<double, 3> field(decomp);

  std::vector<double> arr(2 * 3 * 4);
  for (unsigned int i = 0; i < arr.size(); i++) arr[i] = static_cast<double>(i);
  field.set_data(std::move(arr));

  VtkWriter<double> writer;
  writer.set_uri("results.vti");
  writer.set_field_name("density");
  writer.set_domain(world.get_size(), field.get_size(), field.get_offset());
  writer.set_origin(world.get_origin());
  writer.set_spacing(world.spacing());
  std::cout << "Writing results to file: " << writer.get_uri() << "\n";
  writer.initialize();
  writer.write(field.get_array().get_data());
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

/*
TEST_CASE("VtkWriter", "[VtkWriter]") {
  World world({8, 2, 2});
  Decomposition decomp(world);
  DiscreteField<double, 3> field(decomp);
  field.apply([](auto x, auto y, auto z) { return x + y + z; });
  VtkWriter<double> writer;
  writer.set_uri("results.vtk");
  writer.set_domain(world.get_size(), field.get_size(), field.get_offset());
  writer.write(field.get_array().get_data());
  std::string expectedOutput = R"EXPECTED(<?xml version="1.0" encoding="utf-8"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <ImageData WholeExtent="0 3 0 2 0 1" Origin="1 1 1" Spacing="1 1 1">
    <Piece Extent="0 3 0 2 0 1">
      <PointData>
        <DataArray type="Float64" Name="density" NumberOfComponents="1" format="appended" offset="0"/>
      </PointData>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
</VTKFile>)EXPECTED";
  REQUIRE(true);
}
*/
