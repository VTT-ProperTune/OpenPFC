// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
  auto domain = domain::create({4, 3, 2});
  auto decomposition_obj = decomposition::create(domain, 1);
  // DiscreteField<double, 3> field(decomp);
  auto local_box_0 = decomposition::local_box(decomposition_obj, 0);
  auto dimensions = local_box_0.size;
  auto offsets = local_box_0.low;
  auto origin = domain::get_origin(domain);
  auto discretization = domain::get_spacing(domain);
  DiscreteField<double, 3> field(dimensions, offsets, origin, discretization);

  std::vector<double> arr(2 * 3 * 4);
  for (unsigned int i = 0; i < arr.size(); i++) arr[i] = static_cast<double>(i);
  field.set_data(std::move(arr));

  VtkWriter<double> writer;
  writer.set_uri("results.vti");
  writer.set_field_name("density");
  writer.set_domain(domain::get_size(domain), field.get_size(), field.get_offset());
  writer.set_origin(domain::get_origin(domain));
  writer.set_spacing(domain::get_spacing(domain));
  std::cout << "Writing results to file: " << writer.get_uri() << "\n";
  writer.initialize();
  writer.write(field.get_array().get_data());
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

/*
TEST_CASE("VtkWriter", "[VtkWriter]") {
  auto domain = domain::create({8, 2, 2});
  Decomposition decomp(domain);
  DiscreteField<double, 3> field(decomp);
  field.apply([](auto x, auto y, auto z) { return x + y + z; });
  VtkWriter<double> writer;
  writer.set_uri("results.vtk");
  writer.set_domain(domain::get_size(domain), field.get_size(), field.get_offset());
  writer.write(field.get_array().get_data());
  std::string expectedOutput = R"EXPECTED(<?xml version="1.0" encoding="utf-8"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian"
header_type="UInt64"> <ImageData WholeExtent="0 3 0 2 0 1" Origin="1 1 1"
Spacing="1 1 1"> <Piece Extent="0 3 0 2 0 1"> <PointData> <DataArray
type="Float64" Name="density" NumberOfComponents="1" format="appended"
offset="0"/>
      </PointData>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
</VTKFile>)EXPECTED";
  REQUIRE(true);
}
*/
