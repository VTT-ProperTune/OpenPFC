// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "11_write_results.hpp"
#include <mpi.h>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/worker.hpp>
#include <vector>

using namespace pfc;

// In this example, we will write the results of a simulation to a file.
int main(int argc, char **argv) {
  MPI_Worker worker(argc, argv);
  auto domain = domain::create({4, 3, 2});
  auto decomposition_obj = decomposition::create(domain, 1);
  
  // Create a field from the decomposition using field_from_subdomain_unpadded
  auto field = pfc::data::field_from_subdomain_unpadded<double>(decomposition_obj, 0);

  // Populate the field with data (using the field's apply method for coordinate-based initialization)
  field.apply([](double x, double y, double z) {
    // Simple linear index based initialization for demonstration
    return static_cast<double>(static_cast<int>(x) + static_cast<int>(y) * 4 + static_cast<int>(z) * 12);
  });

  VtkWriter<double> writer;
  writer.set_uri("results.vti");
  writer.set_field_name("density");
  
  // Use the field's geometry methods
  auto global_size = field.global_size();
  auto local_size = field.local_size();
  auto offset = field.lower_global();
  
  writer.set_domain({global_size[0], global_size[1], global_size[2]}, 
                    {local_size[0], local_size[1], local_size[2]},
                    {offset[0], offset[1], offset[2]});
  writer.set_origin(field.origin());
  writer.set_spacing(field.spacing());
  
  std::cout << "Writing results to file: " << writer.get_uri() << "\n";
  writer.initialize();
  
  // Pack owned field data into a buffer for VTK writer
  std::vector<double> buffer;
  field.for_each_owned([&](int i, int j, int k) {
    buffer.push_back(field(i, j, k));
  });
  
  writer.write(buffer);
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

/*
TEST_CASE("VtkWriter", "[VtkWriter]") {
  auto domain = domain::create({8, 2, 2});
  auto decomp = decomposition::create(domain, 1);
  auto field = pfc::data::field_from_subdomain_unpadded<double>(decomp, 0);
  field.apply([](auto x, auto y, auto z) { return x + y + z; });
  VtkWriter<double> writer;
  writer.set_uri("results.vtk");
  auto global_size = field.global_size();
  auto local_size = field.local_size();
  auto offset = field.lower_global();
  writer.set_domain({global_size[0], global_size[1], global_size[2]}, 
                    {local_size[0], local_size[1], local_size[2]},
                    {offset[0], offset[1], offset[2]});
  std::vector<double> buffer;
  field.for_each_owned([&](int i, int j, int k) {
    buffer.push_back(field(i, j, k));
  });
  writer.write(buffer);
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
