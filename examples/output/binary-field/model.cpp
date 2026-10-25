// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto domain = output_lesson::domain(argc, argv);
    const auto partition = pfc::decomposition::create(domain, comm.size());
    auto field = pfc::data::field_from_subdomain<double>(partition, comm.rank(), 0);
    field.apply(
        [](double x, double y, double z) { return output_lesson::value(x, y, z); });
    // start snippet binary
    pfc::BinaryWriter writer("results/field_%04d.bin", comm);
    writer.set_domain(pfc::domain::get_size(domain), field.box().size,
                      field.box().low);
    (void)writer.write(5, field.vec());
    pfc::BinaryReader reader(comm);
    reader.set_domain(pfc::domain::get_size(domain), field.box().size,
                      field.box().low);
    std::vector<double> local_copy(field.size());
    (void)reader.read("results/field_0005.bin", local_copy);
    if (local_copy != field.vec())
      throw std::runtime_error("Local binary read changed a value");
    // end snippet binary
    MPI_Barrier(comm);
    if (comm.rank() == 0) {
      // start snippet read
      std::ifstream file("results/field_0005.bin", std::ios::binary);
      const auto size = pfc::domain::get_size(domain);
      std::vector<double> restored(size[0] * size[1] * size[2]);
      file.read(reinterpret_cast<char *>(restored.data()),
                static_cast<std::streamsize>(restored.size() * sizeof(double)));
      if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Unexpected binary file length");
      double error = 0;
      for (int k = 0; k < size[2]; ++k)
        for (int j = 0; j < size[1]; ++j)
          for (int i = 0; i < size[0]; ++i) {
            const auto x = pfc::domain::to_coords(domain, {i, j, k});
            error =
                std::max(error, std::abs(restored[i + size[0] * (j + size[1] * k)] -
                                         output_lesson::value(x[0], x[1], x[2])));
          }
      if (error != 0) throw std::runtime_error("Binary round trip changed a value");
      // end snippet read
      std::cout << "Restored " << restored.size() << " doubles exactly\n";
    }
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
