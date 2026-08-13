// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <random>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/detail/array_format.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/worker.hpp>

using namespace pfc;
using namespace std;

int main(int argc, char *argv[]) {

  // Create MPI session, Domain and Decomposition
  MPI_Worker worker(argc, argv);
  auto domain = domain::create({4, 3, 2});
  auto decomp = decomposition::create(domain, worker.get_num_ranks());

  // Create input field for this rank
  auto input = data::field_from_subdomain_unpadded<double>(decomp, worker.get_rank(), 0);

  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  // Fill input field with random numbers
  input.apply([&](double /*x*/, double /*y*/, double /*z*/) { return dis(gen); });

  auto fft_instance = fft::create(decomp, worker.get_rank(), MPI_COMM_WORLD);

  // Complex FFT outbox as a Field (r2c first axis is floor(Lx/2)+1).
  auto output = data::Field<complex<double>>(domain, fft::get_outbox(fft_instance),
                                             /*halo_width=*/0);

  auto input_size = input.local_size();
  auto output_size = output.local_size();
  std::cout << "input: {" << input_size[0] << ", " << input_size[1] << ", "
            << input_size[2] << "}" << std::endl; // this is {4, 3, 2}
  std::cout << "output: {" << output_size[0] << ", " << output_size[1] << ", "
            << output_size[2] << "}" << std::endl; // this is {3, 3, 2}

  fft_instance.forward(input.vec(), output.vec());

  pfc::detail::show(output.vec(), output.local_size(), output.box().low);

  return 0;
}
