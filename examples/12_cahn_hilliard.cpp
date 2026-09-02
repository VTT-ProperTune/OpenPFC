// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "11_write_results.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using namespace pfc;

std::string sprintf(const char *fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return std::string(buf);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int Lx = 512;
  constexpr int Ly = 512;
  constexpr int Lz = 1;
  const double dx = 20.0 / Lx;
  const double dy = 20.0 / Ly;
  const double dz = 1.0;
  auto domain =
      domain::create(GridSize{{Lx, Ly, Lz}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                     GridSpacing{{dx, dy, dz}});
  sim::stacks::SpectralCPUStack stack(std::move(domain), rank, nproc);
  auto &fft = stack.fft();
  auto &c = stack.u();

  constexpr double gamma = 1.0e-2;
  constexpr double D = 1.0;
  constexpr double t1 = 1.0;
  constexpr double dt = 1.0e-3;
  std::vector<double> opL(fft.size_outbox());
  std::vector<double> opN(fft.size_outbox());
  std::vector<std::complex<double>> c_F(fft.size_outbox());
  std::vector<std::complex<double>> c_NF(fft.size_outbox());
  pfc::fft::kspace::for_each_kpoint(
      get_outbox(fft), c.domain(),
      [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
        const double kLap = -(ki * ki + kj * kj + kk * kk);
        const double L = kLap * (-D - D * gamma * kLap);
        opL[idx] = std::exp(L * dt);
        opN[idx] = (L != 0.0) ? (opL[idx] - 1.0) / L * kLap : 0.0;
      });

  std::mt19937_64 rng;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &elem : c.vec()) elem = dist(rng);

  VtkWriter<double> writer;
  int file_count = 0;
  writer.set_uri(sprintf("cahn_hilliard_%04i.vti", file_count));
  writer.set_field_name("concentration");
  writer.set_domain(pfc::domain::get_size(c.domain()), get_inbox(fft).size,
                    get_inbox(fft).low);
  writer.set_origin(pfc::domain::get_origin(c.domain()));
  writer.set_spacing(pfc::domain::get_spacing(c.domain()));
  writer.initialize();
  writer.write(c.vec());

  Time time({0.0, t1, dt}, 10.0 * dt);
  auto t_start = std::chrono::high_resolution_clock::now();
  pfc::sim::run(
      time,
      [&](double) {
        fft.forward(c.vec(), c_F);
        for (auto &elem : c.vec()) elem = D * elem * elem * elem;
        fft.forward(c.vec(), c_NF);
        for (size_t i = 0; i < c_F.size(); i++)
          c_F[i] = opL[i] * c_F[i] + opN[i] * c_NF[i];
        fft.backward(c_F, c.vec());
      },
      {}, {},
      [&](const Time &clock) {
        if (time::increment(clock) == 0) return;
        if (rank == 0) std::cout << "t = " << time::current(clock) << std::endl;
        writer.set_uri(sprintf("cahn_hilliard_%04i.vti", file_count));
        writer.write(c.vec());
        file_count++;
      });
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  if (rank == 0) std::cout << "Solution time: " << duration << " ms" << std::endl;

  MPI_Finalize();
  return 0;
}
