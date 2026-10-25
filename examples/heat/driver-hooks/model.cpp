// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 16);
    const double dt = input.value("dt", .07), end = input.value("end", .2);
    const double saveat = input.value("saveat", .1);
    if (n < 8 || n > 64 || !std::isfinite(dt) || dt <= 0 || !std::isfinite(end) ||
        end <= 0 || !std::isfinite(saveat) || saveat <= 0 || end / dt > 10000)
      throw std::invalid_argument("Use n in [8,64] and bounded positive times");
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, 4, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{2 * std::numbers::pi / n, 1, 1}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    auto &u = stack.u();
    auto &fft = stack.fft();
    u.apply([](double x, double, double) { return 1 + .2 * std::cos(x); });
    std::vector<std::complex<double>> modes(fft.size_outbox());
    std::vector<double> k2(modes.size());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          k2[i] = kx * kx + ky * ky + kz * kz;
        });
    std::filesystem::create_directories("results");
    std::ofstream events("results/events_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    std::ofstream fields("results/field_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    events.exceptions(std::ios::badbit | std::ios::failbit);
    fields.exceptions(std::ios::badbit | std::ios::failbit);
    events << std::setprecision(17) << "hook,step,time,interval,amplitude\n";
    fields << std::setprecision(17) << "x,y,z,u\n";
    pfc::Time time({0, end, dt}, saveat);
    double amplitude = .2;
    const auto event = [&](const char *hook, double interval = 0) {
      events << hook << ',' << time.get_increment() << ',' << time.get_current()
             << ',' << interval << ',' << amplitude << '\n';
    };
    // start snippet driver
    pfc::sim::run(
        time,
        [&](double, double interval) {
          fft.forward(u.vec(), modes);
          for (std::size_t i = 0; i < modes.size(); ++i)
            modes[i] /= 1 + interval * k2[i];
          fft.backward(modes, u.vec());
          amplitude /= 1 + interval;
          event("step", interval);
        },
        [&](const auto &) { event("start"); }, [&](const auto &) { event("apply"); },
        [&](const auto &) { event("save"); });
    // end snippet driver
    u.for_each_owned([&](double x, double y, double z, double value) {
      fields << x << ',' << y << ',' << z << ',' << value << '\n';
    });
    if (comm.rank() == 0) std::cout << "Final time " << time.get_current() << '\n';
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
