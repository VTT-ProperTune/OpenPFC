// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
// Parse TOML before optional HIP headers can introduce compiler macros.
#include <openpfc/frontend/ui/settings_loader.hpp>

#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    if (argc < 2) throw std::invalid_argument("Pass input.json or input.toml");
    // start snippet session
    const auto settings = pfc::ui::load_settings_file(argv[1]);
    const auto requested_domain = pfc::ui::from_json<pfc::Domain>(settings);
    const auto requested_time = pfc::ui::from_json<pfc::Time>(settings);
    const auto size = pfc::domain::get_size(requested_domain);
    if (size[0] > 64 || size[1] > 64 || size[2] > 64 ||
        (requested_time.get_t1() - requested_time.get_t0()) /
                requested_time.get_dt() >
            10000)
      throw std::invalid_argument("Use at most 64 samples per axis and 10000 steps");
    using Stack = pfc::sim::stacks::SpectralCPUStack;
    auto session = pfc::ui::make_simulation_session<Stack>(settings, comm.rank(),
                                                           comm.size(), comm);
    auto &u = session.stack().u();
    auto &fft = session.stack().fft();
    const auto &domain = u.domain();
    const double origin = pfc::domain::get_origin(domain)[0];
    const double length = size[0] * pfc::domain::get_spacing(domain)[0];
    u.apply([&](double x, double, double) {
      return 1 + .2 * std::cos(2 * std::numbers::pi * (x - origin) / length);
    });
    // end snippet session
    std::vector<std::complex<double>> modes(fft.size_outbox());
    std::vector<double> k2(modes.size());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          k2[i] = kx * kx + ky * ky + kz * kz;
        });
    // start snippet run
    session.run([&](double, double interval) {
      fft.forward(u.vec(), modes);
      for (std::size_t i = 0; i < modes.size(); ++i)
        modes[i] /= 1 + interval * k2[i];
      fft.backward(modes, u.vec());
    });
    // end snippet run
    std::filesystem::create_directories("results");
    std::ofstream field("results/field_rank_" + std::to_string(comm.rank()) +
                        ".csv");
    if (!field) throw std::runtime_error("Cannot open field CSV");
    field << std::setprecision(17) << "x,y,z,u\n";
    u.for_each_owned([&](double x, double y, double z, double value) {
      field << x << ',' << y << ',' << z << ',' << value << '\n';
    });
    if (!field) throw std::runtime_error("Cannot finish field CSV");
    if (comm.rank() == 0) {
      std::ofstream normalized("results/settings.json");
      normalized << settings.dump(2) << '\n';
      if (!normalized) throw std::runtime_error("Cannot write normalized settings");
      std::cout << session.stack_name() << " final=" << session.time().get_current()
                << '\n';
    }
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
