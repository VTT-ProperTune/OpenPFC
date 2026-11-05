// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <burgers/model.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void usage() {
  std::cerr << "usage: burgers --method "
               "spectral|spectral_aliased|fdm2|fdm2_advective|fvm\n"
            << "               --N <int> --nu <float> --dt <float> --time <float>\n"
            << "               [--outdir <directory>]\n";
}

[[nodiscard]] std::string argument(int &index, int argc, char **argv,
                                   std::string_view flag) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(flag) + " needs a value");
  }
  ++index;
  return argv[index];
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string method_text;
    int count = 0;
    double viscosity = -1.0;
    double dt = 0.0;
    double time = 0.0;
    std::string outdir;
    for (int index = 1; index < argc; ++index) {
      const std::string_view flag = argv[index];
      if (flag == "--help") {
        usage();
        return 0;
      }
      if (flag == "--method") {
        method_text = argument(index, argc, argv, flag);
      } else if (flag == "--N") {
        count = std::stoi(argument(index, argc, argv, flag));
      } else if (flag == "--nu") {
        viscosity = std::stod(argument(index, argc, argv, flag));
      } else if (flag == "--dt") {
        dt = std::stod(argument(index, argc, argv, flag));
      } else if (flag == "--time") {
        time = std::stod(argument(index, argc, argv, flag));
      } else if (flag == "--outdir") {
        outdir = argument(index, argc, argv, flag);
      } else {
        throw std::invalid_argument(std::string("unknown argument ") + argv[index]);
      }
    }
    if (method_text.empty() || count <= 0 || viscosity < 0.0 || !(dt > 0.0) ||
        !(time > 0.0)) {
      usage();
      return 2;
    }
    const double steps_real = time / dt;
    const auto steps = static_cast<long long>(std::llround(steps_real));
    if (std::abs(steps_real - static_cast<double>(steps)) > 1.0e-8 ||
        steps > 1000000000LL) {
      throw std::invalid_argument("time/dt must be a positive integer step count");
    }
    burgers::Simulation simulation(burgers::parse_method(method_text), count,
                                   viscosity);
    simulation.set_initial_sine();
    const double safe_step = simulation.stability_limit();
    simulation.advance(dt, static_cast<int>(steps));
    const double peak = burgers::maximum_abs(simulation.state());
    const double inventory = burgers::mass(simulation.spacing(), simulation.state());
    const double cfl_dt = peak > 0.0 ? simulation.spacing() / peak
                                     : std::numeric_limits<double>::infinity();
    std::ostringstream summary;
    summary << std::setprecision(17);
    summary << "method=" << burgers::method_name(simulation.method())
            << " scheme=ssp-rk3"
            << " nu=" << simulation.viscosity() << " N=" << simulation.count()
            << " dx=" << simulation.spacing() << " dt=" << dt << " time=" << time
            << " steps=" << steps << " cfl_dt=" << cfl_dt
            << " stability_limit_initial=" << safe_step
            << " status=ok mass=" << inventory << " energy="
            << burgers::energy(simulation.spacing(), simulation.state())
            << " max_abs=" << peak << " checksum=" << std::hex
            << burgers::checksum(simulation.state()) << std::dec << "\n";
    if (!outdir.empty()) {
      std::filesystem::create_directories(outdir);
      std::ofstream file(std::filesystem::path(outdir) / "u.dat");
      if (!file) throw std::runtime_error("burgers could not open u.dat");
      file << std::setprecision(17);
      file << "# x";
      if (simulation.method() == burgers::Method::fvm) file << "_center";
      file << " u\n";
      for (int i = 0; i < simulation.count(); ++i) {
        const double x = simulation.method() == burgers::Method::fvm
                             ? (static_cast<double>(i) + 0.5) * simulation.spacing()
                             : static_cast<double>(i) * simulation.spacing();
        file << x << " " << simulation.state()[static_cast<std::size_t>(i)] << "\n";
      }
      file.close();
      if (!file) throw std::runtime_error("burgers could not write u.dat");
    }
    // Publish success only after every requested output has been written.
    std::cout << summary.str();
  } catch (const std::exception &error) {
    std::cerr << "burgers: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
