// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "11_write_results.hpp"
#include <cstdarg>
#include <openpfc/frontend/ui/ui.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/openpfc.hpp>
#include <random>

using namespace pfc;

class CahnHilliard : public Model {
private:
  std::vector<double> opL, opN,
      c; // Define linear operator opL and unknown (real) psi
  std::vector<std::complex<double>> c_F, c_NF; // Define (complex) psi
  double gamma = 1.0e-2;                       // Surface tension
  double D = 1.0;                              // Diffusion coefficient

public:
  using Model::Model; // Inherit the default constructor of base class

  void initialize(double dt) override {
    auto &fft = pfc::get_fft(*this);

    // Allocate space for the main variable and it's fourier transform
    c.resize(fft.size_inbox());
    c_F.resize(fft.size_outbox());
    c_NF.resize(fft.size_outbox());
    opL.resize(fft.size_outbox());
    opN.resize(fft.size_outbox());
    pfc::add_real_field(*this, "concentration", c);

    // prepare operators
    const auto &world = pfc::get_world(*this);
    const auto &domain = pfc::world::get_coordinate_system(world);
    pfc::fft::kspace::for_each_kpoint(
        get_outbox(fft), domain,
        [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
          const double kLap = -(ki * ki + kj * kj + kk * kk);
          const double L = kLap * (-D - D * gamma * kLap);
          opL[idx] = std::exp(L * dt);
          opN[idx] = (L != 0.0) ? (opL[idx] - 1.0) / L * kLap : 0.0;
        });
  }

  void step(double) override {
    auto &fft = pfc::get_fft(*this);
    fft.forward(c, c_F);
    for (auto &elem : c) elem = D * elem * elem * elem;
    fft.forward(c, c_NF);
    for (size_t i = 0; i < c_F.size(); i++)
      c_F[i] = opL[i] * c_F[i] + opN[i] * c_NF[i];
    fft.backward(c_F, c);
  }
};

/**
 * @brief sprintf function for std::string
 *
 * @param fmt
 * @param ...
 * @return std::string
 */
std::string sprintf(const char *fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return std::string(buf);
}

/**
 * @brief Main function
 *
 * @return int
 */
int main(int argc, char **argv) {
  MPI_Worker worker(argc, argv);

  int Lx = 512;
  int Ly = 512;
  int Lz = 1;
  double dx = 20.0 / Lx;
  double dy = 20.0 / Ly;
  double dz = 1.0;
  double x0 = 0.0;
  double y0 = 0.0;
  double z0 = 0.0;

  // Construct domain, decomposition, fft and model
  // Using strong types for type-safe Domain construction
  auto domain =
      ::pfc::domain::create(GridSize{{Lx, Ly, Lz}}, PhysicalOrigin{{x0, y0, z0}},
                            GridSpacing{{dx, dy, dz}});
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);
  const auto size = pfc::domain::get_size(domain);
  CahnHilliard model(
      fft, World({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1}, domain));

  // Define time
  double t = 0.0;
  double t_stop = 1.0;
  double dt = 1.0e-3;
  int n = 0; // increment counter

  // Initialize the model before starting time stepping
  model.initialize(dt);

  // get the concentration field and fill it with random numbers
  std::vector<double> &field = model.get_real_field("concentration");
  std::mt19937_64 rng;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &elem : field) elem = dist(rng);

  // initialize VtkWriter
  VtkWriter<double> writer;
  int file_count = 0;
  // set uri as format cahn_hilliard_%04i.vti, where %04i is replaced by
  // file_count
  writer.set_uri(sprintf("cahn_hilliard_%04i.vti", file_count));
  writer.set_field_name("concentration");
  writer.set_domain(pfc::domain::get_size(domain), get_inbox(fft).size,
                    get_inbox(fft).low);
  writer.set_origin(pfc::domain::get_origin(domain));
  writer.set_spacing(pfc::domain::get_spacing(domain));
  writer.initialize();
  writer.write(field);

  // Initialize high-precision clock
  auto t_start = std::chrono::high_resolution_clock::now();
  // Loop until we are in t_stop
  while (t <= t_stop) {
    model.step(dt);
    if (n % 10 == 0) {
      if (worker.get_rank() == 0) std::cout << "t = " << t << std::endl;
      writer.set_uri(sprintf("cahn_hilliard_%04i.vti", file_count));
      writer.write(field);
      file_count++;
    }
    t += dt;
    n += 1;
  }

  // Stop the clock
  auto t_end = std::chrono::high_resolution_clock::now();
  // Compute the time difference
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  // Print the time difference
  if (worker.get_rank() == 0)
    std::cout << "Solution time: " << duration << " ms" << std::endl;

  return 0;
}
