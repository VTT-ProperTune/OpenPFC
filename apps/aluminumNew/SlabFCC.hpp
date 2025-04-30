// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_INITIAL_CONDITIONS_SLABFCC_HPP
#define PFC_INITIAL_CONDITIONS_SLABFCC_HPP

#include "SeedFCC.hpp"
#include <array>
#include <iostream>
#include <openpfc/openpfc.hpp>
#include <openpfc/ui.hpp>
#include <random>

using json = nlohmann::json;
using namespace pfc;

/**
 * @brief SlabFCC is a FieldModifier that seeds the model with a slab of FCC
 * seeds.
 *
 */
class SlabFCC : public FieldModifier {
private:
  int m_Ny = 2;
  int m_Nz = 1;
  double m_X0;
  double m_amplitude;
  double m_fluctuation;
  double m_rho;
  double m_rseed;
  bool randomized = true;
  std::vector<std::array<double, 3>> m_orientations;

public:
  SlabFCC() = default;

  SlabFCC(int Ny, int Nz, double X0, double amplitude, double fluctuation,
          double rho, double rseed,
          std::vector<std::array<double, 3>> orientations = {})
      : m_Ny(Ny), m_Nz(Nz), m_X0(X0), m_amplitude(amplitude),
        m_fluctuation(fluctuation), m_rho(rho), m_rseed(rseed),
        m_orientations(orientations) {}

  // getters
  int get_Ny() const { return m_Ny; }
  int get_Nz() const { return m_Nz; }
  double get_X0() const { return m_X0; }
  double get_amplitude() const { return m_amplitude; }
  double get_fluctuation() const { return m_fluctuation; }
  double get_rho() const { return m_rho; }
  double get_rseed() const { return m_rseed; }
  std::vector<std::array<double, 3>> get_orientations() const {
    return m_orientations;
  }

  // setters
  void set_Ny(int Ny) { m_Ny = Ny; }
  void set_Nz(int Nz) { m_Nz = Nz; }
  void set_X0(double X0) { m_X0 = X0; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  void set_fluctuation(double fluctuation) { m_fluctuation = fluctuation; }
  void set_rho(double rho) { m_rho = rho; }
  void set_rseed(double rseed) { m_rseed = rseed; }
  void set_orientations(std::vector<std::array<double, 3>> orientations) {
    m_orientations = orientations;
  }

  /**
   * @brief Apply the initial condition to the model.
   *
   * @param m is the model to apply the initial condition to.
   */
  void apply(Model &m, double) override {
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_real_field("psi");
    Vec3<int> low = decomp.get_inbox().low;
    Vec3<int> high = decomp.get_inbox().high;

    // auto Lx = w.Lx;
    auto Ly = w.get_size()[1];
    auto Lz = w.get_size()[2];
    auto dx = w.get_spacing()[0];
    auto dy = w.get_spacing()[1];
    auto dz = w.get_spacing()[2];
    auto x0 = w.get_origin()[0];
    auto y0 = w.get_origin()[1];
    auto z0 = w.get_origin()[2];

    std::vector<SeedFCC> seeds;

    int Ny = m_Ny;
    int Nz = m_Nz;
    double rseed = m_rseed;

    double rho = get_rho();
    double amplitude = get_amplitude();
    double fluctuation = get_fluctuation();

    double X0 = m_X0;
    double Dy = dy * Ly / Ny;
    double Y0 = Dy / 2.0;
    double Dz = dz * Lz / Nz;
    double Z0 = Dz / 2.0;
    int nseeds = Ny * Nz;

    double radius = 1.;

    std::vector<std::array<double, 3>> orientations = m_orientations;

    randomized = (orientations.empty());

    std::mt19937_64 re(rseed);
    std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
    std::uniform_real_distribution<double> rr(0.0, 8.0 * atan(1.0));
    std::uniform_real_distribution<double> ra(-fluctuation, fluctuation);

    if (randomized) {

      std::cout << "Generating " << nseeds << " random seeds up to distance "
                << X0 << "\n";

      for (int j = 0; j < Ny; j++) {
        for (int k = 0; k < Nz; k++) {
          const std::array<double, 3> location = {
              X0 + rt(re), Y0 + Dy * j + rt(re), Z0 + Dz * k + rt(re)};
          const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
          const SeedFCC seed(location, orientation, radius, rho, amplitude);
          seeds.push_back(seed);
        }
      }

    } else {
      std::cout << "Generating " << nseeds
                << " seeds from orientation list up to distance " << X0 << "\n";

      for (int j = 0; j < Ny; j++) {
        for (int k = 0; k < Nz; k++) {
          const std::array<double, 3> location = {
              X0 + rt(re), Y0 + Dy * j + rt(re), Z0 + Dz * k + rt(re)};
          const std::array<double, 3> orientation = orientations[j * Nz + k];
          const SeedFCC seed(location, orientation, radius, rho, amplitude);
          seeds.push_back(seed);
        }
      }
    }

    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const std::array<double, 3> X = {x, y, z};
          if (x < (X0 + ra(re))) {
            const int ns = floor(y / Dy) * Nz + floor(z / Dz);
            field[idx] = seeds[ns].get_value(X);
          }
          idx += 1;
        }
      }
    }
  }
}; // SlabFCC

/**
 * @brief Configure SlabFCC object from a json file.
 *
 * @param params json object containing the parameters.
 * @param ic SlabFCC object to configure.
 * @return void
 *
 * Example json configuration:
 *
 * { "type": "slab_fcc", "X0": 130.0, "Ny": 2, "Nz": 1,
 * "amplitude": 0.2, "fluctuation": 10.0, "rho": -0.036, "rseed": 42 }
 *
 * The "type" field is required and must be "seed_grid_fcc". The "rseed" field
 * is optional and defaults to 0. All other fields are required. The "Ny" and
 * "Nz" fields are the number of seeds in the y and z directions, respectively.
 * The "X0" field is the x-coordinate of the center of the first seed. The
 * "radius" field is the radius of the seeds. The "amplitude" field is the
 * amplitude of the seed. The "rho" field is the background density. The "rseed"
 * field is the random seed.
 *
 */
void from_json(const json &params, SlabFCC &ic) {
  std::cout << "Parsing SlabFCC from json" << std::endl;

  // check for required fields
  if (!params.contains("amplitude") || !params["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Reading SlabFCC failed: missing or invalid 'amplitude' field.");
  }
  if (!params.contains("fluctuation") || !params["fluctuation"].is_number()) {
    throw std::invalid_argument(
        "Reading SlabFCC failed: missing or invalid 'amplitude' field.");
  }
  if (!params.contains("rho") || !params["rho"].is_number()) {
    throw std::invalid_argument(
        "Reading SlabFCC failed: missing or invalid 'rho' field.");
  }
  if (!params.contains("Ny") || !params["Ny"].is_number()) {
    throw std::invalid_argument(
        "Reading SlabFCC failed: missing or invalid 'Ny' field.");
  }
  if (!params.contains("Nz") || !params["Nz"].is_number()) {
    throw std::invalid_argument(
        "Reading SlabFCC failed: missing or invalid 'Nz' field.");
  }
  if (!params.contains("X0") || !params["X0"].is_number()) {
    throw std::invalid_argument(
        "Reading SlabFCC failed: missing or invalid 'X0' field.");
  }
  if (!params.contains("rseed") || !params["rseed"].is_number()) {
    std::cout << "No valid random seed detected, using default value 0."
              << std::endl;
  }
  if (!params.contains("orientations") || !params["orientations"].is_array()) {
    std::cout << "No valid orientation vector detected, randomizing."
              << std::endl;
  }
  int Nz = params["Nz"];
  int Ny = params["Ny"];

  std::vector<std::array<double, 3>> m_orientations;
  auto orientations = params.value("orientations", m_orientations);
  // auto orientations = params["orientations"];
  if (!orientations.empty() && orientations.size() != (Nz * Ny)) {
    throw std::invalid_argument(
        "Orientation vector and seed grid sizes do not match.");
  }

  ic.set_Ny(params["Ny"]);
  ic.set_Nz(params["Nz"]);
  ic.set_X0(params["X0"]);
  ic.set_amplitude(params["amplitude"]);
  ic.set_fluctuation(params["fluctuation"]);
  ic.set_rho(params["rho"]);
  if (params.contains("rseed") && params["rseed"].is_number())
    ic.set_rseed(params["rseed"]);
  // if (params.contains("orientations") && params["orientations"].is_array())
  // ic.set_orientations(params["orientations"]);
  ic.set_orientations(orientations);
}

#endif // PFC_INITIAL_CONDITIONS_SLABFCC_HPP
