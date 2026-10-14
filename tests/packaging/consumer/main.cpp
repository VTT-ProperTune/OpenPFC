// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Smallest possible use of the installed OpenPFC public API: construct a Domain,
// run a two-field SimulationLifecycle, and query the result. If this configures,
// links, and runs, find_package(OpenPFC) and the exported targets are wired.
#include <openpfc/frontend/io/snapshot_series.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/power_spectrum.hpp>
#include <openpfc/kernel/field/face_flux.hpp>
#include <openpfc/kernel/field/fourier_series.hpp>
#include <openpfc/kernel/field/indexed_noise.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/simulation_lifecycle.hpp>
#include <openpfc/kernel/simulation/spectral_flux.hpp>
#include <openpfc/solvers/microelasticity/microelasticity.hpp>

#include <mpi.h>

#include <cstdio>
#include <memory>
#include <type_traits>

int run_spectral_state_consumer(); // spectral_state.cpp
int run_coupled_etd2_consumer();   // coupled_etd2.cpp

int main() {
  static_assert(std::is_trivially_copyable_v<pfc::sim::AsStored>);
  static_assert(static_cast<int>(pfc::solvers::MicroelasticityScheme::EyreMilton) !=
                static_cast<int>(pfc::solvers::MicroelasticityScheme::Basic));
  static_assert(pfc::field::fd::average_face(pfc::field::fd::FaceAverage::Harmonic,
                                             0.0, 1.0) == 0.0);
  static_assert(pfc::fft::r2c_multiplicity(0, 64) == 1.0);
  static_assert(pfc::fft::r2c_multiplicity(1, 64) == 2.0);
  static_assert(pfc::fft::r2c_multiplicity(32, 64) == 1.0);
  static_assert(pfc::field::indexed_noise_sample(1, 2, 3, 4, 8, 8) <= 65535);
  static_assert(pfc::field::indexed_noise_sample(1, 0, 0, 0, 4, 4) !=
                pfc::field::indexed_noise_sample(2, 0, 0, 0, 4, 4));
  static_assert(static_cast<int>(pfc::io::SnapshotFormat::Binary) !=
                static_cast<int>(pfc::io::SnapshotFormat::Vtk));
  constexpr pfc::field::FourierMode mode{{1, 0, 0}, 1.0, 0.0};
  static_assert(mode.index[0] == 1);
  pfc::fft::RadialSpectrum spectrum;
  spectrum.wavenumber = {1.0};
  spectrum.mean_power = {1.0};
  if (pfc::fft::power_near(spectrum, 1.0) != 1.0) {
    return 1;
  }
  pfc::solvers::MicroelasticityParams elastic;
  elastic.stiffness_at_one = pfc::solvers::Stiffness::isotropic(1.0, 0.3);
  elastic.stiffness_at_zero = elastic.stiffness_at_one;
  elastic.relative_tolerance = 1.0e-6;
  if (elastic.max_iterations < 1) {
    return 1;
  }
  pfc::Time clock({0.0, 0.2, 0.1}, 0.2);
  int saves = 0;
  int rejects = 1;
  pfc::sim::run_attempts(
      clock,
      [&](pfc::Time &) {
        if (rejects > 0) {
          --rejects;
          return pfc::sim::StepDecision{false};
        }
        return pfc::sim::StepDecision{true};
      },
      pfc::sim::NoopHook{}, pfc::sim::NoopHook{},
      [&](const pfc::Time &) { ++saves; });
  if (saves != 2 || clock.get_increment() != 2) {
    return 1;
  }
  int mpi_ready = 0;
  MPI_Initialized(&mpi_ready);
  if (mpi_ready == 0) {
    MPI_Init(nullptr, nullptr);
  }
  auto domain = pfc::domain::create({8, 8, 8});
  const auto size = pfc::domain::get_size(domain);
  const auto box =
      pfc::Box3i::from_bounds({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
  pfc::data::Field<double> density(domain, box, 0);
  pfc::data::Field<double> solute(domain, box, 0);
  pfc::sim::SimulationLifecycle life(
      pfc::sim::SimulationLifecycle::schedule(0.0, 0.4, 0.1, 0.4), MPI_COMM_WORLD);
  life.bind_field("density", density);
  life.bind_field("solute", solute);
  life.add_initial_condition("density", std::make_unique<pfc::Constant>(1.0));
  life.add_initial_condition("solute", std::make_unique<pfc::Constant>(2.0));
  int life_saves = 0;
  int accepted_steps = 0;
  life.set_save_observer([&](const pfc::Time &) { ++life_saves; });
  life.set_accepted_step_hook([&](const pfc::Time &) { ++accepted_steps; });
  life.run([](double, double) {});
  const bool fields_ok = density(0, 0, 0) == 1.0 && solute(0, 0, 0) == 2.0;
  const bool domain_ok = size[0] == 8 && size[1] == 8 && size[2] == 8;
  const bool cadence_ok = life_saves == 2 && accepted_steps == 4;
  const bool spectral_ok = run_spectral_state_consumer() == 0;
  const bool coupled_ok = run_coupled_etd2_consumer() == 0;
  if (fields_ok && domain_ok && cadence_ok && spectral_ok && coupled_ok) {
    std::printf("OpenPFC consumer OK: domain %dx%dx%d\n", size[0], size[1], size[2]);
  }
  int mpi_done = 0;
  MPI_Finalized(&mpi_done);
  if (mpi_ready == 0 && mpi_done == 0) {
    MPI_Finalize();
  }
  return (fields_ok && domain_ok && cadence_ok && spectral_ok && coupled_ok) ? 0 : 1;
}
