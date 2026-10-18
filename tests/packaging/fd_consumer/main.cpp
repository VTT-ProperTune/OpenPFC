// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <cstdio>
#include <openpfc/kernel/field/reaction_flux.hpp>
#include <openpfc/kernel/field/sbp_diffusion.hpp>

#include <mpi.h>

#include <openpfc/kernel/field/face_current.hpp>
#include <openpfc/kernel/field/face_flux.hpp>
#include <openpfc/kernel/simulation/simulation_lifecycle.hpp>

int main() {
  int ready = 0;
  MPI_Initialized(&ready);
  if (ready == 0) {
    MPI_Init(nullptr, nullptr);
  }
  using pfc::field::fd::average_face;
  using pfc::field::fd::FaceAverage;
  const double arithmetic = average_face(FaceAverage::Arithmetic, 1.0, 2.0);
  const double harmonic = average_face(FaceAverage::Harmonic, 1.0, 1.0);
  const auto divergence = pfc::field::fd::face_divergence(
      [](int axis, int i, int j, int k) {
        return double((axis == 0 ? i : axis == 1 ? j : k) + 1);
      },
      3, 4, 5, {true, true, true}, .5, .25, 2., pfc::field::fd::model_walls());
  const bool divergence_ok =
      divergence(0, 0, 0) == 6.5 && divergence(2, 3, 4) == 6.5;
  pfc::sim::SimulationLifecycle life(
      pfc::sim::SimulationLifecycle::schedule(0.0, 0.2, 0.1, 0.2), MPI_COMM_WORLD);
  int accepted = 0;
  life.set_accepted_step_hook([&](const pfc::Time &) { ++accepted; });
  life.run([](double, double) {});
  pfc::field::fd::DiffusionAxis axis(17, 1.0 / 16);
  pfc::field::fd::FaceCondition flux{
      pfc::field::fd::BoundaryQuantity::ConstitutiveFlux,
      [](double time) { return time; }};
  double inventory_rate = 0;
  for (int i = 0; i < 17; ++i)
    inventory_rate +=
        axis.weight(i) *
        axis.apply(i, [](int) { return 1.; }, [](int) { return 2.; }, .3, &flux);
  pfc::field::fd::FluxLedger ledger;
  ledger.begin();
  ledger.stage(.3, 1., .1);
  ledger.reject();
  ledger.begin();
  auto reaction = pfc::field::fd::reaction_face(
      {1., {}, {0., 0., 0.}, {-1., 0., 0.}, .3}, 2.,
      [](const auto &p, double k) {
        return pfc::field::fd::ReactionRate{k * p.state};
      },
      [](const auto &p, double) { return p.state >= 0; }, ledger);
  ledger.stage(reaction.flux(.3, 1), 1, .1);
  ledger.reject();
  ledger.begin();
  double source = pfc::field::fd::reaction_source(
      {2., {}, {0., 0., 0.}, {.6, .8, 0.}, .3}, 3., .125, .02, 2.,
      [](const auto &p, double k) {
        return pfc::field::fd::ReactionRate{k * p.state};
      },
      [](const auto &p, double) { return p.state >= 0; }, ledger);
  ledger.accept();
  const bool source_ok = std::abs(source * .125 * .02 + ledger.accepted()) < 1e-14;
  const bool ok = divergence_ok && source_ok && reaction.flux(.3, 1) == 2. &&
                  arithmetic == 1.5 && harmonic == 1.0 && accepted == 2 &&
                  std::abs(inventory_rate + .3) < 1e-11 &&
                  std::abs(ledger.accepted() - .03) < 1e-14;
  if (ok) {
    std::printf("OpenPFC FD consumer OK: accepted %d\n", accepted);
  }
  int done = 0;
  MPI_Finalized(&done);
  if (ready == 0 && done == 0) {
    MPI_Finalize();
  }
  return ok ? 0 : 1;
}
