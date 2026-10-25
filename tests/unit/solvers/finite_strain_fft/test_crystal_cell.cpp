// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_crystal_cell.cpp
 * @brief Homogeneous FCC cell on the existing finite-strain Newton solver.
 *
 * A uniform stress has no compatible fluctuation, so the periodic solution
 * stays at the prescribed macroscopic deformation. That field is compared
 * with a material-point update that sees the same deformation and the same
 * committed history. The solver is not asked to commit; the test accepts
 * each converged increment explicitly.
 */

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>
#include <openpfc/solvers/finite_strain_fft/newton.hpp>

#include "aluminum_slip.hpp"

#include <mpi.h>

#include <cmath>
#include <vector>

using namespace pfc;
using namespace pfc::finite_strain;

namespace {

const FccSlipParameters kAluminum = aluminum_slip_parameters();
constexpr int kSteps = 10;
constexpr double kShearStep = 0.001;

struct GridTools {
  Domain domain;
  fft::CPUFFT fft;
};

GridTools make_grid(Int3 size, MPI_Comm comm) {
  Domain domain = domain::create(size);
  const int ranks = mpi::get_size(comm);
  auto decomposition = decomposition::create(domain, ranks);
  // MPI_Comm is an integer on this MPI, so the two-argument factory is
  // ambiguous with the rank-id overload. Pass the rank explicitly.
  return GridTools{domain, fft::create(decomposition, mpi::get_rank(comm), comm)};
}

Tensor2 simple_shear(double gamma) {
  Tensor2 value = identity2();
  value(0, 1) = gamma;
  return value;
}

Tensor2 rotation_z(double cosine, double sine) {
  Tensor2 value{};
  value(0, 0) = cosine;
  value(0, 1) = sine;
  value(1, 0) = -sine;
  value(1, 1) = cosine;
  value(2, 2) = 1.0;
  return value;
}

Tensor2 cube_quarter_turn() { return rotation_z(0.0, 1.0); }

Tensor2 forty_five_turn() {
  const double half = std::sqrt(0.5);
  return rotation_z(half, half);
}

double relative_difference(const Tensor2 &value, const Tensor2 &reference) {
  return frobenius_norm(axpy(-1.0, reference, value)) /
         std::max(frobenius_norm(reference), 1.0);
}

double field_norm(const std::vector<Tensor2> &field, MPI_Comm comm) {
  double local = 0.0;
  for (const Tensor2 &value : field) {
    local += frobenius_dot(value, value);
  }
  double reduced = 0.0;
  MPI_Allreduce(&local, &reduced, 1, MPI_DOUBLE, MPI_SUM, comm);
  return std::sqrt(reduced);
}

struct PointStep {
  Tensor2 deformation;
  Tensor2 piola;
  FccSlipState state;
};

std::vector<PointStep> material_point_history(const Tensor2 &orientation) {
  std::vector<PointStep> history;
  history.reserve(static_cast<std::size_t>(kSteps));
  FccSlipState state = fcc_identity_state(kAluminum);
  for (int step = 1; step <= kSteps; ++step) {
    const Tensor2 deformation = simple_shear(kShearStep * step);
    const FccSlipUpdate update =
        integrate_fcc(kAluminum, orientation, state, deformation);
    state = update.state;
    history.push_back(PointStep{deformation, update.piola, state});
  }
  return history;
}

double state_gap(const FccSlipState &value, const FccSlipState &reference) {
  double gap = relative_difference(value.plastic, reference.plastic);
  gap = std::max(gap, relative_difference(value.velocity, reference.velocity));
  for (int system = 0; system < 12; ++system) {
    gap = std::max(
        gap, std::abs(value.resistance[system] - reference.resistance[system]) /
                 std::max(std::abs(reference.resistance[system]), 1.0));
    gap = std::max(gap, std::abs(value.shear[system] - reference.shear[system]) /
                            std::max(std::abs(reference.shear[system]), 1e-4));
  }
  return gap;
}

struct CellRun {
  NewtonResult solved;
};

CellRun run_prepared(CompatibleProjector &projector, FccSlipMaterial &material,
                     MPI_Comm comm, const std::vector<PointStep> *reference) {
  const std::size_t count = material.size();
  IncrementStart start;
  start.deformation.assign(count, identity2());
  CellRun run;
  for (int step = 1; step <= kSteps; ++step) {
    const Tensor2 macroscopic = simple_shear(kShearStep * step);
    material.begin_trial();
    run.solved = solve_increment(projector, material, macroscopic, start, {}, comm);
    REQUIRE(run.solved.converged);
    REQUIRE(run.solved.history.size() >= 2);
    REQUIRE(run.solved.history.back().correction_over_reference < 1e-5);
    const double stress_norm = field_norm(run.solved.piola, comm);
    REQUIRE(run.solved.projected_residual / std::max(stress_norm, 1.0) < 1e-5);
    if (reference != nullptr) {
      const PointStep &point = (*reference)[static_cast<std::size_t>(step - 1)];
      for (std::size_t index = 0; index < count; ++index) {
        REQUIRE(relative_difference(run.solved.deformation[index],
                                    point.deformation) < 1e-8);
        REQUIRE(relative_difference(run.solved.piola[index], point.piola) < 1e-6);
      }
    }
    for (std::size_t index = 0; index < count; ++index) {
      material.stage(index, run.solved.deformation[index]);
    }
    material.accept();
    if (reference != nullptr) {
      const PointStep &point = (*reference)[static_cast<std::size_t>(step - 1)];
      for (std::size_t index = 0; index < count; ++index) {
        REQUIRE(state_gap(material.committed_state(index), point.state) < 1e-8);
      }
    }
    start.deformation = run.solved.deformation;
    start.macroscopic = macroscopic;
  }
  return run;
}

CellRun run_shear(CompatibleProjector &projector, FccSlipMaterial &material,
                  const Tensor2 &orientation, MPI_Comm comm,
                  const std::vector<PointStep> *reference) {
  for (std::size_t index = 0; index < material.size(); ++index) {
    material.set_orientation(index, orientation);
  }
  return run_prepared(projector, material, comm, reference);
}

double field_gap(const std::vector<Tensor2> &value,
                 const std::vector<Tensor2> &reference) {
  double gap = 0.0;
  for (std::size_t index = 0; index < value.size(); ++index) {
    gap = std::max(gap, relative_difference(value[index], reference[index]));
  }
  return gap;
}

double spread(const std::vector<Tensor2> &field, MPI_Comm comm) {
  Tensor2 anchor{};
  if (mpi::get_rank(comm) == 0 && !field.empty()) {
    anchor = field.front();
  }
  MPI_Bcast(&anchor.c[0][0], 9, MPI_DOUBLE, 0, comm);
  double local = 0.0;
  for (const Tensor2 &value : field) {
    local = std::max(local, relative_difference(value, anchor));
  }
  double reduced = 0.0;
  MPI_Allreduce(&local, &reduced, 1, MPI_DOUBLE, MPI_MAX, comm);
  return reduced;
}

} // namespace

TEST_CASE("a homogeneous FCC cell matches the material point",
          "[finite_strain][crystal]") {
  auto tools = make_grid(Int3{3, 3, 3}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  FccSlipMaterial material(projector.local_size(), kAluminum);
  const auto reference = material_point_history(identity2());
  const CellRun run =
      run_shear(projector, material, identity2(), MPI_COMM_WORLD, &reference);
  REQUIRE(spread(run.solved.piola, MPI_COMM_WORLD) < 1e-12);
  REQUIRE(spread(run.solved.deformation, MPI_COMM_WORLD) < 1e-12);
}

TEST_CASE("cube symmetry holds and a forty-five degree turn changes the cell",
          "[finite_strain][crystal]") {
  auto tools = make_grid(Int3{3, 3, 3}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const auto aligned_reference = material_point_history(identity2());
  FccSlipMaterial aligned(projector.local_size(), kAluminum);
  const CellRun identity_run =
      run_shear(projector, aligned, identity2(), MPI_COMM_WORLD, &aligned_reference);

  const auto quarter_reference = material_point_history(cube_quarter_turn());
  FccSlipMaterial quarter(projector.local_size(), kAluminum);
  const CellRun quarter_run = run_shear(projector, quarter, cube_quarter_turn(),
                                        MPI_COMM_WORLD, &quarter_reference);
  REQUIRE(field_gap(quarter_run.solved.piola, identity_run.solved.piola) < 1e-8);
  REQUIRE(spread(quarter_run.solved.piola, MPI_COMM_WORLD) < 1e-12);

  const auto turned_reference = material_point_history(forty_five_turn());
  FccSlipMaterial turned(projector.local_size(), kAluminum);
  const CellRun turned_run = run_shear(projector, turned, forty_five_turn(),
                                       MPI_COMM_WORLD, &turned_reference);
  REQUIRE(field_gap(turned_run.solved.piola, identity_run.solved.piola) > 1e-2);
  REQUIRE(spread(turned_run.solved.piola, MPI_COMM_WORLD) < 1e-12);
  REQUIRE(spread(turned_run.solved.deformation, MPI_COMM_WORLD) < 1e-12);
}

TEST_CASE("symmetry-equivalent grains stay on the material-point path",
          "[finite_strain][crystal]") {
  // Frozen 5^3 field: i fastest, grain 1 where i >= 3. The quarter turn is
  // a cube symmetry of this shear, so every point must follow the identity
  // material-point history.
  constexpr int n = 5;
  auto tools = make_grid(Int3{n, n, n}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  FccSlipMaterial material(projector.local_size(), kAluminum);
  const auto box = tools.fft.get_inbox_bounds();
  int local_grain_one = 0;
  std::size_t index = 0;
  for (int z = box.low[2]; z <= box.high[2]; ++z) {
    for (int y = box.low[1]; y <= box.high[1]; ++y) {
      for (int x = box.low[0]; x <= box.high[0]; ++x, ++index) {
        const bool turned = x >= 3;
        material.set_orientation(index, turned ? cube_quarter_turn() : identity2());
        local_grain_one += turned ? 1 : 0;
      }
    }
  }
  REQUIRE(index == material.size());
  int grain_one = 0;
  MPI_Allreduce(&local_grain_one, &grain_one, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(grain_one == 50);

  const auto aligned = material_point_history(identity2());
  const auto quarter = material_point_history(cube_quarter_turn());
  std::vector<int> turned(material.size(), 0);
  index = 0;
  for (int z = box.low[2]; z <= box.high[2]; ++z) {
    for (int y = box.low[1]; y <= box.high[1]; ++y) {
      for (int x = box.low[0]; x <= box.high[0]; ++x, ++index) {
        turned[index] = x >= 3 ? 1 : 0;
      }
    }
  }
  IncrementStart start;
  start.deformation.assign(material.size(), identity2());
  NewtonResult solved;
  for (int step = 1; step <= kSteps; ++step) {
    const Tensor2 macroscopic = simple_shear(kShearStep * step);
    material.begin_trial();
    solved =
        solve_increment(projector, material, macroscopic, start, {}, MPI_COMM_WORLD);
    REQUIRE(solved.converged);
    const double stress_norm = field_norm(solved.piola, MPI_COMM_WORLD);
    REQUIRE(solved.projected_residual / std::max(stress_norm, 1.0) < 1e-5);
    const PointStep &aligned_step = aligned[static_cast<std::size_t>(step - 1)];
    const PointStep &quarter_step = quarter[static_cast<std::size_t>(step - 1)];
    for (std::size_t cell = 0; cell < material.size(); ++cell) {
      const PointStep &point = turned[cell] == 0 ? aligned_step : quarter_step;
      REQUIRE(relative_difference(solved.deformation[cell], point.deformation) <
              1e-8);
      REQUIRE(relative_difference(solved.piola[cell], aligned_step.piola) < 1e-6);
      material.stage(cell, solved.deformation[cell]);
    }
    material.accept();
    for (std::size_t cell = 0; cell < material.size(); ++cell) {
      const PointStep &point = turned[cell] == 0 ? aligned_step : quarter_step;
      REQUIRE(state_gap(material.committed_state(cell), point.state) < 1e-8);
    }
    start.deformation = solved.deformation;
    start.macroscopic = macroscopic;
  }
  REQUIRE(spread(solved.piola, MPI_COMM_WORLD) < 1e-12);
  REQUIRE(spread(solved.deformation, MPI_COMM_WORLD) < 1e-12);
}

TEST_CASE("two ranks reproduce one FCC cell", "[MPI][finite_strain][crystal]") {
  REQUIRE(mpi::get_size(MPI_COMM_WORLD) == 2);
  const Int3 size{3, 3, 3};
  const Tensor2 orientation = identity2();

  auto serial_tools = make_grid(size, MPI_COMM_SELF);
  CompatibleProjector serial_projector(serial_tools.fft, serial_tools.domain);
  FccSlipMaterial serial_material(serial_projector.local_size(), kAluminum);
  const CellRun serial = run_shear(serial_projector, serial_material, orientation,
                                   MPI_COMM_SELF, nullptr);

  auto parallel_tools = make_grid(size, MPI_COMM_WORLD);
  CompatibleProjector parallel_projector(parallel_tools.fft, parallel_tools.domain);
  FccSlipMaterial parallel_material(parallel_projector.local_size(), kAluminum);
  const CellRun parallel = run_shear(parallel_projector, parallel_material,
                                     orientation, MPI_COMM_WORLD, nullptr);
  REQUIRE(serial.solved.converged);
  REQUIRE(parallel.solved.converged);

  const auto serial_box = serial_tools.fft.get_inbox_bounds();
  const auto parallel_box = parallel_tools.fft.get_inbox_bounds();
  double local_stress = 0.0;
  double local_state = 0.0;
  std::size_t index = 0;
  for (int z = parallel_box.low[2]; z <= parallel_box.high[2]; ++z) {
    for (int y = parallel_box.low[1]; y <= parallel_box.high[1]; ++y) {
      for (int x = parallel_box.low[0]; x <= parallel_box.high[0]; ++x, ++index) {
        const auto global = serial_box.to_linear(std::array<int, 3>{x, y, z});
        const auto global_index = static_cast<std::size_t>(global);
        local_stress = std::max(
            local_stress, relative_difference(parallel.solved.piola[index],
                                              serial.solved.piola[global_index]));
        local_state = std::max(
            local_state, state_gap(parallel_material.committed_state(index),
                                   serial_material.committed_state(global_index)));
      }
    }
  }
  double stress_gap = 0.0;
  double history_gap = 0.0;
  MPI_Allreduce(&local_stress, &stress_gap, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_state, &history_gap, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(stress_gap < 1e-10);
  REQUIRE(history_gap < 1e-10);
}
