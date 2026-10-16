// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_equilibrium.cpp
 * @brief Projector identities and the matrix-free Newton equilibrium solve.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/mechanics/constitutive/saint_venant_kirchhoff.hpp>
#include <openpfc/mechanics/constitutive/simo_j2.hpp>
#include <openpfc/solvers/finite_strain_fft/field_stats.hpp>
#include <openpfc/solvers/finite_strain_fft/inexact.hpp>
#include <openpfc/solvers/finite_strain_fft/newton.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace pfc;
using namespace pfc::finite_strain;

namespace {

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

std::vector<Tensor2> random_field(std::size_t n, std::uint32_t seed) {
  std::vector<Tensor2> field(n);
  std::uint32_t state = seed;
  for (Tensor2 &value : field) {
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        state = state * 1664525u + 1013904223u;
        value(a, b) = static_cast<double>(state >> 8) / 16777216.0 - 0.5;
      }
    }
  }
  return field;
}

Tensor2 simple_shear(double gamma) {
  Tensor2 value = identity2();
  value(0, 1) = gamma;
  return value;
}

/// Stress `P = 2 F` and tangent `dP = 2 dF`. Not Saint-Venant-Kirchhoff.
struct ScaledDeformationLaw {
  [[nodiscard]] Tensor2 stress(std::size_t, const Tensor2 &deformation) const {
    return scaled(deformation, 2.0);
  }
  [[nodiscard]] Tensor2 tangent_action(std::size_t, const Tensor2 &,
                                       const Tensor2 &increment) const {
    return scaled(increment, 2.0);
  }
};

double max_abs_diff(const std::vector<Tensor2> &a, const std::vector<Tensor2> &b) {
  double diff = 0.0;
  for (std::size_t n = 0; n < a.size(); ++n) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        diff = std::max(diff, std::abs(a[n](i, j) - b[n](i, j)));
      }
    }
  }
  return diff;
}

} // namespace

TEST_CASE("even-grid projection is idempotent and drops a pure Nyquist mode",
          "[finite_strain][projector]") {
  auto tools = make_grid(Int3{4, 6, 8}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const auto field = random_field(projector.local_size(), 11u);
  std::vector<Tensor2> once;
  std::vector<Tensor2> twice;
  projector.apply(field, once);
  projector.apply(once, twice);
  REQUIRE(max_abs_diff(once, twice) < 1e-10);

  const auto stats =
      tensor_field_stats(once, tools.fft.get_inbox_bounds(), MPI_COMM_WORLD, false);
  const double scale = stats.frobenius_l2 + 1.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      REQUIRE_THAT(stats.component[a][b].sum / scale, WithinAbs(0.0, 1e-10));
    }
  }

  const auto box = tools.fft.get_inbox_bounds();
  std::vector<Tensor2> nyquist(projector.local_size());
  std::size_t index = 0;
  for (int z = box.low[2]; z <= box.high[2]; ++z) {
    for (int y = box.low[1]; y <= box.high[1]; ++y) {
      for (int x = box.low[0]; x <= box.high[0]; ++x, ++index) {
        const double sign = ((x + y + z) & 1) == 0 ? 1.0 : -1.0;
        nyquist[index](0, 0) = sign;
        nyquist[index](1, 2) = 0.3 * sign;
      }
    }
  }
  std::vector<Tensor2> dropped;
  projector.apply(nyquist, dropped);
  double local = 0.0;
  for (const Tensor2 &value : dropped) {
    local = std::max(local, frobenius_norm(value));
  }
  double reduced = 0.0;
  MPI_Allreduce(&local, &reduced, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(reduced < 1e-10);

  // Compatibility: every retained mode is rank-one along k_component_odd.
  const auto size = domain::get_size(tools.domain);
  const auto wave = fft::kspace::k_frequency_scaling(tools.domain);
  const std::size_t n_freq = tools.fft.size_outbox();
  std::vector<double> real(once.size());
  std::array<std::vector<std::complex<double>>, 9> hat;
  for (auto &component : hat) {
    component.resize(n_freq);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (std::size_t i = 0; i < once.size(); ++i) {
        real[i] = once[i](a, b);
      }
      tools.fft.forward(real, hat[static_cast<std::size_t>(a * 3 + b)]);
    }
  }
  fft::kspace::for_each_kpoint(
      tools.fft.get_outbox_bounds(), tools.domain,
      [&](std::size_t mode, double, double, double, int i, int j, int k) {
        const double qx = fft::kspace::k_component_odd(i, size[0], wave[0]);
        const double qy = fft::kspace::k_component_odd(j, size[1], wave[1]);
        const double qz = fft::kspace::k_component_odd(k, size[2], wave[2]);
        const double q2 = qx * qx + qy * qy + qz * qz;
        const double q[3] = {qx, qy, qz};
        for (int c = 0; c < 9; ++c) {
          if (q2 == 0.0) {
            REQUIRE_THAT(std::abs(hat[static_cast<std::size_t>(c)][mode]),
                         WithinAbs(0.0, 1e-8));
          }
        }
        if (q2 == 0.0) {
          return;
        }
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            for (int c = 0; c < 3; ++c) {
              const auto left =
                  hat[static_cast<std::size_t>(a * 3 + b)][mode] * q[c];
              const auto right =
                  hat[static_cast<std::size_t>(a * 3 + c)][mode] * q[b];
              REQUIRE_THAT(std::abs(left - right),
                           WithinAbs(0.0, 1e-8 * (q2 + 1.0)));
            }
          }
        }
        // A self-conjugate Nyquist corner is real. Partner modes in a
        // stored Nyquist plane agree as conjugates when both are local.
        const bool self =
            (j == 0 || j == size[1] / 2) && (k == 0 || k == size[2] / 2);
        if ((i == 0 || i == size[0] / 2) && self) {
          for (int c = 0; c < 9; ++c) {
            REQUIRE_THAT(std::abs(hat[static_cast<std::size_t>(c)][mode].imag()),
                         WithinAbs(0.0, 1e-8));
          }
        }
      });
}

TEST_CASE("an even-grid spectral gradient stays and a raw Nyquist wave does not",
          "[finite_strain][projector]") {
  auto tools = make_grid(Int3{8, 4, 6}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const std::size_t n = projector.local_size();
  const auto inbox = tools.fft.get_inbox_bounds();
  const int nx = domain::get_size(tools.domain)[0];
  std::vector<double> scalar(n);
  std::size_t index = 0;
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z) {
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y) {
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x, ++index) {
        scalar[index] = std::sin(2.0 * constants::pi * static_cast<double>(x) /
                                 static_cast<double>(nx));
      }
    }
  }
  std::vector<std::complex<double>> hat(tools.fft.size_outbox());
  tools.fft.forward(scalar, hat);
  const auto size = domain::get_size(tools.domain);
  const auto wave = fft::kspace::k_frequency_scaling(tools.domain);
  fft::kspace::for_each_kpoint(
      tools.fft.get_outbox_bounds(), tools.domain,
      [&](std::size_t mode, double, double, double, int i, int, int) {
        hat[mode] *= std::complex<double>(
            0.0, fft::kspace::k_component_odd(i, size[0], wave[0]));
      });
  std::vector<double> derivative(n);
  tools.fft.backward(hat, derivative);
  std::vector<Tensor2> gradient(n);
  for (std::size_t i = 0; i < n; ++i) {
    gradient[i](0, 0) = derivative[i];
  }
  std::vector<Tensor2> projected;
  projector.apply(gradient, projected);
  REQUIRE(max_abs_diff(gradient, projected) < 1e-10);

  // The one-sided Nyquist wave is not the derivative this projector uses.
  std::vector<Tensor2> raw(n);
  index = 0;
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z) {
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y) {
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x, ++index) {
        raw[index](0, 0) = ((x & 1) == 0 ? 1.0 : -1.0) *
                           std::sin(2.0 * constants::pi * static_cast<double>(y) /
                                    static_cast<double>(size[1]));
      }
    }
  }
  std::vector<Tensor2> projected_raw;
  projector.apply(raw, projected_raw);
  REQUIRE(max_abs_diff(raw, projected_raw) > 1e-3);
  projector.apply(projected_raw, projected);
  REQUIRE(max_abs_diff(projected_raw, projected) < 1e-10);
}

TEST_CASE("projection removes the mean and is idempotent",
          "[finite_strain][projector]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const auto field = random_field(projector.local_size(), 3u);
  std::vector<Tensor2> once;
  std::vector<Tensor2> twice;
  projector.apply(field, once);
  projector.apply(once, twice);
  REQUIRE(max_abs_diff(once, twice) < 1e-10);

  const auto stats =
      tensor_field_stats(once, tools.fft.get_inbox_bounds(), MPI_COMM_WORLD, false);
  const double scale = stats.frobenius_l2 + 1.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      REQUIRE_THAT(stats.component[a][b].sum / scale, WithinAbs(0.0, 1e-10));
    }
  }
}

TEST_CASE("a constant tensor is removed and a gradient is left unchanged",
          "[finite_strain][projector]") {
  auto tools = make_grid(Int3{7, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const std::size_t n = projector.local_size();
  Tensor2 constant;
  constant(0, 1) = 0.4;
  constant(2, 2) = -0.2;
  std::vector<Tensor2> uniform(n, constant);
  std::vector<Tensor2> projected_uniform;
  projector.apply(uniform, projected_uniform);
  REQUIRE(max_abs_diff(projected_uniform, std::vector<Tensor2>(n)) < 1e-10);

  const auto inbox = tools.fft.get_inbox_bounds();
  const int nx = domain::get_size(tools.domain)[0];
  std::vector<double> scalar(n);
  std::size_t index = 0;
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z) {
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y) {
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x, ++index) {
        scalar[index] = std::sin(2.0 * constants::pi * static_cast<double>(x) /
                                 static_cast<double>(nx));
      }
    }
  }
  std::vector<std::complex<double>> hat(tools.fft.size_outbox());
  tools.fft.forward(scalar, hat);
  const auto outbox = tools.fft.get_outbox_bounds();
  fft::kspace::for_each_kpoint(
      outbox, tools.domain,
      [&](std::size_t mode, double qx, double, double, int, int, int) {
        hat[mode] *= std::complex<double>(0.0, qx);
      });
  std::vector<double> derivative(n);
  tools.fft.backward(hat, derivative);

  std::vector<Tensor2> gradient(n);
  for (std::size_t i = 0; i < n; ++i) {
    gradient[i](0, 0) = derivative[i];
  }
  std::vector<Tensor2> projected_gradient;
  projector.apply(gradient, projected_gradient);
  REQUIRE(max_abs_diff(gradient, projected_gradient) < 1e-10);
}

TEST_CASE("projected nonzero modes are rank-one along the wave vector",
          "[finite_strain][projector]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const auto field = random_field(projector.local_size(), 9u);
  std::vector<Tensor2> projected;
  projector.apply(field, projected);

  const std::size_t n_freq = tools.fft.size_outbox();
  std::vector<double> real(projected.size());
  std::vector<std::vector<std::complex<double>>> hat(
      9, std::vector<std::complex<double>>(n_freq));
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (std::size_t i = 0; i < projected.size(); ++i) {
        real[i] = projected[i](a, b);
      }
      tools.fft.forward(real, hat[static_cast<std::size_t>(a * 3 + b)]);
    }
  }

  const auto outbox = tools.fft.get_outbox_bounds();
  fft::kspace::for_each_kpoint(
      outbox, tools.domain,
      [&](std::size_t mode, double qx, double qy, double qz, int, int, int) {
        const double q2 = qx * qx + qy * qy + qz * qz;
        const double q[3] = {qx, qy, qz};
        if (q2 == 0.0) {
          for (int c = 0; c < 9; ++c) {
            REQUIRE_THAT(std::abs(hat[static_cast<std::size_t>(c)][mode]),
                         WithinAbs(0.0, 1e-8));
          }
          return;
        }
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            for (int c = 0; c < 3; ++c) {
              const auto left =
                  hat[static_cast<std::size_t>(a * 3 + b)][mode] * q[c];
              const auto right =
                  hat[static_cast<std::size_t>(a * 3 + c)][mode] * q[b];
              REQUIRE_THAT(std::abs(left - right),
                           WithinAbs(0.0, 1e-8 * (q2 + 1.0)));
            }
          }
        }
      });
}

TEST_CASE("homogeneous simple shear stays affine and equilibrated",
          "[finite_strain][newton]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const IsotropicModuli soft{0.833, 0.386};
  const std::vector<std::uint8_t> uniform;
  const SaintVenantKirchhoffMaterial material(uniform, projector.local_size(), soft,
                                              soft);
  NewtonControls controls;
  const auto solved = solve_equilibrium(projector, material, simple_shear(0.5),
                                        controls, MPI_COMM_WORLD);
  REQUIRE(solved.converged);
  REQUIRE(solved.history.size() == 2);
  Tensor2 expected = identity2();
  expected(0, 1) = 0.5;
  std::vector<Tensor2> affine(solved.deformation.size(), expected);
  REQUIRE(max_abs_diff(solved.deformation, affine) < 1e-8);
  REQUIRE_THAT(solved.projected_residual, WithinAbs(0.0, 1e-8));
  for (const NewtonStep &step : solved.history) {
    REQUIRE_THAT(step.correction_over_reference, WithinAbs(0.0, 1e-12));
    REQUIRE(step.krylov_iterations == 0);
  }
}

TEST_CASE("a cubic even grid accepts an in-place projection",
          "[finite_strain][projector]") {
  auto tools = make_grid(Int3{4, 4, 4}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  auto field = random_field(projector.local_size(), 19u);
  projector.apply(field, field);
  std::vector<Tensor2> again;
  projector.apply(field, again);
  REQUIRE(max_abs_diff(field, again) < 1e-10);
}

TEST_CASE("even-grid homogeneous and two-phase shear reach equilibrium",
          "[finite_strain][newton][projector]") {
  auto tools = make_grid(Int3{8, 4, 6}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const IsotropicModuli soft{0.833, 0.386};
  const std::vector<std::uint8_t> no_phase;
  const SaintVenantKirchhoffMaterial uniform(no_phase, projector.local_size(), soft,
                                             soft);
  const auto solved =
      solve_equilibrium(projector, uniform, simple_shear(0.5), {}, MPI_COMM_WORLD);
  REQUIRE(solved.converged);
  Tensor2 expected = identity2();
  expected(0, 1) = 0.5;
  REQUIRE(max_abs_diff(solved.deformation,
                       std::vector<Tensor2>(solved.deformation.size(), expected)) <
          1e-8);
  REQUIRE_THAT(solved.projected_residual, WithinAbs(0.0, 1e-8));

  const auto inbox = tools.fft.get_inbox_bounds();
  std::vector<std::uint8_t> phase(projector.local_size(), 0);
  std::size_t index = 0;
  int hard = 0;
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z) {
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y) {
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x, ++index) {
        if (x >= 2 && y < 2 && z >= 2) {
          phase[index] = 1;
          ++hard;
        }
      }
    }
  }
  int hard_global = 0;
  MPI_Allreduce(&hard, &hard_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(hard_global > 0);
  const IsotropicModuli stiff{8.33, 3.86};
  const SaintVenantKirchhoffMaterial material(phase, projector.local_size(), soft,
                                              stiff);
  NewtonControls controls;
  controls.krylov_method = KrylovMethod::GMRES;
  const auto mixed = solve_equilibrium(projector, material, simple_shear(0.25),
                                       controls, MPI_COMM_WORLD);
  REQUIRE(mixed.converged);
  const double stress_norm =
      tensor_field_stats(mixed.piola, inbox, MPI_COMM_WORLD, false).frobenius_l2;
  REQUIRE(stress_norm > 0.1);
  REQUIRE(mixed.projected_residual < 1e-6 * stress_norm);
  controls.krylov_method = KrylovMethod::CG;
  REQUIRE_THROWS_AS(
      solve_equilibrium(projector, material, simple_shear(0.25), controls),
      std::invalid_argument);
}

TEST_CASE("a two-phase shear reaches the projected equilibrium tolerance",
          "[finite_strain][newton]") {
  auto tools = make_grid(Int3{7, 7, 7}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const auto inbox = tools.fft.get_inbox_bounds();
  std::vector<std::uint8_t> phase(projector.local_size(), 0);
  std::size_t index = 0;
  int hard = 0;
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z) {
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y) {
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x, ++index) {
        if (x >= 5 && y < 2 && z >= 5) {
          phase[index] = 1;
          ++hard;
        }
      }
    }
  }
  int hard_global = 0;
  MPI_Allreduce(&hard, &hard_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(hard_global > 0);

  const IsotropicModuli soft{0.833, 0.386};
  const IsotropicModuli stiff{8.33, 3.86};
  const SaintVenantKirchhoffMaterial material(phase, projector.local_size(), soft,
                                              stiff);
  const auto solved =
      solve_equilibrium(projector, material, simple_shear(0.5), {}, MPI_COMM_WORLD);
  REQUIRE(solved.converged);
  REQUIRE(solved.history.size() >= 2);
  const double stress_norm =
      tensor_field_stats(solved.piola, inbox, MPI_COMM_WORLD, false).frobenius_l2;
  REQUIRE(stress_norm > 1.0);
  REQUIRE(solved.projected_residual < 1e-6 * stress_norm);
  REQUIRE(solved.history.back().correction_over_reference < 1e-5);

  double fluctuation = 0.0;
  for (const Tensor2 &value : solved.deformation) {
    fluctuation = std::max(fluctuation, std::abs(value(0, 1) - 0.5));
  }
  double fluctuation_global = 0.0;
  MPI_Allreduce(&fluctuation, &fluctuation_global, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  REQUIRE(fluctuation_global > 1e-3);
}

TEST_CASE("two ranks reproduce the single-rank finite-strain solution",
          "[MPI][finite_strain][newton]") {
  REQUIRE(mpi::get_size(MPI_COMM_WORLD) == 2);
  const Int3 size{5, 5, 7};
  const IsotropicModuli soft{1.0, 0.4};
  const IsotropicModuli stiff{3.0, 1.2};

  auto serial_domain = domain::create(size);
  auto serial_decomposition = decomposition::create(serial_domain, 1);
  auto serial_fft =
      fft::create(serial_decomposition, mpi::get_rank(MPI_COMM_SELF), MPI_COMM_SELF);
  CompatibleProjector serial_projector(serial_fft, serial_domain);
  const auto serial_box = serial_fft.get_inbox_bounds();
  std::vector<std::uint8_t> serial_phase(serial_projector.local_size(), 0);
  std::size_t serial_index = 0;
  for (int z = serial_box.low[2]; z <= serial_box.high[2]; ++z) {
    for (int y = serial_box.low[1]; y <= serial_box.high[1]; ++y) {
      for (int x = serial_box.low[0]; x <= serial_box.high[0]; ++x, ++serial_index) {
        if (x >= 3 && y < 2 && z >= 4) {
          serial_phase[serial_index] = 1;
        }
      }
    }
  }
  const SaintVenantKirchhoffMaterial serial_material(
      serial_phase, serial_projector.local_size(), soft, stiff);
  const auto serial = solve_equilibrium(serial_projector, serial_material,
                                        simple_shear(0.35), {}, MPI_COMM_SELF);

  auto parallel_tools = make_grid(size, MPI_COMM_WORLD);
  CompatibleProjector parallel_projector(parallel_tools.fft, parallel_tools.domain);
  const auto parallel_box = parallel_tools.fft.get_inbox_bounds();
  std::vector<std::uint8_t> parallel_phase(parallel_projector.local_size(), 0);
  std::size_t parallel_index = 0;
  for (int z = parallel_box.low[2]; z <= parallel_box.high[2]; ++z) {
    for (int y = parallel_box.low[1]; y <= parallel_box.high[1]; ++y) {
      for (int x = parallel_box.low[0]; x <= parallel_box.high[0];
           ++x, ++parallel_index) {
        if (x >= 3 && y < 2 && z >= 4) {
          parallel_phase[parallel_index] = 1;
        }
      }
    }
  }
  const SaintVenantKirchhoffMaterial parallel_material(
      parallel_phase, parallel_projector.local_size(), soft, stiff);
  const auto parallel = solve_equilibrium(parallel_projector, parallel_material,
                                          simple_shear(0.35), {}, MPI_COMM_WORLD);
  REQUIRE(parallel.converged);
  REQUIRE(serial.converged);

  parallel_index = 0;
  double local_diff = 0.0;
  for (int z = parallel_box.low[2]; z <= parallel_box.high[2]; ++z) {
    for (int y = parallel_box.low[1]; y <= parallel_box.high[1]; ++y) {
      for (int x = parallel_box.low[0]; x <= parallel_box.high[0];
           ++x, ++parallel_index) {
        const auto global = serial_box.to_linear(std::array<int, 3>{x, y, z});
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            local_diff = std::max(
                local_diff,
                std::abs(
                    parallel.deformation[parallel_index](a, b) -
                    serial.deformation[static_cast<std::size_t>(global)](a, b)));
            local_diff = std::max(
                local_diff,
                std::abs(parallel.piola[parallel_index](a, b) -
                         serial.piola[static_cast<std::size_t>(global)](a, b)));
          }
        }
      }
    }
  }
  double diff = 0.0;
  MPI_Allreduce(&local_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(diff < 1e-8);
}

TEST_CASE("a homogeneous prescribed deformation stays affine",
          "[finite_strain][newton]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  Tensor2 macroscopic = identity2();
  macroscopic(0, 0) = 1.2;
  macroscopic(1, 2) = -0.3;
  macroscopic(2, 0) = 0.15;
  const IsotropicModuli soft{0.833, 0.386};
  const std::vector<std::uint8_t> uniform;
  const SaintVenantKirchhoffMaterial material(uniform, projector.local_size(), soft,
                                              soft);
  const auto solved =
      solve_equilibrium(projector, material, macroscopic, {}, MPI_COMM_WORLD);
  REQUIRE(solved.converged);
  REQUIRE(max_abs_diff(
              solved.deformation,
              std::vector<Tensor2>(solved.deformation.size(), macroscopic)) < 1e-8);
  REQUIRE_THAT(solved.projected_residual, WithinAbs(0.0, 1e-8));
}

TEST_CASE("equilibrium uses the supplied constitutive law",
          "[finite_strain][newton]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const Tensor2 macroscopic = simple_shear(0.4);
  const auto solved = solve_equilibrium(projector, ScaledDeformationLaw{},
                                        macroscopic, {}, MPI_COMM_WORLD);
  REQUIRE(solved.converged);
  std::vector<Tensor2> expected_stress(solved.piola.size(),
                                       scaled(macroscopic, 2.0));
  REQUIRE(max_abs_diff(solved.piola, expected_stress) < 1e-8);

  const IsotropicModuli soft{0.833, 0.386};
  const Tensor2 saint_venant = constitutive_response(macroscopic, soft).piola;
  REQUIRE(frobenius_norm(add(saint_venant, scaled(expected_stress.front(), -1.0))) >
          1e-3);
}

TEST_CASE("an increment from the identity matches the one-shot equilibrium",
          "[finite_strain][newton]") {
  auto tools = make_grid(Int3{7, 7, 7}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const auto inbox = tools.fft.get_inbox_bounds();
  std::vector<std::uint8_t> phase(projector.local_size(), 0);
  std::size_t index = 0;
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z) {
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y) {
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x, ++index) {
        if (x >= 5 && y < 2 && z >= 5) {
          phase[index] = 1;
        }
      }
    }
  }
  const IsotropicModuli soft{0.833, 0.386};
  const IsotropicModuli stiff{8.33, 3.86};
  const SaintVenantKirchhoffMaterial material(phase, projector.local_size(), soft,
                                              stiff);
  const Tensor2 macroscopic = simple_shear(0.5);
  const auto direct =
      solve_equilibrium(projector, material, macroscopic, {}, MPI_COMM_WORLD);
  IncrementStart start;
  start.deformation.assign(projector.local_size(), identity2());
  const auto incremental =
      solve_increment(projector, material, macroscopic, start, {}, MPI_COMM_WORLD);
  REQUIRE(direct.converged);
  REQUIRE(incremental.converged);
  REQUIRE(direct.history.size() == incremental.history.size());
  REQUIRE(max_abs_diff(direct.deformation, incremental.deformation) < 1e-12);
  REQUIRE(max_abs_diff(direct.piola, incremental.piola) < 1e-12);
  for (std::size_t step = 0; step < direct.history.size(); ++step) {
    REQUIRE(direct.history[step].krylov_iterations ==
            incremental.history[step].krylov_iterations);
    REQUIRE_THAT(
        direct.history[step].correction_over_reference,
        WithinAbs(incremental.history[step].correction_over_reference, 1e-12));
  }
}

TEST_CASE("a homogeneous J2 increment stays affine until the caller commits",
          "[finite_strain][newton][j2]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const J2Moduli soft{0.833, 0.386, 0.004, 0.003};
  const std::vector<std::uint8_t> uniform;
  SimoJ2Material material(uniform, projector.local_size(), soft, soft);
  Tensor2 macroscopic = identity2();
  macroscopic(0, 0) = 1.05;
  macroscopic(1, 1) = 1.0 / 1.05;
  IncrementStart start;
  start.deformation.assign(projector.local_size(), identity2());
  material.begin_trial();
  const auto solved =
      solve_increment(projector, material, macroscopic, start, {}, MPI_COMM_WORLD);
  REQUIRE(solved.converged);
  REQUIRE(max_abs_diff(
              solved.deformation,
              std::vector<Tensor2>(solved.deformation.size(), macroscopic)) < 1e-8);
  REQUIRE_THAT(solved.projected_residual, WithinAbs(0.0, 1e-8));
  for (std::size_t i = 0; i < material.size(); ++i) {
    REQUIRE(material.accumulated(i) == 0.0);
  }
  material.reject();
  for (std::size_t i = 0; i < material.size(); ++i) {
    REQUIRE(material.accumulated(i) == 0.0);
  }

  material.begin_trial();
  for (std::size_t i = 0; i < material.size(); ++i) {
    material.stage(i, solved.deformation[i]);
  }
  material.accept();
  REQUIRE(material.accumulated(0) > 0.0);
  for (std::size_t i = 1; i < material.size(); ++i) {
    REQUIRE(material.accumulated(i) == material.accumulated(0));
  }
}

namespace {

Tensor2 published_pure_shear(double lambda) {
  Tensor2 macroscopic = identity2();
  macroscopic(0, 0) = 1.0 + lambda;
  macroscopic(1, 1) = 1.0 / macroscopic(0, 0);
  return macroscopic;
}

Tensor2 published_pure_shear_step() {
  // First increment of the released program: lam += 0.2/50.
  return published_pure_shear(0.2 / 50.0);
}

std::vector<std::uint8_t> inclusion_phase(const fft::CPUFFT &fft,
                                          std::size_t count) {
  const auto box = fft.get_inbox_bounds();
  std::vector<std::uint8_t> phase(count, 0);
  std::size_t index = 0;
  for (int z = box.low[2]; z <= box.high[2]; ++z) {
    for (int y = box.low[1]; y <= box.high[1]; ++y) {
      for (int x = box.low[0]; x <= box.high[0]; ++x, ++index) {
        if (x < 3 && y < 3) {
          phase[index] = 1;
        }
      }
    }
  }
  return phase;
}

void accept_field(SimoJ2Material &material,
                  const std::vector<Tensor2> &deformation) {
  material.begin_trial();
  for (std::size_t i = 0; i < material.size(); ++i) {
    material.stage(i, deformation[i]);
  }
  material.accept();
}

} // namespace

TEST_CASE("a heterogeneous J2 increment converges", "[finite_strain][newton][j2]") {
  auto tools = make_grid(Int3{5, 5, 7}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  const J2Moduli soft{0.833, 0.386, 0.004, 0.003};
  const J2Moduli hard{0.833, 0.386, 0.008, 0.006};
  const auto phase = inclusion_phase(tools.fft, projector.local_size());
  SimoJ2Material material(phase, projector.local_size(), soft, hard);
  const Tensor2 macroscopic = published_pure_shear_step();
  IncrementStart start;
  start.deformation.assign(projector.local_size(), identity2());
  material.begin_trial();
  const auto solved =
      solve_increment(projector, material, macroscopic, start, {}, MPI_COMM_WORLD);
  REQUIRE(solved.converged);
  REQUIRE(solved.history.size() >= 2);
  REQUIRE(solved.history.back().krylov_iterations < 200);
  for (std::size_t i = 0; i < material.size(); ++i) {
    REQUIRE(material.accumulated(i) == 0.0);
  }
  material.reject();
  for (std::size_t i = 0; i < material.size(); ++i) {
    REQUIRE(material.accumulated(i) == 0.0);
  }
  accept_field(material, solved.deformation);
  double plastic_max = 0.0;
  for (std::size_t i = 0; i < material.size(); ++i) {
    plastic_max = std::max(plastic_max, material.accumulated(i));
  }
  REQUIRE(plastic_max > 0.0);

  const Tensor2 second = published_pure_shear(2.0 * 0.2 / 50.0);
  IncrementStart continued;
  continued.deformation = solved.deformation;
  continued.macroscopic = macroscopic;
  material.begin_trial();
  const auto next =
      solve_increment(projector, material, second, continued, {}, MPI_COMM_WORLD);
  REQUIRE(next.converged);
  REQUIRE(next.history.front().krylov_iterations < 200);
  REQUIRE(next.history.back().krylov_iterations < 200);
}

TEST_CASE("two ranks reproduce one J2 increment", "[MPI][finite_strain][j2]") {
  REQUIRE(mpi::get_size(MPI_COMM_WORLD) == 2);
  const Int3 size{5, 5, 7};
  const J2Moduli soft{0.833, 0.386, 0.004, 0.003};
  const J2Moduli hard{0.833, 0.386, 0.008, 0.006};
  const Tensor2 macroscopic = published_pure_shear_step();

  auto serial_domain = domain::create(size);
  auto serial_decomposition = decomposition::create(serial_domain, 1);
  auto serial_fft =
      fft::create(serial_decomposition, mpi::get_rank(MPI_COMM_SELF), MPI_COMM_SELF);
  CompatibleProjector serial_projector(serial_fft, serial_domain);
  const auto serial_box = serial_fft.get_inbox_bounds();
  std::vector<std::uint8_t> serial_phase(serial_projector.local_size(), 0);
  std::size_t serial_index = 0;
  for (int z = serial_box.low[2]; z <= serial_box.high[2]; ++z) {
    for (int y = serial_box.low[1]; y <= serial_box.high[1]; ++y) {
      for (int x = serial_box.low[0]; x <= serial_box.high[0]; ++x, ++serial_index) {
        if (x < 3 && y < 3) {
          serial_phase[serial_index] = 1;
        }
      }
    }
  }
  SimoJ2Material serial_material(serial_phase, serial_projector.local_size(), soft,
                                 hard);
  IncrementStart serial_start;
  serial_start.deformation.assign(serial_projector.local_size(), identity2());
  const auto serial = solve_increment(serial_projector, serial_material, macroscopic,
                                      serial_start, {}, MPI_COMM_SELF);

  auto parallel_tools = make_grid(size, MPI_COMM_WORLD);
  CompatibleProjector parallel_projector(parallel_tools.fft, parallel_tools.domain);
  const auto parallel_box = parallel_tools.fft.get_inbox_bounds();
  std::vector<std::uint8_t> parallel_phase(parallel_projector.local_size(), 0);
  std::size_t parallel_index = 0;
  for (int z = parallel_box.low[2]; z <= parallel_box.high[2]; ++z) {
    for (int y = parallel_box.low[1]; y <= parallel_box.high[1]; ++y) {
      for (int x = parallel_box.low[0]; x <= parallel_box.high[0];
           ++x, ++parallel_index) {
        if (x < 3 && y < 3) {
          parallel_phase[parallel_index] = 1;
        }
      }
    }
  }
  SimoJ2Material parallel_material(parallel_phase, parallel_projector.local_size(),
                                   soft, hard);
  IncrementStart parallel_start;
  parallel_start.deformation.assign(parallel_projector.local_size(), identity2());
  const auto parallel =
      solve_increment(parallel_projector, parallel_material, macroscopic,
                      parallel_start, {}, MPI_COMM_WORLD);
  REQUIRE(serial.converged);
  REQUIRE(parallel.converged);

  auto field_diff = [&](const NewtonResult &parallel_result,
                        const NewtonResult &serial_result) {
    parallel_index = 0;
    double local_diff = 0.0;
    for (int z = parallel_box.low[2]; z <= parallel_box.high[2]; ++z) {
      for (int y = parallel_box.low[1]; y <= parallel_box.high[1]; ++y) {
        for (int x = parallel_box.low[0]; x <= parallel_box.high[0];
             ++x, ++parallel_index) {
          const auto global = serial_box.to_linear(std::array<int, 3>{x, y, z});
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              local_diff = std::max(
                  local_diff,
                  std::abs(
                      parallel_result.deformation[parallel_index](a, b) -
                      serial_result.deformation[static_cast<std::size_t>(global)](
                          a, b)));
              local_diff = std::max(
                  local_diff,
                  std::abs(
                      parallel_result.piola[parallel_index](a, b) -
                      serial_result.piola[static_cast<std::size_t>(global)](a, b)));
            }
          }
        }
      }
    }
    double reduced = 0.0;
    MPI_Allreduce(&local_diff, &reduced, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return reduced;
  };
  REQUIRE(field_diff(parallel, serial) < 1e-8);

  accept_field(serial_material, serial.deformation);
  accept_field(parallel_material, parallel.deformation);
  const Tensor2 second = published_pure_shear(2.0 * 0.2 / 50.0);
  IncrementStart serial_continued;
  serial_continued.deformation = serial.deformation;
  serial_continued.macroscopic = macroscopic;
  IncrementStart parallel_continued;
  parallel_continued.deformation = parallel.deformation;
  parallel_continued.macroscopic = macroscopic;
  serial_material.begin_trial();
  const auto serial_again =
      solve_increment(serial_projector, serial_material, second, serial_continued,
                      {}, MPI_COMM_SELF);
  parallel_material.begin_trial();
  const auto parallel_again =
      solve_increment(parallel_projector, parallel_material, second,
                      parallel_continued, {}, MPI_COMM_WORLD);
  REQUIRE(serial_again.converged);
  REQUIRE(parallel_again.converged);
  REQUIRE(field_diff(parallel_again, serial_again) < 1e-8);
}

TEST_CASE("FFT exposes GMRES and caller-established compatible SPD CG",
          "[finite_strain][krylov]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  struct HeterogeneousScaled {
    std::vector<double> stiffness;
    Tensor2 stress(std::size_t i, const Tensor2 &f) const {
      return scaled(f, stiffness[i]);
    }
    Tensor2 tangent_action(std::size_t i, const Tensor2 &, const Tensor2 &v) const {
      return scaled(v, stiffness[i]);
    }
  } law;
  const auto inbox = tools.fft.get_inbox_bounds();
  for (int z = inbox.low[2]; z <= inbox.high[2]; ++z)
    for (int y = inbox.low[1]; y <= inbox.high[1]; ++y)
      for (int x = inbox.low[0]; x <= inbox.high[0]; ++x)
        law.stiffness.push_back(2 + .5 * std::sin(2 * std::acos(-1.) * x / 5));
  NewtonControls c;
  c.krylov_method = KrylovMethod::CG;
  REQUIRE_THROWS_AS(solve_equilibrium(projector, law, simple_shear(.1), c),
                    std::invalid_argument);
  // G:a(X)I is SPD on range(G): v:G:a v = sum a(X)|v|² > 0.
  c.operator_is_spd = true;
  auto cg_result = solve_equilibrium(projector, law, simple_shear(.1), c);
  c.krylov_method = KrylovMethod::GMRES;
  auto gmres_result = solve_equilibrium(projector, law, simple_shear(.1), c);
  REQUIRE(cg_result.converged);
  REQUIRE(cg_result.history[1].krylov_iterations > 0);
  REQUIRE(gmres_result.converged);
  REQUIRE(max_abs_diff(cg_result.deformation, gmres_result.deformation) < 1e-12);
  for (const auto &step : gmres_result.history) REQUIRE(step.linear.converged());
}

TEST_CASE("residual Newton preserves FFT prescribed macro and compatibility",
          "[finite_strain][nonlinear]") {
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  IncrementStart start;
  start.deformation.assign(projector.local_size(), identity2());
  solvers::NewtonControls c;
  auto r = solve_nonlinear_increment(projector, ScaledDeformationLaw{},
                                     simple_shear(.1), start, c);
  REQUIRE(r.nonlinear.converged());
  auto expected = std::vector<Tensor2>(projector.local_size(), simple_shear(.1));
  REQUIRE(max_abs_diff(r.deformation, expected) < 1e-12);
  REQUIRE(max_abs_diff(start.deformation,
                       std::vector<Tensor2>(projector.local_size(), identity2())) ==
          0);
}

TEST_CASE("FFT Newton propagates a rank-local material failure collectively",
          "[MPI][finite_strain][nonlinear][material_failure]") {
  REQUIRE(mpi::get_size(MPI_COMM_WORLD) == 2);
  auto tools = make_grid(Int3{5, 5, 5}, MPI_COMM_WORLD);
  CompatibleProjector projector(tools.fft, tools.domain);
  struct FailingLaw : ScaledDeformationLaw {
    int rank;
    bool fail_tangent;
    int committed = 27;
    Tensor2 stress(std::size_t i, const Tensor2 &f) const {
      if (rank == 1 && !fail_tangent)
        throw std::runtime_error("rank-local stress failure");
      return ScaledDeformationLaw::stress(i, f);
    }
    Tensor2 tangent_action(std::size_t i, const Tensor2 &f, const Tensor2 &v) const {
      if (rank == 1 && fail_tangent)
        throw std::runtime_error("rank-local tangent failure");
      return ScaledDeformationLaw::tangent_action(i, f, v);
    }
  } law{{}, mpi::get_rank(MPI_COMM_WORLD), false};
  IncrementStart start;
  auto fluctuation = random_field(projector.local_size(), 417u + law.rank);
  projector.apply(fluctuation, start.deformation);
  for (auto &f : start.deformation) f = axpy(.01, f, identity2());
  const auto before = start.deformation;
  for (bool tangent : {false, true}) {
    law.fail_tangent = tangent;
    REQUIRE_THROWS_AS(solve_nonlinear_increment(projector, law, identity2(), start),
                      std::runtime_error);
    REQUIRE(law.committed == 27);
    REQUIRE(max_abs_diff(start.deformation, before) == 0);
  }
}
