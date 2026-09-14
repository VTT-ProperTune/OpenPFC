// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file microelasticity_hip_parity.cpp
 * @brief CPU vs device Green-operator parity: homogeneous Eshelby, a
 *        heterogeneous modulus, and d f_el / d phi.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "microelasticity_hip_parity requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc_apps/microelasticity.hpp>
#include <openpfc_apps/microelasticity_hip.hpp>

namespace {

using RealField = pfc::data::Field<double>;
using pfc::apps::DeviceEigenstrainMicroelasticity;
using pfc::apps::EigenstrainMicroelasticity;
using pfc::apps::kSymComponents;
using pfc::apps::MicroelasticityParams;
using pfc::apps::Stiffness;
using pfc::apps::Sym3;

[[nodiscard]] double reduce_max(double x, MPI_Comm comm) {
  double out = 0.0;
  MPI_Allreduce(&x, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
  return out;
}

[[nodiscard]] double field_maxdiff(const RealField &a, const RealField &b,
                                    MPI_Comm comm) {
  double local = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    local = std::max(local, std::abs(a.data()[i] - b.data()[i]));
  return reduce_max(local, comm);
}

} // namespace

int main(int argc, char **argv) {
  pfc::runtime::gpu::bind_local_device_before_mpi();
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int rc = 0;
  {
  int N = 24;
  double tol_strain = 1.0e-10;
  double tol_dfel = 1.0e-9;
  for (int i = 1; i < argc; ++i) {
    const std::string_view tok(argv[i]);
    const auto eq = tok.find('=');
    if (!tok.starts_with("--") || eq == std::string_view::npos) continue;
    const auto key = tok.substr(2, eq - 2);
    const auto val = std::string(tok.substr(eq + 1));
    if (key == "n") N = std::stoi(val);
    else if (key == "tol-strain") tol_strain = std::stod(val);
    else if (key == "tol-dfel") tol_dfel = std::stod(val);
  }

  const pfc::Domain domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto host_stack = std::make_unique<pfc::sim::stacks::SpectralCPUStack>(
      domain, rank, nproc, MPI_COMM_WORLD);
  auto dev_stack =
      std::make_unique<pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>(
          domain, rank, nproc, MPI_COMM_WORLD);

  auto h = pfc::data::field_from_inbox<double>(
      domain, host_stack->fft().get_inbox_bounds());
  auto amp = pfc::data::field_from_inbox<double>(
      domain, host_stack->fft().get_inbox_bounds());
  auto dh = pfc::data::field_from_inbox<double>(
      domain, host_stack->fft().get_inbox_bounds());
  auto damp = pfc::data::field_from_inbox<double>(
      domain, host_stack->fft().get_inbox_bounds());

  const Stiffness solid = Stiffness::isotropic(1.0, 0.3);
  const Stiffness liquid = Stiffness::isotropic(0.25, 0.3);
  auto fill_inclusion = [&]() {
    const double e0 = 2.0e-3;
    h.for_each_owned([&](int i, int j, int k) {
      const auto x = h.coords(i, j, k);
      const double r = std::sqrt((x[0] - 0.5 * N) * (x[0] - 0.5 * N) +
                                 (x[1] - 0.5 * N) * (x[1] - 0.5 * N) +
                                 (x[2] - 0.5 * N) * (x[2] - 0.5 * N));
      const double hv = 0.5 * (1.0 + std::tanh((7.0 - r) / 1.5));
      h(i, j, k) = hv;
      amp(i, j, k) = hv * e0;
      dh(i, j, k) = 0.5;
      damp(i, j, k) = 0.5 * e0;
    });
    h.note_host_write();
    amp.note_host_write();
    dh.note_host_write();
    damp.note_host_write();
  };
  fill_inclusion();

  auto run_case = [&](const char *name, const Stiffness &c_liq,
                       int expect_host_iters_max) -> int {
    MicroelasticityParams p;
    p.c_solid = solid;
    p.c_liquid = c_liq;
    p.eigenstrain_pattern = Sym3::identity();
    p.warm_start = false;
    p.tol_el = 1.0e-10;
    p.n_el_iter = 80;
    p.comm = MPI_COMM_WORLD;

    EigenstrainMicroelasticity host(domain, host_stack->fft(), p);
    DeviceEigenstrainMicroelasticity device(domain, dev_stack->fft(), p);
    const auto rh = host.solve(h, amp, &dh, &damp);
    const auto rd = device.solve(h, amp, &dh, &damp);

    double worst_eps = 0.0;
    for (int c = 0; c < kSymComponents; ++c) {
      worst_eps = std::max(
          worst_eps, field_maxdiff(host.strain()[static_cast<std::size_t>(c)],
                                   device.strain()[static_cast<std::size_t>(c)],
                                   MPI_COMM_WORLD));
    }
    const double worst_dfel =
        field_maxdiff(host.dfel_dphi(), device.dfel_dphi(), MPI_COMM_WORLD);
    const double dE =
        std::abs(host.total_elastic_energy() - device.total_elastic_energy());
    int rc = 0;
    if (rank == 0) {
      std::cout << "ELASTIC_HIP_PARITY case=" << name << " ranks=" << nproc
                << " N=" << N << " host_iters=" << rh.iterations
                << " device_iters=" << rd.iterations << " host_res=" << rh.residual
                << " device_res=" << rd.residual << " max|deps|=" << worst_eps
                << " max|ddfel|=" << worst_dfel << " |dE|=" << dE << "\n";
      if (std::abs(rh.iterations - rd.iterations) > 1) {
        std::cout << "ELASTIC_HIP_PARITY_FAIL " << name << " iteration count\n";
        rc = 1;
      }
      if (rh.iterations > expect_host_iters_max) {
        std::cout << "ELASTIC_HIP_PARITY_FAIL " << name
                  << " host iteration bound\n";
        rc = 1;
      }
      if (worst_eps > tol_strain) {
        std::cout << "ELASTIC_HIP_PARITY_FAIL " << name << " strain\n";
        rc = 1;
      }
      if (worst_dfel > tol_dfel) {
        std::cout << "ELASTIC_HIP_PARITY_FAIL " << name << " dfel\n";
        rc = 1;
      }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return rc;
  };

  int rc_cases = run_case("eshelby", solid, 1);
  rc_cases |= run_case("heterogeneous", liquid, 80);
  if (rank == 0 && rc_cases == 0) std::cout << "ELASTIC_HIP_PARITY_PASS\n";
  rc = rc_cases;
  host_stack.reset();
  dev_stack.reset();
  MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return rc;
}
