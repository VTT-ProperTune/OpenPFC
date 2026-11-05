// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file cahn_hilliard_elastic.cpp
 * @brief Coherent Vegard Cahn–Hilliard on the HIP Green solver.
 *
 * Usage: `cahn_hilliard_elastic_hip CASE.json`.
 * The elastic iteration is `DeviceEigenstrainMicroelasticity::solve_resident`.
 * CUDA is not selected by this binary.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "cahn_hilliard_elastic_hip requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <cahn_hilliard/elastic_coupled_step.hpp>
#include <cahn_hilliard/elastic_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>

int main(int argc, char *argv[]) {
  pfc::runtime::gpu::bind_local_device_before_mpi();
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  try {
    if (argc != 2) {
      throw std::invalid_argument("usage: cahn_hilliard_elastic_hip CASE.json");
    }
    std::ifstream in(argv[1]);
    if (!in) {
      throw std::runtime_error(std::string("cannot open ") + argv[1]);
    }
    const nlohmann::json cfg = nlohmann::json::parse(in);
    const auto domain = pfc::ui::from_json<pfc::Domain>(cfg);
    cahn_hilliard::CahnHilliardParams ch;
    if (cfg.contains("model") && cfg.at("model").contains("params")) {
      cahn_hilliard::apply_cahn_hilliard_json(cfg.at("model").at("params"), ch);
    } else {
      ch.recompute_derived();
    }
    const auto el = cahn_hilliard::parse_elasticity(cfg);
    const double c_ref = (el.c_ref >= 0.0) ? el.c_ref : ch.c0;
    const auto stiffness = cahn_hilliard::stiffness_in_rt_vm(el, ch);
    const auto &ts = cfg.at("timestepping");
    const double t1 = ts.at("t1").get<double>();
    const double dt = ts.at("dt").get<double>();
    if (!(dt > 0.0) || !(t1 >= 0.0)) {
      throw std::invalid_argument("timestepping: require dt>0 and t1>=0");
    }
    const int steps = static_cast<int>(std::llround(t1 / dt));

    pfc::sim::stacks::SpectralCPUStack host(domain, rank, nproc, MPI_COMM_WORLD);
    cahn_hilliard::apply_initial_condition(cfg, domain, host.u(), MPI_COMM_WORLD);
    const auto report = cahn_hilliard::advance_coherent_elastic_hip(
        domain, ch, stiffness, el.eps0, c_ref, host.u().vec(), dt, steps, rank,
        nproc, MPI_COMM_WORLD);
    if (!std::isfinite(report.mass) || !std::isfinite(report.elastic_energy) ||
        !std::isfinite(report.max_abs_mu_el) || !std::isfinite(report.residual)) {
      throw std::runtime_error("non-finite elastic Cahn-Hilliard result");
    }
    if (rank == 0) {
      const auto n = pfc::domain::get_size(domain);
      const double ntot = static_cast<double>(n[0]) * static_cast<double>(n[1]) *
                          static_cast<double>(n[2]);
      const auto sp = pfc::domain::get_spacing(domain);
      const double volume = ntot * sp[0] * sp[1] * sp[2];
      const double mean = (volume != 0.0) ? report.mass / volume : 0.0;
      std::printf("CAHN_HILLIARD_ELASTIC_HIP mean=%.17g el_energy=%.17g "
                  "max_abs_mu_el=%.17g iterations=%d residual=%.17g converged=%d\n",
                  mean, report.elastic_energy, report.max_abs_mu_el,
                  report.iterations, report.residual, report.converged ? 1 : 0);
    }
  } catch (const std::exception &e) {
    // A rank-local JSON/device exception must not leave peer ranks blocked
    // inside a later FFT or microelasticity MPI collective.
    std::cerr << "cahn_hilliard_elastic_hip rank " << rank << ": " << e.what()
              << "\n" << std::flush;
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2; // MPI_Abort is normally non-returning.
  } catch (...) {
    std::cerr << "cahn_hilliard_elastic_hip rank " << rank
              << ": unknown fatal error\n" << std::flush;
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2;
  }
  MPI_Finalize();
  return 0;
}
