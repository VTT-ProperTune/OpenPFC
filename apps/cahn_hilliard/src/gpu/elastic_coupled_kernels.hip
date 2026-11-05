// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file elastic_coupled_kernels.hip
 * @brief Device coherent Cahn–Hilliard step on the installed HIP Green solver.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "elastic_coupled_kernels.hip requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <cahn_hilliard/cahn_hilliard_pointwise.hpp>
#include <cahn_hilliard/elastic_coupled_step.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>
#include <openpfc/solvers/microelasticity/microelasticity_device.hpp>

namespace {

void check_hip(hipError_t err, const char *what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("advance_coherent_elastic_hip: ") + what +
                             ": " + hipGetErrorString(err));
  }
}

__global__ void elastic_amplitude_kernel(const double *c, double *amp, double c_ref,
                                         long long n) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    amp[i] = c[i] - c_ref;
  }
}

__global__ void elastic_nonlinear_kernel(const double *c, const double *mu,
                                         double *n,
                                         cahn_hilliard::CahnHilliardPointwise pw,
                                         long long count) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    n[i] = pw.n_nl(c[i]) + mu[i];
  }
}

void launch_pair(long long n, const double *c, double *amp, double c_ref,
                 const double *mu, double *nonlinear,
                 const cahn_hilliard::CahnHilliardPointwise &pw,
                 bool with_nonlinear) {
  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((static_cast<long long>(n) + threads - 1) / threads);
  if (blocks < 1) {
    return;
  }
  elastic_amplitude_kernel<<<blocks, threads>>>(c, amp, c_ref, n);
  check_hip(hipGetLastError(), "amplitude kernel");
  if (with_nonlinear) {
    elastic_nonlinear_kernel<<<blocks, threads>>>(c, mu, nonlinear, pw, n);
    check_hip(hipGetLastError(), "nonlinear kernel");
  }
  check_hip(hipDeviceSynchronize(), "elastic kernels");
}

} // namespace

namespace cahn_hilliard {

ElasticCoupledReport advance_coherent_elastic_hip(
    const pfc::Domain &domain, const CahnHilliardParams &chemistry,
    const pfc::solvers::Stiffness &stiffness, double eps0, double c_ref,
    const std::vector<double> &initial, double dt, int steps, int rank, int nproc,
    MPI_Comm comm) {
  using Ops = pfc::sim::SpectralETDOps<pfc::HIPSpace>;
  using Device = pfc::solvers::DeviceEigenstrainMicroelasticity<pfc::HIPSpace>;
  if (!(dt > 0.0) || steps < 0 || !std::isfinite(eps0) || eps0 < 0.0 ||
      !std::isfinite(c_ref)) {
    throw std::invalid_argument(
        "advance_coherent_elastic_hip: require dt>0, steps>=0, eps0>=0");
  }

  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace> stack(domain, rank, nproc, comm);
  auto &c = stack.u();
  if (initial.size() != c.size()) {
    throw std::invalid_argument(
        "advance_coherent_elastic_hip: initial size does not match the inbox");
  }
  c.with_host_view([&](double *dst, std::size_t n) {
    if (n != initial.size()) {
      throw std::invalid_argument(
          "advance_coherent_elastic_hip: host view size does not match");
    }
    std::copy(initial.begin(), initial.end(), dst);
  });
  c.sync_to_device();

  const std::size_t n_in = stack.fft().size_inbox();
  const std::size_t n_out = stack.fft().size_outbox();
  if (n_in != c.size()) {
    throw std::invalid_argument(
        "advance_coherent_elastic_hip: inbox halo is not zero");
  }
  const auto sp = c.spacing();
  const double cell = sp[0] * sp[1] * sp[2];

  pfc::solvers::MicroelasticityParams mp;
  mp.stiffness_at_one = stiffness;
  mp.stiffness_at_zero = stiffness;
  mp.eigenstrain_pattern = pfc::solvers::Sym3{{eps0, eps0, eps0, 0.0, 0.0, 0.0}};
  mp.comm = comm;
  mp.warm_start = true;
  Device solver(domain, stack.fft(), mp);
  if (solver.n_local() != n_in) {
    throw std::runtime_error(
        "advance_coherent_elastic_hip: solver inbox does not match the field");
  }

  pfc::data::Field<double, pfc::HIPSpace> n_real(domain,
                                                 stack.fft().get_inbox_bounds(), 0);
  Ops::ComplexField c_hat(domain, stack.fft().get_outbox_bounds(), 0);
  Ops::ComplexField n_hat(domain, stack.fft().get_outbox_bounds(), 0);
  auto candidate = Ops::make_complex(n_out);
  auto exp_Ldt = Ops::make_real(n_out);
  auto n_weight = Ops::make_real(n_out);
  auto mask = Ops::make_real(n_out);
  {
    std::vector<double> L(n_out), expv(n_out), w(n_out), msk(n_out, 1.0);
    pfc::fft::kspace::for_each_kpoint(
        stack.fft().get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          const double k_lap = pfc::fft::kspace::k_laplacian_value(kx, ky, kz);
          L[i] = chemistry.M *
                 (chemistry.fpp0 * k_lap - chemistry.kappa * k_lap * k_lap);
          const double Mnl = chemistry.M * k_lap;
          const auto coeff = pfc::integrator::spectral_exp_coeffs(L[i], dt);
          expv[i] = coeff.exp_Ldt;
          w[i] = Mnl * coeff.phi1_L;
        });
    pfc::fft::kspace::fill_two_thirds_mask(
        stack.fft().get_outbox_bounds(), pfc::domain::get_size(domain),
        pfc::domain::get_spacing(domain), msk.data(), msk.size());
    Ops::upload(exp_Ldt, expv);
    Ops::upload(n_weight, w);
    Ops::upload(mask, msk);
  }

  const CahnHilliardPointwise ch_pw{.omega_nd = chemistry.omega_nd,
                                    .l1_nd = chemistry.l1_nd,
                                    .c0 = chemistry.c0,
                                    .fprime0 = chemistry.fprime0,
                                    .fpp0 = chemistry.fpp0};

  const long long n_local = static_cast<long long>(n_in);
  auto prepare = [&] {
    pfc::solvers::device_detail::me_fill(solver.stiffness_weight_device(), 1.0,
                                         n_local);
    pfc::solvers::device_detail::me_fill(solver.stiffness_weight_derivative_device(),
                                         0.0, n_local);
    pfc::solvers::device_detail::me_fill(
        solver.eigenstrain_amplitude_derivative_device(), 1.0, n_local);
    check_hip(hipGetLastError(), "fill elastic inputs");
    launch_pair(n_local, c.data(), solver.eigenstrain_amplitude_device(), c_ref,
                solver.elastic_energy_derivative_device(), n_real.data(), ch_pw,
                false);
  };

  ElasticCoupledReport report;
  {
    std::vector<double> host_c(n_in);
    c.with_host_read([&](const double *data, std::size_t n) {
      std::copy(data, data + n, host_c.begin());
    });
    report.mass0 = composition_mass(host_c.data(), n_in, cell, comm);
  }

  for (int step = 0; step < steps; ++step) {
    prepare();
    const auto step_report = solver.solve_resident(true);
    (void)step_report;
    check_hip(hipDeviceSynchronize(), "solve_resident");
    launch_pair(n_local, c.data(), solver.eigenstrain_amplitude_device(), c_ref,
                solver.elastic_energy_derivative_device(), n_real.data(), ch_pw,
                true);
    n_real.note_device_write();
    Ops::forward(stack.fft(), n_real, n_hat);
    Ops::forward(stack.fft(), c, c_hat);
    Ops::multiply(n_hat, mask, n_hat);
    Ops::combine(c_hat, n_hat, exp_Ldt, n_weight, candidate);
    Ops::swap(c_hat, candidate);
    Ops::backward(stack.fft(), c_hat, c);
    check_hip(hipDeviceSynchronize(), "spectral step");
  }

  prepare();
  const auto rep = solver.solve_resident(true);
  check_hip(hipDeviceSynchronize(), "final solve");
  double mean_p = 0.0;
  double max_mu = 0.0;
  double max_vm = 0.0;
  solver.stress_invariants(mean_p, max_mu, max_vm);
  report.elastic_energy = solver.total_elastic_energy();
  report.max_abs_mu_el = max_mu;
  report.iterations = rep.iterations;
  report.residual = rep.residual;
  report.converged = rep.converged;
  report.composition.resize(n_in);
  c.with_host_read([&](const double *data, std::size_t n) {
    std::copy(data, data + n, report.composition.begin());
  });
  report.mass = composition_mass(report.composition.data(), n_in, cell, comm);
  return report;
}

} // namespace cahn_hilliard
