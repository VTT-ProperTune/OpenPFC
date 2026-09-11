// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_dendrite_hip_parity.cpp
 * @brief The deliverable of the HIP port: a measured CPU/GPU and
 *        1-rank/N-rank agreement on one deterministic case.
 *
 * @details
 * ## Why the test, and not the kernels, is the point
 *
 * Four kernels that compile and produce a dendrite-shaped picture prove
 * nothing. A GPU port of a conservative scheme fails in ways that look fine:
 * a halo group left unexchanged shows up as a faint seam that the eye reads
 * as a side branch; a missing `note_device_write` shows up only when the run
 * length crosses the point where the host mirror was last pulled; a `z`
 * stencil read on an unexchanged face is exactly zero in the 2-D slab that
 * every quick check uses. The only thing that catches all three is running
 * the *same* deterministic case through both paths and subtracting.
 *
 * So this binary does two comparisons, and both of them are numbers:
 *
 *  1. **CPU against GPU in the same process.** One `alloy_dendrite::Stepper`
 *     and one `alloy_dendrite::DeviceStepper` on the same decomposition, the
 *     same initial condition copied bit-for-bit from the host fields into the
 *     device fields, the same number of steps, then `max |phi_cpu - phi_gpu|`
 *     over owned cells reduced across ranks. Same for `U` and `theta`.
 *  2. **1 rank against N ranks on the GPU.** `--dump` writes the gathered
 *     global fields in global index order; `--compare` reads that file back
 *     in a later run at a different rank count and reports the same three
 *     max-norms. Global index order is what makes the comparison meaningful:
 *     a checksum reduced in rank order would differ between decompositions
 *     for reasons that have nothing to do with correctness.
 *
 * ## What the tolerance means
 *
 * Not zero, and deliberately not claimed to be. The two paths evaluate the
 * same expressions in the same order (see `alloy_dendrite_hip_kernels.hip`),
 * but `hipcc` contracts `a*b+c` into a fused multiply-add and `g++` on the
 * host does not do so at the same places, so every stage leaks a few ULP and
 * the explicit time loop accumulates them. The right acceptance criterion is
 * therefore a drift bound checked against a real run, which is what `--tol`
 * is, and the README records the number a real run produced rather than a
 * number that looked safe. The 1-vs-N-rank comparison is a different animal:
 * both sides are the same kernels on the same hardware, so the only source of
 * difference is the halo decomposition, and the bound there is genuinely
 * tight.
 *
 * ## The case
 *
 * A tanh sphere in a supersaturated melt, no noise, no random seeding,
 * `eps4 != 0` so the anisotropy branch is exercised, `D_th > 0` and
 * `M_c != 0` so the thermal Laplacian and the thermal feedback are both live,
 * and `at_scale = 1` so the anti-trapping current -- the one term that
 * depends on `d_t phi` from the previous stage, i.e. the stage-ordering bug
 * most likely to survive a smoke test -- is on.
 *
 * Usage: `alloy_dendrite_hip_parity [--key=value ...]`, `--help` for the list.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "alloy_dendrite_hip_parity requires HIP (-DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>

#include <alloy_dendrite/cli.hpp>
#include <alloy_dendrite/device_stepper_hip.hpp>
#include <alloy_dendrite/diagnostics.hpp>
#include <alloy_dendrite/parameters.hpp>
#include <alloy_dendrite/step.hpp>

namespace {

using HostField = pfc::data::Field<double, pfc::HostSpace>;
using DevField = pfc::data::Field<double, pfc::HIPSpace>;

struct ParityConfig {
  alloy_dendrite::ModelParams model{};
  int nx = 64;
  int ny = 64;
  int nz = 64;
  double dx = 0.8;
  int fd_order = 4;
  int steps = 100;
  double dt = 0.0;
  double dt_safety = 0.2;
  double omega = 0.55;
  double seed_radius = 8.0;
  /// Acceptance bound on `max |cpu - gpu|` over `phi`, `U` and `theta`.
  /// Non-positive disables the check and the binary only reports.
  double tol = 0.0;
  /// Skip the host stepper (used when only the cross-rank dump is wanted).
  bool with_cpu = true;
  std::string dump;
  std::string compare;
  std::string run_id = "parity";
  bool quiet = false;
};

/// Max |a - b| over owned cells, reduced across `comm`.
struct DiffNorms {
  double linf{0.0};
  double scale{0.0}; ///< max |a| over the same cells, for the relative number.
};

[[nodiscard]] DiffNorms owned_diff(const HostField &a, const std::vector<double> &b,
                                   MPI_Comm comm) {
  DiffNorms d;
  const auto n = a.local_size();
  std::size_t q = 0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i, ++q) {
        d.linf = std::max(d.linf, std::fabs(a(i, j, k) - b[q]));
        d.scale = std::max(d.scale, std::fabs(a(i, j, k)));
      }
    }
  }
  double in[2] = {d.linf, d.scale};
  double out[2] = {0.0, 0.0};
  MPI_Allreduce(in, out, 2, MPI_DOUBLE, MPI_MAX, comm);
  d.linf = out[0];
  d.scale = out[1];
  return d;
}

/// Pull the owned cells of a device field into a flat host vector, leaving
/// the device copy authoritative (this is a read, not a hand-over).
void pull_owned(DevField &dev, std::vector<double> &out) {
  const auto n = dev.local_size();
  const int hw = dev.storage_halo();
  const std::size_t npx = static_cast<std::size_t>(n[0] + 2 * hw);
  const std::size_t npy = static_cast<std::size_t>(n[1] + 2 * hw);
  out.resize(static_cast<std::size_t>(n[0]) * static_cast<std::size_t>(n[1]) *
             static_cast<std::size_t>(n[2]));
  dev.with_host_view([&](const double *data, std::size_t) {
    std::size_t q = 0;
    for (int k = 0; k < n[2]; ++k) {
      for (int j = 0; j < n[1]; ++j) {
        for (int i = 0; i < n[0]; ++i, ++q) {
          out[q] = data[(static_cast<std::size_t>(i) + hw) +
                        (static_cast<std::size_t>(j) + hw) * npx +
                        (static_cast<std::size_t>(k) + hw) * npx * npy];
        }
      }
    }
  });
  dev.note_device_write();
}

/// Copy the padded buffer of a host field into a device field verbatim. The
/// two have the same box and the same halo, so this reproduces the initial
/// condition bit-for-bit -- which is what makes the later difference
/// attributable to the step and to nothing else.
void push_padded(const HostField &src, DevField &dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error("push_padded: size mismatch");
  }
  dst.with_host_view([&](double *data, std::size_t n) {
    std::copy(src.data(), src.data() + n, data);
  });
  dst.sync_to_device();
}

/**
 * @brief Gather owned cells of every rank into one global array on rank 0,
 *        written in global index order (`x` fastest).
 *
 * Global index order is the whole point: two decompositions of the same
 * domain hold the same numbers in different places, so any comparison that
 * is not re-indexed globally compares the decomposition rather than the
 * physics.
 */
void gather_global(const pfc::decomposition::Decomposition &decomp, int rank,
                   int nproc, MPI_Comm comm, const std::vector<double> &local,
                   int NX, int NY, int NZ, std::vector<double> &global_out) {
  const auto box = pfc::decomposition::local_box(decomp, rank);
  int mine[6] = {box.low[0],  box.low[1],  box.low[2],
                 box.size[0], box.size[1], box.size[2]};
  std::vector<int> boxes(static_cast<std::size_t>(6 * nproc));
  MPI_Allgather(mine, 6, MPI_INT, boxes.data(), 6, MPI_INT, comm);

  const int my_count = static_cast<int>(local.size());
  std::vector<int> counts(static_cast<std::size_t>(nproc));
  MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  std::vector<int> displs(static_cast<std::size_t>(nproc), 0);
  int total = 0;
  for (int r = 0; r < nproc; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }

  std::vector<double> flat;
  if (rank == 0) {
    flat.resize(static_cast<std::size_t>(total));
  }
  MPI_Gatherv(local.data(), my_count, MPI_DOUBLE,
              rank == 0 ? flat.data() : nullptr, counts.data(), displs.data(),
              MPI_DOUBLE, 0, comm);

  if (rank != 0) {
    global_out.clear();
    return;
  }
  global_out.assign(static_cast<std::size_t>(NX) * static_cast<std::size_t>(NY) *
                        static_cast<std::size_t>(NZ),
                    0.0);
  for (int r = 0; r < nproc; ++r) {
    const int *b = &boxes[static_cast<std::size_t>(6 * r)];
    std::size_t q = static_cast<std::size_t>(displs[static_cast<std::size_t>(r)]);
    for (int k = 0; k < b[5]; ++k) {
      for (int j = 0; j < b[4]; ++j) {
        for (int i = 0; i < b[3]; ++i, ++q) {
          const std::size_t gi = static_cast<std::size_t>(b[0] + i);
          const std::size_t gj = static_cast<std::size_t>(b[1] + j);
          const std::size_t gk = static_cast<std::size_t>(b[2] + k);
          global_out[gi + gj * static_cast<std::size_t>(NX) +
                     gk * static_cast<std::size_t>(NX) *
                         static_cast<std::size_t>(NY)] = flat[q];
        }
      }
    }
  }
}

/// Pack the owned cells of a host field into a flat vector in local order,
/// ready for @ref gather_global.
void pack_owned(const HostField &f, std::vector<double> &out) {
  const auto n = f.local_size();
  out.resize(static_cast<std::size_t>(n[0]) * static_cast<std::size_t>(n[1]) *
             static_cast<std::size_t>(n[2]));
  std::size_t q = 0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i, ++q) {
        out[q] = f(i, j, k);
      }
    }
  }
}

/// Sum in global index order, so the checksum is decomposition-independent.
[[nodiscard]] double ordered_sum(const std::vector<double> &v) {
  double s = 0.0;
  for (const double x : v) {
    s += x;
  }
  return s;
}

[[nodiscard]] double linf_diff(const std::vector<double> &a,
                               const std::vector<double> &b) {
  double m = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

void write_dump(const std::string &path, const std::vector<double> &phi,
                const std::vector<double> &u, const std::vector<double> &th) {
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os) {
    throw std::runtime_error("cannot open dump file " + path);
  }
  const std::uint64_t n = phi.size();
  os.write(reinterpret_cast<const char *>(&n), sizeof(n));
  os.write(reinterpret_cast<const char *>(phi.data()),
           static_cast<std::streamsize>(n * sizeof(double)));
  os.write(reinterpret_cast<const char *>(u.data()),
           static_cast<std::streamsize>(n * sizeof(double)));
  os.write(reinterpret_cast<const char *>(th.data()),
           static_cast<std::streamsize>(n * sizeof(double)));
}

void read_dump(const std::string &path, std::size_t expect,
               std::vector<double> &phi, std::vector<double> &u,
               std::vector<double> &th) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("cannot open reference file " + path);
  }
  std::uint64_t n = 0;
  is.read(reinterpret_cast<char *>(&n), sizeof(n));
  if (n != expect) {
    throw std::runtime_error("reference file " + path + " holds " +
                             std::to_string(n) + " cells, this run has " +
                             std::to_string(expect));
  }
  phi.resize(n);
  u.resize(n);
  th.resize(n);
  const auto bytes = static_cast<std::streamsize>(n * sizeof(double));
  is.read(reinterpret_cast<char *>(phi.data()), bytes);
  is.read(reinterpret_cast<char *>(u.data()), bytes);
  is.read(reinterpret_cast<char *>(th.data()), bytes);
}

template <int Dim>
int run_parity(const ParityConfig &cfg, int rank, int nproc, MPI_Comm comm) {
  using alloy_dendrite::DeviceStepper;
  using alloy_dendrite::ModelParams;
  using alloy_dendrite::Stepper;

  const ModelParams p = cfg.model;
  const double dt_lim = alloy_dendrite::explicit_dt_limit(p, cfg.dx, Dim);
  const double dt = (cfg.dt > 0.0) ? cfg.dt : cfg.dt_safety * dt_lim;
  if (dt > dt_lim) {
    throw std::invalid_argument("alloy_dendrite_hip_parity: dt exceeds the "
                                "explicit stability limit");
  }

  const auto domain = pfc::domain::create(
      pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));

  pfc::comm::HaloExchangeOptions opt;
  opt.directions = Stepper<Dim>::directions();
  pfc::sim::stacks::FDPaddedCPUStack stack(domain, cfg.fd_order / 2, rank, nproc,
                                           comm, opt);
  Stepper<Dim> host(stack, p, cfg.fd_order);

  // ---- deterministic initial condition, set on the host fields ----------
  const int i_seed = cfg.nx / 2;
  const int j_seed = cfg.ny / 2;
  const int k_seed = cfg.nz / 2;
  const double xc = static_cast<double>(i_seed) * cfg.dx;
  const double yc = static_cast<double>(j_seed) * cfg.dx;
  const double zc = static_cast<double>(k_seed) * cfg.dx;
  const double inv_w = 1.0 / (std::sqrt(2.0) * p.W0);
  const double u0 = -cfg.omega;
  host.phi().for_each_owned([&](int i, int j, int kk) {
    const auto c = host.phi().coords(i, j, kk);
    const double dz = (Dim == 3) ? (c[2] - zc) : 0.0;
    const double r =
        std::sqrt((c[0] - xc) * (c[0] - xc) + (c[1] - yc) * (c[1] - yc) + dz * dz);
    host.phi()(i, j, kk) = std::tanh((cfg.seed_radius * p.W0 - r) * inv_w);
    host.solute()(i, j, kk) = u0;
    host.temperature()(i, j, kk) = 0.0;
  });
  host.seed_conserved_solute();

  DeviceStepper<Dim> dev(domain, stack.decomposition(), rank, comm, p,
                         cfg.fd_order);
  push_padded(host.phi(), dev.phi());
  push_padded(host.solute(), dev.solute());
  push_padded(host.temperature(), dev.temperature());
  dev.seed_conserved_solute();

  // ---- run both -----------------------------------------------------
  MPI_Barrier(comm);
  const double t_gpu0 = MPI_Wtime();
  for (int s = 0; s < cfg.steps; ++s) {
    dev.step(dt);
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    throw std::runtime_error("hipDeviceSynchronize failed after the GPU loop");
  }
  MPI_Barrier(comm);
  const double t_gpu = MPI_Wtime() - t_gpu0;

  double t_cpu = 0.0;
  if (cfg.with_cpu) {
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    for (int s = 0; s < cfg.steps; ++s) {
      host.step(dt);
    }
    MPI_Barrier(comm);
    t_cpu = MPI_Wtime() - t0;
  }

  std::vector<double> g_phi, g_u, g_th;
  pull_owned(dev.phi(), g_phi);
  pull_owned(dev.solute(), g_u);
  pull_owned(dev.temperature(), g_th);

  DiffNorms d_phi{}, d_u{}, d_th{};
  if (cfg.with_cpu) {
    d_phi = owned_diff(host.phi(), g_phi, comm);
    d_u = owned_diff(host.solute(), g_u, comm);
    d_th = owned_diff(host.temperature(), g_th, comm);
  }

  // ---- global, decomposition-independent view ------------------------
  std::vector<double> G_phi, G_u, G_th;
  gather_global(stack.decomposition(), rank, nproc, comm, g_phi, cfg.nx, cfg.ny,
                cfg.nz, G_phi);
  gather_global(stack.decomposition(), rank, nproc, comm, g_u, cfg.nx, cfg.ny,
                cfg.nz, G_u);
  gather_global(stack.decomposition(), rank, nproc, comm, g_th, cfg.nx, cfg.ny,
                cfg.nz, G_th);

  // The same view of the host run. The pointwise L-inf above is the right
  // measure while the two solutions are still the same solution; once the
  // interface goes morphologically unstable it stops being one, because a
  // one-ULP difference at the tip is amplified by the physics rather than by
  // the port. Domain integrals do not care where the amplified difference
  // sits, so they keep saying something after the pointwise norm has stopped.
  std::vector<double> C_phi, C_u, C_th;
  if (cfg.with_cpu) {
    std::vector<double> local;
    pack_owned(host.phi(), local);
    gather_global(stack.decomposition(), rank, nproc, comm, local, cfg.nx, cfg.ny,
                  cfg.nz, C_phi);
    pack_owned(host.solute(), local);
    gather_global(stack.decomposition(), rank, nproc, comm, local, cfg.nx, cfg.ny,
                  cfg.nz, C_u);
    pack_owned(host.temperature(), local);
    gather_global(stack.decomposition(), rank, nproc, comm, local, cfg.nx, cfg.ny,
                  cfg.nz, C_th);
  }

  int rc = EXIT_SUCCESS;
  const std::size_t ncell = static_cast<std::size_t>(cfg.nx) *
                            static_cast<std::size_t>(cfg.ny) *
                            static_cast<std::size_t>(cfg.nz);

  double rank_linf[3] = {0.0, 0.0, 0.0};
  bool compared_ranks = false;
  if (rank == 0 && !cfg.compare.empty()) {
    std::vector<double> R_phi, R_u, R_th;
    read_dump(cfg.compare, ncell, R_phi, R_u, R_th);
    rank_linf[0] = linf_diff(G_phi, R_phi);
    rank_linf[1] = linf_diff(G_u, R_u);
    rank_linf[2] = linf_diff(G_th, R_th);
    compared_ranks = true;
  }
  {
    int flag = compared_ranks ? 1 : 0;
    MPI_Bcast(&flag, 1, MPI_INT, 0, comm);
    MPI_Bcast(rank_linf, 3, MPI_DOUBLE, 0, comm);
    compared_ranks = flag != 0;
  }

  if (rank == 0 && !cfg.dump.empty()) {
    write_dump(cfg.dump, G_phi, G_u, G_th);
  }

  if (rank == 0 && !cfg.quiet) {
    std::cout << std::setprecision(17);
    std::cout << "ALLOY_PARITY run_id=" << cfg.run_id << " dim=" << Dim
              << " nx=" << cfg.nx << " ny=" << cfg.ny << " nz=" << cfg.nz
              << " dx=" << cfg.dx << " fd_order=" << cfg.fd_order
              << " steps=" << cfg.steps << " dt=" << dt << " ranks=" << nproc
              << "\n";
    std::cout << "ALLOY_PARITY_SUM"
              << " phi=" << std::hexfloat << ordered_sum(G_phi) << std::defaultfloat
              << " U=" << std::hexfloat << ordered_sum(G_u) << std::defaultfloat
              << " theta=" << std::hexfloat << ordered_sum(G_th) << std::defaultfloat
              << "\n";
    if (cfg.with_cpu) {
      std::cout << "ALLOY_PARITY_CPU_GPU"
                << " linf_phi=" << d_phi.linf << " linf_U=" << d_u.linf
                << " linf_theta=" << d_th.linf << " scale_phi=" << d_phi.scale
                << " scale_U=" << d_u.scale << " scale_theta=" << d_th.scale
                << " rel_phi=" << (d_phi.scale > 0.0 ? d_phi.linf / d_phi.scale : 0.0)
                << " rel_U=" << (d_u.scale > 0.0 ? d_u.linf / d_u.scale : 0.0)
                << " rel_theta="
                << (d_th.scale > 0.0 ? d_th.linf / d_th.scale : 0.0) << "\n";
    }
    if (cfg.with_cpu && !C_phi.empty()) {
      const double sp_c = ordered_sum(C_phi);
      const double su_c = ordered_sum(C_u);
      const double st_c = ordered_sum(C_th);
      const double sp_g = ordered_sum(G_phi);
      const double su_g = ordered_sum(G_u);
      const double st_g = ordered_sum(G_th);
      const auto rel = [](double a, double b) {
        const double s = std::max(std::fabs(a), std::fabs(b));
        return (s > 0.0) ? std::fabs(a - b) / s : 0.0;
      };
      std::cout << "ALLOY_PARITY_INTEGRAL"
                << " sum_phi_cpu=" << sp_c << " sum_phi_gpu=" << sp_g
                << " rel_phi=" << rel(sp_c, sp_g) << " rel_U=" << rel(su_c, su_g)
                << " rel_theta=" << rel(st_c, st_g) << "\n";
    }
    if (compared_ranks) {
      std::cout << "ALLOY_PARITY_RANKS"
                << " ref=" << cfg.compare << " linf_phi=" << rank_linf[0]
                << " linf_U=" << rank_linf[1] << " linf_theta=" << rank_linf[2]
                << "\n";
    }
    std::cout << "ALLOY_PARITY_TIME"
              << " gpu_loop_s=" << t_gpu << " cpu_loop_s=" << t_cpu
              << " gpu_per_step_ms=" << 1.0e3 * t_gpu / std::max(1, cfg.steps)
              << " cpu_per_step_ms=" << 1.0e3 * t_cpu / std::max(1, cfg.steps)
              << " speedup=" << (t_gpu > 0.0 ? t_cpu / t_gpu : 0.0) << "\n";
  }

  if (cfg.tol > 0.0) {
    const double worst_local =
        std::max({d_phi.linf, d_u.linf, d_th.linf, rank_linf[0], rank_linf[1],
                  rank_linf[2]});
    if (!(worst_local <= cfg.tol)) {
      if (rank == 0) {
        std::cerr << "ALLOY_PARITY_FAIL worst=" << worst_local
                  << " tol=" << cfg.tol << "\n";
      }
      rc = EXIT_FAILURE;
    } else if (rank == 0 && !cfg.quiet) {
      std::cout << "ALLOY_PARITY_PASS worst=" << worst_local << " tol=" << cfg.tol
                << "\n";
    }
  }
  return rc;
}

void print_usage(std::ostream &os, const char *exe) {
  ParityConfig d;
  os << "Usage: " << exe << " [--key=value ...]\n\n"
     << "CPU/GPU and 1-rank/N-rank parity for the HIP twin of equations "
        "(1)-(4).\n\n"
     << "Grid and integration\n"
     << "  --nx=N --ny=N --nz=N   grid (" << d.nx << "," << d.ny << "," << d.nz
     << "); nz=1 runs the 2-D slab\n"
     << "  --dx=X                 spacing in W0            (" << d.dx << ")\n"
     << "  --fd-order=N           even FD order, 2..14     (" << d.fd_order << ")\n"
     << "  --steps=N              steps to take            (" << d.steps << ")\n"
     << "  --dt=X                 explicit step; 0 = auto  (" << d.dt << ")\n"
     << "  --dt-safety=X          fraction of the limit    (" << d.dt_safety
     << ")\n\n"
     << "Case\n"
     << "  --omega=X              initial supersaturation  (" << d.omega << ")\n"
     << "  --seed-radius=X        seed radius in W0        (" << d.seed_radius
     << ")\n"
     << "  --lambda --k --Dl --Dth --Mc --eps4 --at-scale --W0 --tau0\n"
     << "  --aniso-form=S      karma-rappel (spec, default) or "
        "unnormalised (PR #147)\n"
     << "  --evolve-theta=0|1     integrate equation (4)\n\n"
     << "Comparison\n"
     << "  --tol=X                fail above this L-inf    (" << d.tol
     << "; 0 = report only)\n"
     << "  --with-cpu=0|1         also run the host stepper (" << d.with_cpu
     << ")\n"
     << "  --dump=PATH            write the gathered global fields\n"
     << "  --compare=PATH         compare against a dump from another run\n"
     << "  --run-id=NAME          label in the output       (" << d.run_id << ")\n"
     << "  --quiet=1              suppress the report\n";
}

int run(int argc, char **argv, int rank, int nproc) {
  alloy_dendrite::Options opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) {
      print_usage(std::cout, argv[0]);
    }
    return EXIT_SUCCESS;
  }

  ParityConfig cfg;
  // Anisotropic, thermally coupled and anti-trapping on: every branch of the
  // four stages is live in the default case, because a parity test that only
  // exercises the isotropic isothermal path would pass with three of the four
  // kernels wrong.
  cfg.model.eps4 = 0.2;
  cfg.model.D_th = 2.0;
  cfg.model.M_c = 0.5;
  cfg.model.lambda = 1.0;

  cfg.nx = opt.integer("nx", cfg.nx);
  cfg.ny = opt.integer("ny", cfg.ny);
  cfg.nz = opt.integer("nz", cfg.nz);
  cfg.dx = opt.real("dx", cfg.dx);
  cfg.fd_order = opt.integer("fd-order", cfg.fd_order);
  cfg.steps = opt.integer("steps", cfg.steps);
  cfg.dt = opt.real("dt", cfg.dt);
  cfg.dt_safety = opt.real("dt-safety", cfg.dt_safety);
  cfg.omega = opt.real("omega", cfg.omega);
  cfg.seed_radius = opt.real("seed-radius", cfg.seed_radius);
  cfg.model.lambda = opt.real("lambda", cfg.model.lambda);
  cfg.model.k = opt.real("k", cfg.model.k);
  cfg.model.D_l = opt.real("Dl", cfg.model.D_l);
  cfg.model.D_th = opt.real("Dth", cfg.model.D_th);
  cfg.model.M_c = opt.real("Mc", cfg.model.M_c);
  cfg.model.eps4 = opt.real("eps4", cfg.model.eps4);
  cfg.model.aniso_form = alloy_dendrite::parse_anisotropy_form(opt.text(
      "aniso-form", alloy_dendrite::anisotropy_form_name(cfg.model.aniso_form)));
  cfg.model.W0 = opt.real("W0", cfg.model.W0);
  cfg.model.tau0 = opt.real("tau0", cfg.model.tau0);
  cfg.model.at_scale = opt.real("at-scale", cfg.model.at_scale);
  cfg.model.evolve_theta = opt.flag("evolve-theta", cfg.model.evolve_theta);
  cfg.tol = opt.real("tol", cfg.tol);
  cfg.with_cpu = opt.flag("with-cpu", cfg.with_cpu);
  cfg.dump = opt.text("dump", cfg.dump);
  cfg.compare = opt.text("compare", cfg.compare);
  cfg.run_id = opt.text("run-id", cfg.run_id);
  cfg.quiet = opt.flag("quiet", cfg.quiet);
  opt.require_all_consumed();

  pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);

  return (cfg.nz == 1) ? run_parity<2>(cfg, rank, nproc, MPI_COMM_WORLD)
                       : run_parity<3>(cfg, rank, nproc, MPI_COMM_WORLD);
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        try {
          return run(app_argc, app_argv, rank, nproc);
        } catch (const std::exception &e) {
          if (rank == 0) {
            std::cerr << "alloy_dendrite_hip_parity: " << e.what() << "\n";
          }
          return EXIT_FAILURE;
        }
      });
}
