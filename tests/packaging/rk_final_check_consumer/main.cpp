// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/explicit_rk_gpu.hpp>
#endif
struct Value {
  double value{};
};
struct Local {
  Value u, v;
};
struct Increment {
  double du{}, dv{};
  auto as_tuple() const { return std::tie(du, dv); }
};
struct Oscillator {
  OPENPFC_HD Increment rhs(double, const Local &q) const {
    return {q.v.value, -q.u.value};
  }
};
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  try {
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
    int count = 0;
#if defined(CHECK_CUDA)
    require(cudaGetDeviceCount(&count) == cudaSuccess,
            "actual CUDA device required");
    using Space = pfc::CUDASpace;
#else
    require(hipGetDeviceCount(&count) == hipSuccess, "actual HIP device required");
    using Space = pfc::HIPSpace;
#endif
    require(count >= size, "distinct visible device per rank required");
    pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);
#else
    using Space = pfc::HostSpace;
#endif
    using Field = pfc::data::Field<double, Space>;
    auto geometry = pfc::domain::create({8, 6, 4});
    auto dec = pfc::decomposition::create(geometry, pfc::Int3{size, 1, 1});
    for (bool fourth : {false, true}) {
      Field u(geometry, pfc::decomposition::local_box(dec, rank), 1),
          v(geometry, pfc::decomposition::local_box(dec, rank), 1);
      std::vector<double> old_u, old_v;
      auto init = [&](auto &f, double owned, double ghost, auto &saved) {
        f.with_host_view([&](double *data, std::size_t count) {
          std::fill(data, data + count, ghost);
          f.for_each_owned(
              [&](int i, int j, int k) { data[f.idx(i, j, k)] = owned; });
          saved.assign(data, data + count);
        });
        f.sync_to_device();
      };
      init(u, .3, -777., old_u);
      init(v, .9, -333., old_v);
#if defined(CHECK_CUDA) || defined(CHECK_HIP)
      auto eval = pfc::gpu::create_composite_device<Local>(
          pfc::gpu::create<Value>(u, 2), pfc::gpu::create<Value>(v, 2));
#else
      auto eval = pfc::field::create_composite<Local>(
          pfc::gradient::FDGradient<Value>(u, 2),
          pfc::gradient::FDGradient<Value>(v, 2));
#endif
      Oscillator model;
      const double dt = .1, t = .7;
      const auto tableau = fourth ? pfc::sim::steppers::make_rk4_classical<double>()
                                  : pfc::sim::steppers::make_rk2_heun<double>();
      auto rk = pfc::sim::steppers::create(std::tie(u, v), eval, model, dt, tableau);
      const auto *ptr_u = u.data(), *ptr_v = v.data();
      bool reject = true;
      int calls = 0;
      auto check = [&](double proposed) {
        ++calls;
        require(std::abs(proposed - (t + dt)) < 1e-15, "wrong proposed time");
        const double a = 1 - dt * dt / 2 + (fourth ? dt * dt * dt * dt / 24 : 0),
                     b = dt - (fourth ? dt * dt * dt / 6 : 0);
        auto matches = [&](auto &f, double expected) {
          f.with_host_read([&](const double *data, std::size_t) {
            const auto n = f.local_size();
            for (int k = 0; k < n[2]; ++k)
              for (int j = 0; j < n[1]; ++j)
                for (int i = 0; i < n[0]; ++i)
                  require(std::abs(data[f.idx(i, j, k)] - expected) < 1e-14,
                          "wrong final state");
          });
        };
        matches(u, a * .3 + b * .9);
        matches(v, a * .9 - b * .3);
        int admitted = !(reject && rank == 0);
        MPI_Allreduce(MPI_IN_PLACE, &admitted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!admitted) throw std::runtime_error("rejected candidate");
      };
      bool rejected = false;
      try {
        rk.step_checked(t, check, u, v);
      } catch (const std::runtime_error &e) {
        require(std::string(e.what()) == "rejected candidate", "unexpected error");
        rejected = true;
      }
      require(rejected && calls == 1, "final rejection not propagated");
      auto restored = [&](auto &f, const auto &saved) {
        f.with_host_read([&](const double *data, std::size_t count) {
          require(count == saved.size() &&
                      std::memcmp(data, saved.data(), count * sizeof(double)) == 0,
                  "padded rollback failed");
        });
      };
      restored(u, old_u);
      restored(v, old_v);
      require(ptr_u == u.data() && ptr_v == v.data(), "field address changed");
      reject = false;
      require(std::abs(rk.step_checked(t, check, u, v) - (t + dt)) < 1e-15,
              "wrong accepted time");
      require(calls == 2, "retry final check count");
    }
    std::cout << "installed final-state checks pass on rank " << rank << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}
