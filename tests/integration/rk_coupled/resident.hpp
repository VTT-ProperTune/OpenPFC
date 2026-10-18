// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "host.hpp"
#include <cstdio>
#include <cstring>
#include <limits>
#include <openpfc/kernel/grain/diagnostics.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#if defined(__HIPCC__) || defined(__HIP__)
#include <openpfc/runtime/hip/explicit_rk.hpp>
#include <openpfc/runtime/hip/fd_gradient_device.hpp>
#include <openpfc/runtime/hip/for_each_interior_device.hpp>
#else
#include <openpfc/runtime/cuda/explicit_rk.hpp>
#include <openpfc/runtime/cuda/fd_gradient_device.hpp>
#include <openpfc/runtime/cuda/for_each_interior_device.hpp>
#endif
namespace coupled_rk {
#if defined(__HIPCC__) || defined(__HIP__)
using MemorySpace = pfc::HIPSpace;
#else
using MemorySpace = pfc::CUDASpace;
#endif
inline double verify_resident(bool fourth, bool uniform = false, double dt = .01,
                              int steps = 1) {
  constexpr int n = 24, hw = 3;
  auto geometry = domain(n);
  auto dec = decomposition(geometry);
  pfc::data::Field<double, MemorySpace> u(
      geometry, pfc::decomposition::local_box(dec, rank()), hw);
  pfc::data::Field<double, MemorySpace> v(
      geometry, pfc::decomposition::local_box(dec, rank()), hw);
  pfc::comm::HaloExchangeOptions opts;
  opts.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<MemorySpace, double> halos({&u, &v}, dec, rank(),
                                                     MPI_COMM_WORLD, opts);
  auto gu = pfc::gpu::create<UGrads>(u, 6);
  auto gv = pfc::gpu::create<VGrads>(v, 6);
  auto eval = pfc::gpu::create_composite_device<Local>(gu, gv);
  Model model;
  model.first = .31;
  model.nonlinear = .17;
  model.time_source = .23;
  model.manufactured_uniform = uniform;
  const double t = uniform ? 0. : .27;
  auto global_initial = initial(n, uniform);
  Oracle first_gold(global_initial, n, t, dt, fourth, model);
  const Oracle *gold = &first_gold;
  int stage = 0;
  double maximum = 0;
  auto fill_owned = [&](const Pair &q) {
    auto publish = [&](auto &field, int f) {
      field.with_host_view([&](double *ptr, std::size_t count) {
        std::fill(ptr, ptr + count, std::numeric_limits<double>::quiet_NaN());
        auto extent = field.size3();
        for (int k = 0; k < extent[2]; ++k)
          for (int j = 0; j < extent[1]; ++j)
            for (int i = 0; i < extent[0]; ++i) {
              auto g = field.global(i, j, k);
              ptr[field.idx(i, j, k)] = q[f][index(g[0], g[1], g[2], n)];
            }
      });
      field.sync_to_device();
    };
    publish(u, 0);
    publish(v, 1);
  };
  eval.set_prepare([&] {
    require(stage < gold->stages, "resident extra RK stage");
    halos.exchange();
    auto check = [&](auto &field, int f) {
      field.with_host_read([&](const double *ptr, std::size_t) {
        auto extent = field.size3();
        for (int k = -hw; k < extent[2] + hw; ++k)
          for (int j = -hw; j < extent[1] + hw; ++j)
            for (int i = -hw; i < extent[0] + hw; ++i) {
              const double got = ptr[field.idx(i, j, k)];
              require(std::isfinite(got), "resident stale poisoned Full halo");
              auto g = field.global(i, j, k);
              maximum = std::max(
                  maximum,
                  std::abs(got -
                           gold->inputs[stage][f][index(g[0], g[1], g[2], n)]));
            }
      });
    };
    check(u, 0);
    check(v, 1);
    ++stage;
  });
  auto rk =
      pfc::sim::steppers::create(std::tie(u, v), eval, model, dt, tableau(fourth));
  fill_owned(global_initial);
  rk.step(t, u, v);
  fill_owned(global_initial);
  stage = 0;
  maximum = 0;
  gold = &first_gold;
  pfc::grain::Diagnostics ordinary_traffic;
  double next = 0;
  {
    pfc::grain::diagnostics::Scope observing(&ordinary_traffic);
    next = rk.step(t, u, v);
  }
  require(std::abs(next - (t + dt)) < 1e-15, "wrong accepted time");
  require(stage == gold->stages, "resident missing RK stage");
  auto compare_accepted = [&](const Pair &expected) {
    auto check = [&](auto &field, int f) {
      field.with_host_read([&](const double *ptr, std::size_t) {
        auto extent = field.size3();
        for (int k = 0; k < extent[2]; ++k)
          for (int j = 0; j < extent[1]; ++j)
            for (int i = 0; i < extent[0]; ++i) {
              auto g = field.global(i, j, k);
              const double got = ptr[field.idx(i, j, k)];
              require(std::isfinite(got), "nonfinite resident accepted output");
              maximum = std::max(
                  maximum, std::abs(got - expected[f][index(g[0], g[1], g[2], n)]));
            }
      });
    };
    check(u, 0);
    check(v, 1);
  };
  compare_accepted(first_gold.result);
  require(maximum < 2e-11,
          "observed resident RK differs from independent stage/state oracle");
  if (!uniform) {
    fill_owned(global_initial);
    stage = 0;
    std::array<std::vector<double>, 2> before;
    auto snapshot = [&](auto &field, int f) {
      field.with_host_read([&](const double *ptr, std::size_t count) {
        before[f].assign(ptr, ptr + count);
      });
    };
    snapshot(u, 0);
    snapshot(v, 1);
    const auto *address_u = u.data(), *address_v = v.data();
    pfc::grain::Diagnostics rejected_traffic, accepted_traffic;
    int final_calls = 0;
    bool rejected = false;
    try {
      pfc::grain::diagnostics::Scope observing(&rejected_traffic);
      rk.step_checked(
          t,
          [&](double proposed) {
            ++final_calls;
            require(std::abs(proposed - (t + dt)) < 1e-15,
                    "final checker received wrong time");
            require(stage == first_gold.stages, "final checker ran before stages");
            compare_accepted(first_gold.result);
            // Only one rank has the physical constraint failure. The caller
            // agrees it before throwing so every integrator restores its state.
            int admitted = rank() != 0;
            MPI_Allreduce(MPI_IN_PLACE, &admitted, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD);
            if (!admitted) throw std::runtime_error("forced final rejection");
          },
          u, v);
    } catch (const std::runtime_error &e) {
      require(std::string(e.what()) == "forced final rejection",
              "unexpected resident final-check exception");
      rejected = true;
    }
    require(rejected && final_calls == 1, "final rejection not propagated");
    auto restored = [&](auto &field, int f) {
      require(!field.residency().device_needs_refresh(),
              "final rejection made host state authoritative");
      field.with_host_read([&](const double *ptr, std::size_t count) {
        require(count == before[f].size() &&
                    std::memcmp(ptr, before[f].data(), count * sizeof(double)) == 0,
                "final rejection changed accepted padded device values");
      });
    };
    restored(u, 0);
    restored(v, 1);
    require(u.data() == address_u && v.data() == address_v,
            "final rejection replaced field storage");
    stage = 0;
    {
      pfc::grain::diagnostics::Scope observing(&accepted_traffic);
      const auto accepted_time = rk.step_checked(
          t,
          [&](double proposed) {
            ++final_calls;
            require(std::abs(proposed - (t + dt)) < 1e-15,
                    "successful checker received wrong time");
            compare_accepted(first_gold.result);
          },
          u, v);
      require(std::abs(accepted_time - (t + dt)) < 1e-15,
              "wrong time after resident retry");
    }
    require(final_calls == 2, "successful final check not called once");
    for (const auto *traffic : {&rejected_traffic, &accepted_traffic}) {
      require(
          traffic->allocations(pfc::grain::diagnostics::Space::Device).requests == 0,
          "final check allocated additional device buffers");
      const auto direction =
          static_cast<std::size_t>(pfc::grain::diagnostics::Direction::HostToDevice);
      for (std::size_t f = 0; f < pfc::grain::diagnostics::field_count; ++f)
        require(traffic->copies.bytes[direction][f] ==
                    ordinary_traffic.copies.bytes[direction][f],
                "final check added a host-to-device upload");
    }
    require(maximum < 2e-11,
            "checked resident state differs from independent oracle");
  }
  Pair global_accepted = first_gold.result;
  for (int step = 1; step < steps; ++step) {
    Oracle expected(global_accepted, n, t + step * dt, dt, fourth, model);
    gold = &expected;
    stage = 0;
    rk.step(t + step * dt, u, v);
    compare_accepted(expected.result);
    global_accepted = expected.result;
  }
  require(maximum < 2e-11,
          "resident RK differs from independent CPU stencil/stage recurrence");
  if (uniform) {
    double temporal = 0;
    auto check = [&](auto &field, int f) {
      field.with_host_read([&](const double *ptr, std::size_t) {
        auto extent = field.size3();
        for (int k = 0; k < extent[2]; ++k)
          for (int j = 0; j < extent[1]; ++j)
            for (int i = 0; i < extent[0]; ++i) {
              const double got = ptr[field.idx(i, j, k)];
              temporal = std::max(
                  temporal,
                  std::abs(got - std::exp((f == 0 ? 1. : -1.) * (t + steps * dt))));
            }
      });
    };
    check(u, 0);
    check(v, 1);
    return temporal;
  }
  return maximum;
}

// The point RHS reads a forcing field assembled by the host hook. Passing
// time only to rhs() cannot make this nonautonomous problem correct.
struct PreparedForcing {
  OPENPFC_HD Increments rhs(double t, const Local &g) const noexcept {
    return {g.u.value + t, -2. * g.v.value + t};
  }
};

inline void verify_prepared_time() {
  using namespace pfc::sim::steppers;
  constexpr double start = .37, dt = .2;
  const std::array<std::vector<double>, 4> clocks{
      std::vector<double>{0.}, {0., .5}, {0., 1.}, {0., .5, .5, 1.}};
  const std::array<std::vector<double>, 4> weights{
      std::vector<double>{1.},
      {0., 1.},
      {.5, .5},
      {1. / 6., 1. / 3., 1. / 3., 1. / 6.}};
  const std::array<ButcherTableau<double>, 4> methods{
      ButcherTableau<double>(1, {0.}, {1.}, {0.}, {}, "Euler", 1),
      make_rk2_midpoint<double>(), make_rk2_heun<double>(),
      make_rk4_classical<double>()};
  for (std::size_t method = 0; method < methods.size(); ++method) {
    auto geometry = domain(16);
    auto dec = decomposition(geometry);
    const auto box = pfc::decomposition::local_box(dec, rank());
    pfc::data::Field<double, MemorySpace> u(geometry, box, 1), v(geometry, box, 1),
        forcing(geometry, box, 1);
    auto initialize = [](auto &field, double value) {
      field.with_host_view(
          [&](double *ptr, std::size_t n) { std::fill(ptr, ptr + n, value); });
      field.sync_to_device();
    };
    initialize(u, .25);
    initialize(v, -.75);
    auto gu = pfc::gpu::create<UGrads>(forcing, 2);
    auto gv = pfc::gpu::create<VGrads>(forcing, 2);
    auto eval = pfc::gpu::create_composite_device<Local>(gu, gv);
    std::vector<double> observed;
    bool reject = false;
    auto timed = [&](double time) {
      const auto stage = observed.size();
      require(stage < clocks[method].size(), "extra timed preparation");
      observed.push_back(time);
#if defined(__HIPCC__) || defined(__HIP__)
      pfc::fill_hip_impl(forcing.data(), forcing.size(), time * time * time);
#else
      pfc::fill_cuda_impl(forcing.data(), forcing.size(), time * time * time);
#endif
      forcing.note_device_write();
      if (reject && stage == (method == 0 ? 0 : 1))
        throw std::runtime_error("forced timed preparation rejection");
    };
    eval.set_prepare(timed);
    bool missing_clock = false;
    try {
      eval.prepare();
    } catch (const std::logic_error &) {
      missing_clock = true;
    }
    require(missing_clock && observed.empty(), "timed hook invented a clock");
    int legacy_calls = 0;
    eval.set_prepare([&] { ++legacy_calls; });
    eval.prepare();
    eval.prepare(start);
    require(legacy_calls == 2 && observed.empty(), "legacy hook replacement");
    eval.set_prepare(timed);
    eval.set_prepare({});
    eval.prepare();
    eval.prepare(start);
    eval.set_prepare(timed);
    eval.set_prepare(nullptr);
    eval.prepare();
    eval.prepare(start);
    require(observed.empty() && legacy_calls == 2, "clear retained a hook");
    eval.set_prepare(std::function<void(double)>{});
    eval.prepare();
    eval.set_prepare(timed);
    PreparedForcing model;
    auto rk = create(std::tie(u, v), eval, model, dt, methods[method]);
    std::array<std::vector<double>, 2> before;
    u.with_host_read(
        [&](const double *p, std::size_t n) { before[0].assign(p, p + n); });
    v.with_host_read(
        [&](const double *p, std::size_t n) { before[1].assign(p, p + n); });
    const auto *u_address = u.data(), *v_address = v.data();
    reject = true;
    bool rejected = false;
    try {
      rk.step(start, u, v);
    } catch (const std::runtime_error &e) {
      require(std::string(e.what()) == "forced timed preparation rejection",
              "unexpected timed preparation exception");
      rejected = true;
    }
    require(rejected, "timed preparation rejection lost");
    auto restored = [&](auto &field, int f) {
      field.with_host_read([&](const double *ptr, std::size_t n) {
        require(n == before[f].size() &&
                    std::memcmp(ptr, before[f].data(), n * sizeof(double)) == 0,
                "timed preparation rejection changed padded state");
      });
      require(!field.residency().device_needs_refresh(),
              "timed preparation rollback requires host upload");
    };
    restored(u, 0);
    restored(v, 1);
    require(u.data() == u_address && v.data() == v_address,
            "timed preparation rejection replaced storage");
    reject = false;
    double expected_u = .25, expected_v = -.75;
    for (int step = 0; step < 2; ++step) {
      observed.clear();
      const double time = start + step * dt;
      const auto next = rk.step(time, u, v);
      require(std::abs(next - (time + dt)) < 1e-15, "timed accepted clock");
      require(observed.size() == clocks[method].size(), "missing timed stage");
      for (std::size_t stage = 0; stage < observed.size(); ++stage) {
        const double expected_time = time + dt * clocks[method][stage];
        require(std::abs(observed[stage] - expected_time) < 1e-15,
                "wrong preparation stage clock");
        const double force = expected_time * expected_time * expected_time;
        expected_u += dt * weights[method][stage] * (force + expected_time);
        expected_v += dt * weights[method][stage] * (-2. * force + expected_time);
      }
      auto check = [&](auto &field, double expected) {
        field.with_host_read([&](const double *ptr, std::size_t) {
          auto n = field.size3();
          for (int k = 0; k < n[2]; ++k)
            for (int j = 0; j < n[1]; ++j)
              for (int i = 0; i < n[0]; ++i) {
                double value = ptr[field.idx(i, j, k)];
                require(std::isfinite(value) && std::abs(value - expected) < 3e-14,
                        "prepared forcing changed accepted solution");
              }
        });
      };
      check(u, expected_u);
      check(v, expected_v);
    }
    if (method == 3) {
      const double end = start + 2 * dt;
      const double linear = (end * end - start * start) / 2.;
      const double cubic = (std::pow(end, 4) - std::pow(start, 4)) / 4.;
      require(std::abs(expected_u - (.25 + linear + cubic)) < 3e-14 &&
                  std::abs(expected_v - (-.75 + linear - 2 * cubic)) < 3e-14,
              "RK4 prepared polynomial differs from analytic integral");
    }
    std::cout << "resident prepared-time method=" << method
              << " clocks/forcing/rollback/retry/legacy passed\n";
  }
}

inline int run_resident() {
  verify_prepared_time();
  for (bool fourth : {false, true}) {
    std::cout << (fourth ? "RK4" : "RK2")
              << " resident independent nonlinear stage/RHS/halo/state error="
              << verify_resident(fourth) << '\n';
    double coarse = verify_resident(fourth, true, .05, 4),
           fine = verify_resident(fourth, true, .025, 8);
    require(coarse / fine > (fourth ? 12. : 3.5),
            "resident nonlinear-time temporal order");
    require(fine < (fourth ? 1e-7 : 1e-4),
            "resident nonlinear-time manufactured tolerance");
    std::cout << (fourth ? "RK4" : "RK2")
              << " resident nonlinear-time coarse=" << coarse << " fine=" << fine
              << " ratio=" << coarse / fine << '\n';
  }
  return 0;
}
} // namespace coupled_rk
