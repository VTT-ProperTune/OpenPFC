// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "oracle.hpp"
#include <iostream>
#include <limits>
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#if defined(OpenPFC_ENABLE_HEFFTE) && !defined(__HIP__) && !defined(__HIPCC__) &&   \
    !defined(__CUDACC__)
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#endif
namespace coupled_rk {
using Field = pfc::data::Field<double>;
inline auto domain(int n) {
  const double h = 2 * std::numbers::pi / n;
  return pfc::domain::create(pfc::GridSize({n, n, n}),
                             pfc::PhysicalOrigin({0., 0., 0.}),
                             pfc::GridSpacing({h, h, h}));
}
inline int rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}
inline int ranks() {
  int r = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &r);
  return r;
}
inline auto decomposition(const pfc::Domain &g) {
  return pfc::decomposition::create(g, pfc::Int3{ranks(), 1, 1});
}
inline void store(Field &u, Field &v, const Pair &q, int n) {
  auto extent = u.local_size();
  for (int k = 0; k < extent[2]; ++k)
    for (int j = 0; j < extent[1]; ++j)
      for (int i = 0; i < extent[0]; ++i) {
        auto g = u.global(i, j, k);
        auto at = index(g[0], g[1], g[2], n);
        u(i, j, k) = q[0][at];
        v(i, j, k) = q[1][at];
      }
}
inline void poison(Field &f) {
  auto n = f.local_size();
  int hw = f.storage_halo();
  for (int k = -hw; k < n[2] + hw; ++k)
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i)
        if (i < 0 || j < 0 || k < 0 || i >= n[0] || j >= n[1] || k >= n[2])
          f(i, j, k) = std::numeric_limits<double>::quiet_NaN();
}
inline double error(const Field &u, const Field &v, const Pair &q, int n) {
  double e = 0;
  auto extent = u.local_size();
  for (int k = 0; k < extent[2]; ++k)
    for (int j = 0; j < extent[1]; ++j)
      for (int i = 0; i < extent[0]; ++i) {
        auto g = u.global(i, j, k);
        auto at = index(g[0], g[1], g[2], n);
        require(std::isfinite(u(i, j, k)) && std::isfinite(v(i, j, k)),
                "nonfinite accepted output");
        e = std::max(
            {e, std::abs(u(i, j, k) - q[0][at]), std::abs(v(i, j, k) - q[1][at])});
      }
  return e;
}
struct Audit {
  const Oracle *expected{};
  Field *u{}, *v{};
  int n{}, stage{};
  double stage_error{}, time_error{}, halo_error{}, rhs_error{};
  const Model *model{};
  std::size_t points{}, preparations{};
};
// Same evaluator protocol; validation observes stages, not an alternate RHS ABI.
template <class Eval> struct CheckedEval {
  Eval eval;
  Audit *audit;
  void prepare() {
    auto &a = *audit;
    require(a.stage < a.expected->stages, "extra RK stage");
    a.stage_error =
        std::max(a.stage_error, error(*a.u, *a.v, a.expected->inputs[a.stage], a.n));
    require(a.stage_error < 2e-11,
            "RK factory evaluated original/stale stage fields");
    poison(*a.u);
    poison(*a.v);
    eval.prepare();
    ++a.preparations;
    const int hw = a.u->storage_halo();
    auto extent = a.u->local_size();
    for (int k = -hw; k < extent[2] + hw; ++k)
      for (int j = -hw; j < extent[1] + hw; ++j)
        for (int i = -hw; i < extent[0] + hw; ++i) {
          auto g = a.u->global(i, j, k);
          const auto at = index(g[0], g[1], g[2], a.n);
          for (int f = 0; f < 2; ++f) {
            const double got = (f == 0 ? *a.u : *a.v)(i, j, k);
            require(std::isfinite(got), "stage Full halo remained poisoned");
            a.halo_error = std::max(
                a.halo_error, std::abs(got - a.expected->inputs[a.stage][f][at]));
          }
        }
    require(a.halo_error < 2e-11, "stage halo content belongs to another RK stage");
    ++a.stage;
  }
  auto operator()(int i, int j, int k) const noexcept {
    ++audit->points;
    auto g = eval(i, j, k);
    auto &a = *audit;
    auto x = a.u->global(i, j, k);
    auto at = index(x[0], x[1], x[2], a.n);
    auto got = a.model->rhs(a.expected->times[a.stage - 1], g);
    a.rhs_error = std::max(
        {a.rhs_error, std::abs(got.du - a.expected->slopes[a.stage - 1][0][at]),
         std::abs(got.dv - a.expected->slopes[a.stage - 1][1][at])});
    return g;
  }
  auto idx(int i, int j, int k) const noexcept { return eval.idx(i, j, k); }
  int imin() const { return eval.imin(); }
  int imax() const { return eval.imax(); }
  int jmin() const { return eval.jmin(); }
  int jmax() const { return eval.jmax(); }
  int kmin() const { return eval.kmin(); }
  int kmax() const { return eval.kmax(); }
};
struct CheckedModel {
  Model model;
  Audit *audit;
  Increments rhs(double t, const Local &g) const noexcept {
    audit->time_error = std::max(
        audit->time_error, std::abs(t - audit->expected->times[audit->stage - 1]));
    return model.rhs(t, g);
  }
};
inline auto tableau(bool fourth) {
  return fourth ? pfc::sim::steppers::make_rk4_classical<double>()
                : pfc::sim::steppers::make_rk2_midpoint<double>();
}
using Observe = void (*)(bool);
inline double verify_fd_stages(bool fourth, Observe observe = nullptr,
                               bool harmonic = false) {
  constexpr int n = 24;
  auto g = domain(n);
  auto dec = decomposition(g);
  auto u = pfc::data::field_from_subdomain<double>(dec, rank(), 3), v = u;
  Model model;
  model.first = .31;
  model.nonlinear = .17;
  model.time_source = .23;
  if (harmonic) model.first = model.nonlinear = model.time_source = 0;
  auto accepted = initial(n);
  const double dt = .01, t = .27;
  Oracle gold(accepted, n, t, dt, fourth, model);
  Audit audit{&gold, &u, &v, n};
  audit.model = &model;
  pfc::comm::HaloExchangeOptions opts;
  opts.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hu(u, dec, rank(), MPI_COMM_WORLD,
                                                     opts);
  opts.exchange_base = 1;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hv(v, dec, rank(), MPI_COMM_WORLD,
                                                     opts);
  auto gu = pfc::gradient::FDGradient<UGrads>(
      u, 6, [&] { hu.exchange(); }, opts.connectivity);
  auto gv = pfc::gradient::FDGradient<VGrads>(
      v, 6, [&] { hv.exchange(); }, opts.connectivity);
  auto eval = CheckedEval{pfc::field::create_composite<Local>(gu, gv), &audit};
  CheckedModel checked{model, &audit};
  auto rk =
      pfc::sim::steppers::create(std::tie(u, v), eval, checked, dt, tableau(fourth));
  store(u, v, accepted, n);
  rk.step(t, u, v); // warm MPI/stepper before observation
  store(u, v, accepted, n);
  audit.stage = 0;
  audit.points = 0;
  audit.preparations = 0;
  if (observe) observe(true);
  const double next = rk.step(t, u, v);
  if (observe) observe(false);
  require(std::abs(next - (t + dt)) < 1e-15, "wrong accepted time");
  require(audit.stage == gold.stages &&
              audit.preparations == std::size_t(gold.stages),
          "wrong stage count");
  require(audit.points == std::size_t(gold.stages) * u.local_size()[0] *
                              u.local_size()[1] * u.local_size()[2],
          "not exactly one composite evaluation per cell/stage");
  require(audit.time_error < 1e-15, "explicit RK stage time lost");
  const double e = error(u, v, gold.result, n);
  require(e < 2e-11, "nonlinear coupled RK independent stencil recurrence");
  require(audit.rhs_error < 2e-11,
          "actual stage RHS differs from independent stencil oracle");
  return std::max({e, audit.stage_error, audit.halo_error, audit.rhs_error});
}
inline double spatial_error(int n) {
  Model m;
  auto geometry = domain(n);
  auto dec = decomposition(geometry);
  auto u = pfc::data::field_from_subdomain<double>(dec, rank(), 3), v = u;
  store(u, v, initial(n), n);
  poison(u);
  poison(v);
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hu(u, dec, rank(), MPI_COMM_WORLD,
                                                     opt);
  opt.exchange_base = 1;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hv(v, dec, rank(), MPI_COMM_WORLD,
                                                     opt);
  auto gu = pfc::gradient::FDGradient<UGrads>(
      u, 6, [&] { hu.exchange(); }, opt.connectivity);
  auto gv = pfc::gradient::FDGradient<VGrads>(
      v, 6, [&] { hv.exchange(); }, opt.connectivity);
  auto eval = pfc::field::create_composite<Local>(gu, gv);
  eval.prepare();
  double e = 0;
  auto extent = u.local_size();
  for (int k = 0; k < extent[2]; ++k)
    for (int j = 0; j < extent[1]; ++j)
      for (int i = 0; i < extent[0]; ++i) {
        auto g = u.global(i, j, k);
        double phase = 2 * std::numbers::pi * (g[0] + 2 * g[1] + 3 * g[2]) / n;
        double a = std::cos(phase), b = .6 * a;
        double du = b - 2 * m.a * a - m.first * std::sin(phase) +
                    m.nonlinear * a * a + m.time_source * .27;
        double dv = (-14 * m.b - 6 * m.d) * a - 3 * m.c * b - m.nonlinear * a * b +
                    m.time_source * std::sin(.27);
        auto got = m.rhs(.27, eval(i, j, k));
        e = std::max({e, std::abs(du - got.du), std::abs(dv - got.dv)});
      }
  double global = 0;
  MPI_Allreduce(&e, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return global;
}
// Uses the production Field factory, not a hand-written stage loop.
template <class Eval>
double uniform_time(Field &u, Field &v, Eval &eval, bool fourth, double dt) {
  Model m;
  m.first = .31;
  m.nonlinear = .17;
  m.time_source = .23;
  m.manufactured_uniform = true;
  auto rk = pfc::sim::steppers::create(std::tie(u, v), eval, m, dt, tableau(fourth));
  store(u, v, initial(u.global_size()[0], true), u.global_size()[0]);
  const int steps = int(std::lround(.2 / dt));
  double t = 0;
  for (int s = 0; s < steps; ++s) t = rk.step(t, u, v);
  double e = 0;
  for (int k = 0; k < u.local_size()[2]; ++k)
    for (int j = 0; j < u.local_size()[1]; ++j)
      for (int i = 0; i < u.local_size()[0]; ++i)
        e = std::max({e, std::abs(u(i, j, k) - std::exp(t)),
                      std::abs(v(i, j, k) - std::exp(-t))});
  require(std::isfinite(e), "nonfinite manufactured temporal error");
  return e;
}
inline void verify_fd_time(bool fourth) {
  constexpr int n = 24;
  auto g = domain(n);
  auto dec = decomposition(g);
  auto u = pfc::data::field_from_subdomain<double>(dec, rank(), 3), v = u;
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hu(u, dec, rank(), MPI_COMM_WORLD,
                                                     opt);
  opt.exchange_base = 1;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hv(v, dec, rank(), MPI_COMM_WORLD,
                                                     opt);
  auto gu = pfc::gradient::FDGradient<UGrads>(
      u, 6, [&] { hu.exchange(); }, opt.connectivity);
  auto gv = pfc::gradient::FDGradient<VGrads>(
      v, 6, [&] { hv.exchange(); }, opt.connectivity);
  auto eval = pfc::field::create_composite<Local>(gu, gv);
  const double coarse = uniform_time(u, v, eval, fourth, .05),
               fine = uniform_time(u, v, eval, fourth, .025);
  std::cout << (fourth ? "RK4" : "RK2") << " FD6 nonlinear-time coarse=" << coarse
            << " fine=" << fine << " ratio=" << coarse / fine << '\n';
  require(coarse / fine > (fourth ? 12. : 3.5),
          "nonlinear explicit-time RK temporal order");
  require(fine < (fourth ? 1e-7 : 1e-4), "manufactured temporal tolerance");
}
#if defined(OpenPFC_ENABLE_HEFFTE) && !defined(__HIP__) && !defined(__HIPCC__) &&   \
    !defined(__CUDACC__)
inline double verify_spectral(bool fourth, Observe observe = nullptr) {
  constexpr int n = 24;
  auto g = domain(n);
  auto dec = decomposition(g);
  auto fft = pfc::fft::create(dec);
  auto u = pfc::data::field_from_subdomain<double>(dec, rank(), 0), v = u;
  auto gu = pfc::field::create<UGrads>(u, fft);
  auto gv = pfc::field::create<VGrads>(v, fft);
  auto base = pfc::field::create_composite<Local>(gu, gv);
  store(u, v, initial(n), n);
  base.prepare();
  double residual = 0;
  auto extent = u.local_size();
  for (int k = 0; k < extent[2]; ++k)
    for (int j = 0; j < extent[1]; ++j)
      for (int i = 0; i < extent[0]; ++i) {
        auto q = base(i, j, k);
        auto got = Model{}.rhs(0, q);
        auto g = u.global(i, j, k);
        const double mode =
            std::cos(2 * std::numbers::pi * (g[0] + 2 * g[1] + 3 * g[2]) / n);
        residual = std::max({residual, std::abs(got.du), std::abs(got.dv + mode)});
      }
  require(residual < 2e-11, "matched bare spectral manufactured spatial RHS");
  std::cout << "matched bare spectral spatial residual=" << residual << '\n';
  Model model;
  auto accepted = initial(n);
  const double t = .27, dt = .01;
  Oracle gold(accepted, n, t, dt, fourth, model, true);
  Audit audit{&gold, &u, &v, n};
  audit.model = &model;
  auto eval = CheckedEval{base, &audit};
  CheckedModel checked{model, &audit};
  auto rk =
      pfc::sim::steppers::create(std::tie(u, v), eval, checked, dt, tableau(fourth));
  store(u, v, accepted, n);
  rk.step(t, u, v);
  store(u, v, accepted, n);
  audit.stage = 0;
  if (observe) observe(true);
  rk.step(t, u, v);
  if (observe) observe(false);
  const double e = error(u, v, gold.result, n);
  require(e < 2e-11, "spectral RK exact Fourier stage oracle");
  require(audit.rhs_error < 2e-11,
          "spectral stage RHS differs from independent Fourier oracle");
  require(audit.stage == gold.stages && audit.time_error < 1e-15,
          "spectral stages/time");
  const double coarse = uniform_time(u, v, base, fourth, .05),
               fine = uniform_time(u, v, base, fourth, .025);
  std::cout << (fourth ? "RK4" : "RK2")
            << " spectral nonlinear-time coarse=" << coarse << " fine=" << fine
            << " ratio=" << coarse / fine << '\n';
  require(coarse / fine > (fourth ? 12. : 3.5), "spectral nonlinear-time RK order");
  require(fine < (fourth ? 1e-7 : 1e-4), "spectral nonlinear-time tolerance");
  return e;
}
#endif
} // namespace coupled_rk
