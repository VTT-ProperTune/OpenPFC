// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "../equations.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mpi.h>
#include <numbers>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <stdexcept>
#include <vector>
#if defined(OpenPFC_ENABLE_HEFFTE) && !defined(__HIP__) && !defined(__HIPCC__)
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#endif
namespace coupled {
inline void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
using Field = pfc::data::Field<double>;
inline auto domain(int n) {
  const double h = 2 * std::numbers::pi / n;
  return pfc::domain::create(pfc::GridSize({n, n, n}),
                             pfc::PhysicalOrigin({0., 0., 0.}),
                             pfc::GridSpacing({h, h, h}));
}
inline void fill(Field &u, Field &v, double t) {
  u.apply([t](double x, double y, double z) { return exact_u(t, {x, y, z}); });
  v.apply([t](double x, double y, double z) { return exact_v(t, {x, y, z}); });
}
inline void poison_ghosts(Field &f) {
  const auto n = f.local_size();
  const int hw = f.storage_halo();
  for (int k = -hw; k < n[2] + hw; ++k)
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i)
        if (i < 0 || j < 0 || k < 0 || i >= n[0] || j >= n[1] || k >= n[2])
          f(i, j, k) = std::numeric_limits<double>::quiet_NaN();
}
// Instrument invocation counts, not numerical loads or timing.
template <class Eval> struct Counted {
  Eval eval;
  std::atomic<std::size_t> *points;
  std::size_t *preparations;
  void prepare() {
    ++*preparations;
    eval.prepare();
  }
  auto operator()(int i, int j, int k) const noexcept {
    points->fetch_add(1, std::memory_order_relaxed);
    return eval(i, j, k);
  }
  std::size_t idx(int i, int j, int k) const noexcept { return eval.idx(i, j, k); }
  int imin() const noexcept { return eval.imin(); }
  int imax() const noexcept { return eval.imax(); }
  int jmin() const noexcept { return eval.jmin(); }
  int jmax() const noexcept { return eval.jmax(); }
  int kmin() const noexcept { return eval.kmin(); }
  int kmax() const noexcept { return eval.kmax(); }
};
inline double residual(Field &u, Field &v, const std::vector<double> &du,
                       const std::vector<double> &dv, double t) {
  double error = 0;
  const auto n = u.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto x = u.coords(i, j, k);
        auto expected = exact_rhs(t, {x[0], x[1], x[2]});
        auto at = u.idx(i, j, k);
        require(std::isfinite(du[at]) && std::isfinite(dv[at]), "nonfinite RHS");
        error = std::max(
            {error, std::abs(du[at] - expected.du), std::abs(dv[at] - expected.dv)});
      }
  return error;
}
// A model-owned position adapter leaves the framework RHS ABI unchanged.
struct PositionedLocal {
  Local fields;
  Position position;
};
struct ForcedModel {
  Model model;
  double rate;
  Increments rhs(double t, const PositionedLocal &g) const noexcept {
    auto inc = model.rhs(t, g.fields);
    const double source = rate * t * (g.position.x + g.position.y + g.position.z);
    return {inc.du + source, inc.dv + source};
  }
};
template <class Eval> struct Positioned {
  Eval *eval;
  const Field *field;
  void prepare() { eval->prepare(); }
  PositionedLocal operator()(int i, int j, int k) const noexcept {
    auto x = field->coords(i, j, k);
    return {(*eval)(i, j, k), {x[0], x[1], x[2]}};
  }
  std::size_t idx(int i, int j, int k) const noexcept { return eval->idx(i, j, k); }
  int imin() const noexcept { return eval->imin(); }
  int imax() const noexcept { return eval->imax(); }
  int jmin() const noexcept { return eval->jmin(); }
  int jmax() const noexcept { return eval->jmax(); }
  int kmin() const noexcept { return eval->kmin(); }
  int kmax() const noexcept { return eval->kmax(); }
};
inline void verify_model() {
  for (double t : {0., .125, .75})
    for (Position x : {Position{.2, .3, .4}, Position{1.1, -.5, .7}}) {
      auto got = Model{}.rhs(t, exact_local(t, x));
      auto expected = exact_rhs(t, x);
      require(std::abs(got.du - expected.du) < 1e-14 &&
                  std::abs(got.dv - expected.dv) < 1e-14,
              "homogeneous analytic manufactured model");
    }
}
inline double d1_symbol(double k, double h) {
  return (1.5 * std::sin(k * h) - .3 * std::sin(2 * k * h) +
          std::sin(3 * k * h) / 30) /
         h;
}
inline double d2_symbol(double k, double h) {
  return (-490 + 540 * std::cos(k * h) - 54 * std::cos(2 * k * h) +
          4 * std::cos(3 * k * h)) /
         (180 * h * h);
}
struct Verification {
  double residual{}, stage_error{};
  std::size_t sweeps{};
};
// A teaching driver, not a replacement stepper ABI. Stable fields/evaluators,
// output vectors and stage fields are allocated once before the first stage.
inline Verification verify_fd(int n = 32, void (*observe)(bool) = nullptr) {
  auto geometry = domain(n);
  auto decomp = pfc::decomposition::create(geometry, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, 3);
  auto v = pfc::data::field_from_subdomain<double>(decomp, 0, 3);
  auto su = pfc::data::field_from_subdomain<double>(decomp, 0, 3);
  auto sv = pfc::data::field_from_subdomain<double>(decomp, 0, 3);
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hu(su, decomp, 0, MPI_COMM_SELF,
                                                     opt);
  opt.exchange_base = 1;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hv(sv, decomp, 0, MPI_COMM_SELF,
                                                     opt);
  std::size_t prepares_u = 0, prepares_v = 0;
  std::atomic<std::size_t> points_u{0}, points_v{0};
  auto gu = pfc::gradient::FDGradient<UGrads>(
      su, 6, [&] { hu.exchange(); }, pfc::comm::HaloConnectivity::Full);
  auto gv = pfc::gradient::FDGradient<VGrads>(
      sv, 6, [&] { hv.exchange(); }, pfc::comm::HaloConnectivity::Full);
  auto eval = pfc::field::create_composite<Local>(
      Counted<decltype(gu)>{gu, &points_u, &prepares_u},
      Counted<decltype(gv)>{gv, &points_v, &prepares_v});
  Model model{};
  std::vector<double> du(u.size(), -999), dv(v.size(), -999), first_u(u.size()),
      first_v(v.size());
  Verification proof;
  for (double t : {0., .125, .75}) {
    fill(su, sv, t);
    poison_ghosts(su);
    poison_ghosts(sv);
    pfc::sim::for_each_interior(
        model, eval,
        std::make_tuple(du.data() + u.idx(0, 0, 0), dv.data() + v.idx(0, 0, 0)), t);
    proof.residual = std::max(proof.residual, residual(su, sv, du, dv, t));
    ++proof.sweeps;
  }
  require(proof.residual < .004, "FD6 manufactured RHS residual");
  require(prepares_u == 3 && prepares_v == 3 &&
              points_u == 3 * std::size_t(n) * n * n && points_v == points_u,
          "unexpected manufactured sweep evaluations");
  // Explicit forcing is only a context/scatter control, not the bare PDE.
  Positioned<decltype(eval)> located{&eval, &su};
  const double stage_time = 1.25, rate = .017;
  pfc::sim::for_each_interior(
      ForcedModel{model, rate}, located,
      std::make_tuple(du.data() + u.idx(0, 0, 0), dv.data() + v.idx(0, 0, 0)),
      stage_time);
  ++proof.sweeps;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        auto x = su.coords(i, j, k);
        auto at = su.idx(i, j, k);
        auto bare = model.rhs(stage_time, eval(i, j, k));
        double forcing = rate * stage_time * (x[0] + x[1] + x[2]);
        require(std::abs(du[at] - bare.du - forcing) < 1e-13 &&
                    std::abs(dv[at] - bare.dv - forcing) < 1e-13,
                "stage time/position forcing or tuple scatter");
      }
  // Golden calls above directly inspect the same prepared evaluator once more.
  points_u.store(0);
  points_v.store(0);
  prepares_u = prepares_v = 0;
  const auto early_sweeps = proof.sweeps;

  fill(u, v, 0);
  const double h = geometry.spacing[0], dt = .002;
  const auto p = model.parameters;
  const double alpha = -p.a * d1_symbol(1, h) * d1_symbol(2, h);
  const double beta = p.b * (d2_symbol(1, h) + d2_symbol(2, h) + d2_symbol(3, h)) -
                      p.d * d1_symbol(2, h) * d1_symbol(3, h);
  const double delta = -p.c * d1_symbol(1, h) * d1_symbol(3, h);
  double A = 1, B = .6;
  if (observe) observe(true);
  for (int step = 0; step < 4; ++step) {
    // Actual stage zero, not the original bound input from a factory.
    std::copy(u.data(), u.data() + u.size(), su.data());
    std::copy(v.data(), v.data() + v.size(), sv.data());
    poison_ghosts(su);
    poison_ghosts(sv);
    pfc::sim::for_each_interior(model, eval,
                                std::make_tuple(first_u.data() + u.idx(0, 0, 0),
                                                first_v.data() + v.idx(0, 0, 0)),
                                step * dt);
    ++proof.sweeps;
    const auto size = u.local_size();
    for (int k = 0; k < size[2]; ++k)
      for (int j = 0; j < size[1]; ++j)
        for (int i = 0; i < size[0]; ++i) {
          auto at = u.idx(i, j, k);
          su(i, j, k) = u(i, j, k) + dt * .5 * first_u[at];
          sv(i, j, k) = v(i, j, k) + dt * .5 * first_v[at];
        }
    poison_ghosts(su);
    poison_ghosts(sv);
    pfc::sim::for_each_interior(
        model, eval,
        std::make_tuple(du.data() + u.idx(0, 0, 0), dv.data() + v.idx(0, 0, 0)),
        (step + .5) * dt);
    ++proof.sweeps;
    const double Am = A + dt * .5 * (alpha * A + B),
                 Bm = B + dt * .5 * (beta * A + delta * B);
    A += dt * (alpha * Am + Bm);
    B += dt * (beta * Am + delta * Bm);
    for (int k = 0; k < size[2]; ++k)
      for (int j = 0; j < size[1]; ++j)
        for (int i = 0; i < size[0]; ++i) {
          auto at = u.idx(i, j, k);
          u(i, j, k) += dt * du[at];
          v(i, j, k) += dt * dv[at];
          auto x = u.coords(i, j, k);
          double mode = std::cos(x[0] + 2 * x[1] + 3 * x[2]);
          proof.stage_error =
              std::max({proof.stage_error, std::abs(u(i, j, k) - A * mode),
                        std::abs(v(i, j, k) - B * mode)});
        }
  }
  if (observe) observe(false);
  require(proof.stage_error < 1e-11, "stale stage input or incorrect scatter");
  require(prepares_u == proof.sweeps - early_sweeps &&
              prepares_v == proof.sweeps - early_sweeps,
          "incorrect prepare count");
  require(points_u == (proof.sweeps - early_sweeps) * std::size_t(n) * n * n &&
              points_v == points_u,
          "unexpected field evaluation count");
  // The canonical scatter leaves every output ghost untouched.
  for (int k = -3; k < n + 3; ++k)
    for (int j = -3; j < n + 3; ++j)
      for (int i = -3; i < n + 3; ++i)
        if (i < 0 || j < 0 || k < 0 || i >= n || j >= n || k >= n) {
          auto at = u.idx(i, j, k);
          require(du[at] == -999 && dv[at] == -999, "output halo overwritten");
        }
  return proof;
}
#if defined(OpenPFC_ENABLE_HEFFTE) && !defined(__HIP__) && !defined(__HIPCC__)
inline double verify_spectral(int n = 32) {
  auto geometry = domain(n);
  auto decomp = pfc::decomposition::create(geometry, 1);
  auto fft = pfc::fft::create(decomp);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  auto v = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  auto gu = pfc::field::create<UGrads>(u, fft);
  auto gv = pfc::field::create<VGrads>(v, fft);
  auto eval = pfc::field::create_composite<Local>(gu, gv);
  std::vector<double> du(u.size()), dv(v.size());
  double error = 0;
  for (double t : {0., .125, .75}) {
    fill(u, v, t);
    pfc::sim::for_each_interior(
        Model{}, eval,
        std::make_tuple(du.data() + u.idx(0, 0, 0), dv.data() + v.idx(0, 0, 0)), t);
    error = std::max(error, residual(u, v, du, dv, t));
  }
  require(error < 1e-10, "spectral manufactured RHS residual");
  return error;
}
#endif
} // namespace coupled
