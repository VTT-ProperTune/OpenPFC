// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../equations.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#if defined(OpenPFC_ENABLE_HEFFTE)
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#endif

// The two evaluators differ; composition, equations and tuple scatter do not.
template <class U, class V>
void write_rates(U &gu, V &gv, pfc::data::Field<double> &u,
                 pfc::data::Field<double> &v, double time) {
  // start snippet scatter
  auto evaluator = pfc::field::create_composite<coupled::Local>(gu, gv);
  std::vector<double> du(u.size()), dv(v.size());
  pfc::sim::for_each_interior(
      coupled::Model{}, evaluator,
      std::make_tuple(du.data() + u.idx(0, 0, 0), dv.data() + v.idx(0, 0, 0)), time);
  // end snippet scatter

  std::filesystem::create_directories("results");
  std::ofstream file("results/rates.csv");
  if (!file) throw std::runtime_error("Cannot write coupled rates");
  file << std::setprecision(17) << "i,j,k,x,y,z,u,v,du,dv,exact_du,exact_dv\n";
  double error = 0;
  u.for_each_owned([&](int i, int j, int k) {
    const auto x = u.coords(i, j, k);
    const auto at = u.idx(i, j, k);
    const auto exact = coupled::exact_rhs(time, {x[0], x[1], x[2]});
    if (!std::isfinite(du[at]) || !std::isfinite(dv[at]))
      throw std::runtime_error("Nonfinite coupled rate");
    error =
        std::max({error, std::abs(du[at] - exact.du), std::abs(dv[at] - exact.dv)});
    file << i << ',' << j << ',' << k << ',' << x[0] << ',' << x[1] << ',' << x[2]
         << ',' << u(i, j, k) << ',' << v(i, j, k) << ',' << du[at] << ',' << dv[at]
         << ',' << exact.du << ',' << exact.dv << '\n';
  });
  if (!file) throw std::runtime_error("Cannot finish coupled rates");
  std::cout << "Snapshot t=" << time << " maximum RHS error=" << error << '\n';
}

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    if (comm.size() != 1)
      throw std::invalid_argument("This lesson uses one MPI process");
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 24);
    const double time = input.value("time", 0.75);
    const auto method = input.value("method", std::string("fd"));
    if (n < 12 || n > 64 || !std::isfinite(time) || std::abs(time) > 10 ||
        (method != "fd" && method != "spectral"))
      throw std::invalid_argument(
          "Use n in [12,64], finite |time| <= 10, method fd or spectral");
    const double h = 2 * std::numbers::pi / n;
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, n, n}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{h, h, h}});
    const auto decomposition = pfc::decomposition::create(domain, 1);
    const int halo = method == "fd" ? 3 : 0;
    auto u = pfc::data::field_from_subdomain<double>(decomposition, 0, halo);
    auto v = pfc::data::field_from_subdomain<double>(decomposition, 0, halo);
    u.apply([&](double x, double y, double z) {
      return coupled::exact_u(time, {x, y, z});
    });
    v.apply([&](double x, double y, double z) {
      return coupled::exact_v(time, {x, y, z});
    });

    if (method == "fd") {
      // start snippet differences
      pfc::comm::HaloExchangeOptions options;
      options.connectivity = pfc::comm::HaloConnectivity::Full;
      pfc::comm::HaloExchange<pfc::HostSpace, double> hu(u, decomposition, 0,
                                                         MPI_COMM_SELF, options);
      options.exchange_base = 1;
      pfc::comm::HaloExchange<pfc::HostSpace, double> hv(v, decomposition, 0,
                                                         MPI_COMM_SELF, options);
      auto gu = pfc::gradient::FDGradient<coupled::UGrads>(
          u, 6, [&] { hu.exchange(); }, options.connectivity);
      auto gv = pfc::gradient::FDGradient<coupled::VGrads>(
          v, 6, [&] { hv.exchange(); }, options.connectivity);
      // end snippet differences
      write_rates(gu, gv, u, v, time);
    } else {
#if defined(OpenPFC_ENABLE_HEFFTE)
      // start snippet spectral
      auto fft = pfc::fft::create(decomposition);
      auto gu = pfc::field::create<coupled::UGrads>(u, fft);
      auto gv = pfc::field::create<coupled::VGrads>(v, fft);
      write_rates(gu, gv, u, v, time);
      // end snippet spectral
#else
      throw std::invalid_argument("Spectral gradients require a HeFFTe SDK");
#endif
    }
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    return 1;
  }
}
