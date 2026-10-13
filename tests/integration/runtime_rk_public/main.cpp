// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <cstring>
#include <iostream>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/steppers/runtime_field_rk.hpp>
#include <stdexcept>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int failure = 0;
  try {
    using namespace pfc::sim::steppers;
    using Field = RuntimeFieldRKStepper::Field;
    const auto domain = pfc::domain::create({16, 2, 2});
    const pfc::decomposition::Decomposition decomposition(domain, {ranks, 1, 1});
    std::vector<Field> fields;
    fields.reserve(17);
    for (int f = 0; f < 17; ++f)
      fields.push_back(
          pfc::data::field_from_subdomain<double>(decomposition, rank, 1));
    std::vector<Field *> refs;
    std::vector<std::vector<double>> initial;
    std::vector<std::unique_ptr<pfc::comm::HaloExchange<pfc::HostSpace, double>>>
        halos;
    for (auto &field : fields) {
      std::fill(field.vec().begin(), field.vec().end(), -99.);
      const auto n = field.local_size();
      for (int k = 0; k < n[2]; ++k)
        for (int j = 0; j < n[1]; ++j)
          for (int i = 0; i < n[0]; ++i) field(i, j, k) = 2.;
      refs.push_back(&field);
      initial.push_back(field.vec());
      halos.push_back(
          std::make_unique<pfc::comm::HaloExchange<pfc::HostSpace, double>>(
              field, decomposition, rank, MPI_COMM_WORLD));
    }
    auto require = [](bool valid) {
      if (!valid) throw std::runtime_error("installed runtime RK contract failed");
    };
    bool reject = true;
    auto rhs = [&](double time, RuntimeFieldRKStepper::Rates rates) {
      for (auto &halo : halos) halo->exchange();
      if (reject && rank == 0) {
        fields.back()(0, 0, 0) = 999.;
        throw std::runtime_error("rank-zero rejection after all communication");
      }
      for (std::size_t f = 0; f < rates.size(); ++f)
        std::fill(rates[f].begin(), rates[f].end(),
                  (double(f) - 8.) * time * time * time);
    };
    auto rk =
        create_runtime(refs, 0.2, make_rk4_classical<double>(), rhs, MPI_COMM_WORLD);
    require(rk.field_count() == 17 && rk.workspace_bytes() > 0);
    bool caught = false;
    try {
      const auto failed = rk.attempt(0.7);
      (void)failed;
    } catch (const std::runtime_error &) {
      caught = true;
    }
    require(caught);
    for (std::size_t f = 0; f < fields.size(); ++f)
      require(std::memcmp(fields[f].data(), initial[f].data(),
                          initial[f].size() * sizeof(double)) == 0);
    reject = false;
    const auto trial = rk.attempt(0.7);
    for (std::size_t f = 0; f < fields.size(); ++f) {
      require(std::memcmp(fields[f].data(), initial[f].data(),
                          initial[f].size() * sizeof(double)) == 0);
      const double exact =
          2. + (double(f) - 8.) * (std::pow(0.9, 4) - std::pow(0.7, 4)) / 4.;
      for (double value : rk.candidate_field(trial, f))
        require(std::abs(value - exact) < 2e-14);
    }
    rk.commit(trial);
    double inventory = 0;
    for (const auto &field : fields) {
      const auto n = field.local_size();
      for (int k = 0; k < n[2]; ++k)
        for (int j = 0; j < n[1]; ++j)
          for (int i = 0; i < n[0]; ++i) inventory += field(i, j, k);
      require(field(-1, 0, 0) == -99.);
    }
    double total = 0;
    MPI_Allreduce(&inventory, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    require(std::abs(total - 2. * 17 * 16 * 2 * 2) < 1e-10);
    caught = false;
    try {
      rk.commit(trial);
    } catch (const std::invalid_argument &) {
      caught = true;
    }
    require(caught);
    if (rank == 0)
      std::cout << "17-field installed consumer PASS on " << ranks << " ranks\n";
  } catch (const std::exception &error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    failure = 1;
  }
  int result = 0;
  MPI_Allreduce(&failure, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return result;
}
