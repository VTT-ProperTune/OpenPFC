// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// Automatic single-rank FFT selection (#508). The cases call
// automatic_single_rank_fft and then the session factory, and they
// transform through the stack that factory builds.

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/kernel/fft/fft_selection.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

using nlohmann::json;

namespace {

json spectral_doc(std::array<int, 3> n) {
  return json{
      {"method", "spectral"},
      {"backend", "cpu"},
      {"domain",
       {{"size", {n[0], n[1], n[2]}},
        {"lower", {0.0, 0.0, 0.0}},
        {"upper",
         {static_cast<double>(n[0]), static_cast<double>(n[1]),
          static_cast<double>(n[2])}}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.1}, {"dt", 0.1}, {"saveat", 0.1}}}};
}

void fill(std::vector<double> &real) {
  for (std::size_t i = 0; i < real.size(); ++i)
    real[i] =
        std::sin(0.17 * static_cast<double>(i)) + 0.01 * static_cast<double>(i % 13);
}

void round_trip(pfc::fft::IHostFFT &fft, const std::vector<double> &expected) {
  std::vector<double> real = expected;
  const auto original = real;
  std::vector<std::complex<double>> spectrum(fft.size_outbox());
  fft.forward(real, spectrum);
  REQUIRE(real == original);
  const auto saved = spectrum;
  std::vector<double> restored(real.size());
  fft.backward(spectrum, restored);
  REQUIRE(spectrum == saved);
  for (std::size_t i = 0; i < restored.size(); ++i) {
    REQUIRE(std::isfinite(restored[i]));
    CHECK(std::abs(restored[i] - expected[i]) < 1e-12);
  }
}

} // namespace

TEST_CASE("Automatic FFT selection follows rank ownership and vendor",
          "[fft_selection]") {
  const std::array<std::array<int, 3>, 5> grids{{
      {32, 32, 32},
      {64, 64, 64},
      {48, 16, 8},
      {17, 13, 11},
      {1, 24, 20},
  }};
  for (const auto &n : grids) {
    INFO("grid " << n[0] << " x " << n[1] << " x " << n[2]);
    for (const auto hardware :
         {pfc::fft::FftHardware::cpu, pfc::fft::FftHardware::cuda}) {
      REQUIRE(pfc::fft::automatic_single_rank_fft(
                  {hardware, 1, true, n[0], n[1], n[2]}) ==
              pfc::fft::AutomaticFft::heffte);
    }
    REQUIRE(pfc::fft::automatic_single_rank_fft(
                {pfc::fft::FftHardware::hip, 1, true, n[0], n[1], n[2]}) ==
            pfc::fft::AutomaticFft::local);
    for (const auto hardware :
         {pfc::fft::FftHardware::cpu, pfc::fft::FftHardware::cuda,
          pfc::fft::FftHardware::hip}) {
      REQUIRE(pfc::fft::automatic_single_rank_fft(
                  {hardware, 2, true, n[0], n[1], n[2]}) ==
              pfc::fft::AutomaticFft::heffte);
      REQUIRE(pfc::fft::automatic_single_rank_fft(
                  {hardware, 1, false, n[0], n[1], n[2]}) ==
              pfc::fft::AutomaticFft::heffte);
    }
  }
  REQUIRE(pfc::fft::automatic_single_rank_fft(
              {pfc::fft::FftHardware::hip, 1, true, 0, 8, 8}) ==
          pfc::fft::AutomaticFft::heffte);
}

TEST_CASE("Omitted fft builds the policy stack and preserves the field",
          "[fft_selection]") {
  const std::array<int, 3> n{17, 13, 11};
  json doc = spectral_doc(n);
  auto session =
      pfc::ui::make_simulation_session<pfc::sim::stacks::SpectralCPUStack>(
          doc, 0, 1, MPI_COMM_SELF);
  const auto decision = pfc::fft::automatic_single_rank_fft(
      {pfc::fft::FftHardware::cpu, 1, true, n[0], n[1], n[2]});
  REQUIRE(decision == pfc::fft::AutomaticFft::heffte);
  REQUIRE(session.stack().fft_choice() == pfc::fft::Implementation::Heffte);

  std::vector<double> field(session.stack().fft().size_inbox());
  fill(field);
  round_trip(session.stack().fft(), field);

  doc["fft"] = "local";
  auto local = pfc::ui::make_simulation_session<pfc::sim::stacks::SpectralCPUStack>(
      doc, 0, 1, MPI_COMM_SELF);
  REQUIRE(local.stack().fft_choice() == pfc::fft::Implementation::Local);
  round_trip(local.stack().fft(), field);

  doc["fft"] = "heffte";
  auto heffte = pfc::ui::make_simulation_session<pfc::sim::stacks::SpectralCPUStack>(
      doc, 0, 1, MPI_COMM_SELF);
  REQUIRE(heffte.stack().fft_choice() == pfc::fft::Implementation::Heffte);

  std::vector<double> local_real = field;
  std::vector<double> heffte_real = field;
  std::vector<std::complex<double>> local_hat(local.stack().fft().size_outbox());
  std::vector<std::complex<double>> heffte_hat(heffte.stack().fft().size_outbox());
  local.stack().fft().forward(local_real, local_hat);
  heffte.stack().fft().forward(heffte_real, heffte_hat);
  REQUIRE(local_real == field);
  REQUIRE(heffte_real == field);
  for (std::size_t i = 0; i < local_hat.size(); ++i) {
    CHECK(std::abs(local_hat[i] - heffte_hat[i]) <
          1e-12 * (1.0 + std::abs(heffte_hat[i])));
  }
  std::vector<double> local_back(field.size()), heffte_back(field.size());
  local.stack().fft().backward(local_hat, local_back);
  heffte.stack().fft().backward(heffte_hat, heffte_back);
  for (std::size_t i = 0; i < field.size(); ++i) {
    CHECK(std::abs(local_back[i] - heffte_back[i]) < 1e-12);
    CHECK(std::abs(local_back[i] - field[i]) < 1e-12);
  }
}
