// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_constitutive_point.cpp
 * @brief One material point on the device against the host J2 and FCC cores.
 *
 * The launchers are the device oracle. They are not a field solver and they
 * do not publish history. A missing device skips the test.
 */

#include "test_helpers.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/runtime/gpu/constitutive_point.hpp>

#include <cmath>
#include <string>

using pfc::finite_strain::constitutive_integrations;
using pfc::finite_strain::evaluate_fcc;
using pfc::finite_strain::evaluate_j2;
using pfc::finite_strain::fcc_identity_state;
using pfc::finite_strain::FccSlipParameters;
using pfc::finite_strain::FccStatus;
using pfc::finite_strain::identity2;
using pfc::finite_strain::identity_j2_state;
using pfc::finite_strain::J2Evaluation;
using pfc::finite_strain::J2EvaluationResult;
using pfc::finite_strain::J2Moduli;
using pfc::finite_strain::J2State;
using pfc::finite_strain::J2Status;
using pfc::finite_strain::PiolaTangent;
using pfc::finite_strain::reset_constitutive_integrations;
using pfc::finite_strain::SimoJ2Material;
using pfc::finite_strain::Tensor2;
using pfc::finite_strain::try_evaluate_fcc;
using pfc::finite_strain::try_evaluate_j2;

namespace {

constexpr double kState = 2.0e-10;
constexpr double kTangent = 2.0e-8;

J2Moduli j2_moduli() {
  J2Moduli moduli;
  moduli.bulk = 1.0;
  moduli.shear = 0.5;
  moduli.hardening = 0.2;
  moduli.initial_yield = 0.05;
  return moduli;
}

FccSlipParameters slipping_fcc() {
  FccSlipParameters parameters;
  parameters.c11 = 2.0;
  parameters.c12 = 1.0;
  parameters.c44 = 0.5;
  parameters.reference_rate = 0.001;
  parameters.rate_exponent = 5.0;
  parameters.hardening_exponent = 2.25;
  parameters.hardening_coefficient = 0.1;
  parameters.initial_resistance = 0.02;
  parameters.saturation_resistance = 0.2;
  parameters.time_step = 0.1;
  parameters.coplanar = 1.0;
  parameters.latent = 1.4;
  return parameters;
}

Tensor2 uniaxial(double stretch) {
  Tensor2 deformation = identity2();
  deformation(0, 0) = stretch;
  return deformation;
}

Tensor2 simple_shear(double gamma) {
  Tensor2 deformation = identity2();
  deformation(0, 1) = gamma;
  return deformation;
}

Tensor2 basis(int index) {
  Tensor2 direction{};
  direction(index / 3, index % 3) = 1.0;
  return direction;
}

double gap(const Tensor2 &left, const Tensor2 &right) {
  double sum = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double delta = left(i, j) - right(i, j);
      sum += delta * delta;
    }
  }
  return std::sqrt(sum);
}

double max_abs(const Tensor2 &left, const Tensor2 &right) {
  double largest = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      largest = std::max(largest, std::abs(left(i, j) - right(i, j)));
    }
  }
  return largest;
}

double max_abs(const PiolaTangent &left, const PiolaTangent &right) {
  double largest = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          largest =
              std::max(largest, std::abs(left(i, j, k, l) - right(i, j, k, l)));
        }
      }
    }
  }
  return largest;
}

bool available() {
#if defined(OPENPFC_TEST_J2_HIP)
  return pfc::gpu::test::is_hip_available();
#else
  return pfc::gpu::test::is_cuda_available();
#endif
}

J2EvaluationResult launch_j2(J2Moduli moduli, const J2State &state,
                             const Tensor2 &deformation) {
  J2EvaluationResult out;
#if defined(OPENPFC_TEST_J2_HIP)
  pfc::finite_strain::evaluate_j2_point_hip(moduli, state, deformation, &out);
#else
  pfc::finite_strain::evaluate_j2_point_cuda(moduli, state, deformation, &out);
#endif
  return out;
}

pfc::finite_strain::FccEvaluationResult
launch_fcc(const FccSlipParameters &parameters, const Tensor2 &orientation,
           const pfc::finite_strain::FccSlipState &state,
           const Tensor2 &deformation) {
  pfc::finite_strain::FccEvaluationResult out;
#if defined(OPENPFC_TEST_J2_HIP)
  pfc::finite_strain::evaluate_fcc_point_hip(parameters, orientation, state,
                                             deformation, &out);
#else
  pfc::finite_strain::evaluate_fcc_point_cuda(parameters, orientation, state,
                                              deformation, &out);
#endif
  return out;
}

void require_j2(const J2EvaluationResult &device, const J2Evaluation &host) {
  REQUIRE(device.status == J2Status::Success);
  REQUIRE(max_abs(device.value.response.piola, host.response.piola) < kState);
  REQUIRE(max_abs(device.value.response.kirchhoff, host.response.kirchhoff) <
          kState);
  REQUIRE(max_abs(device.value.response.finger, host.response.finger) < kState);
  REQUIRE(std::abs(device.value.response.plastic - host.response.plastic) < kState);
  REQUIRE(std::abs(device.value.response.equivalent - host.response.equivalent) <
          kState);
  REQUIRE(max_abs(device.value.trial.finger, host.trial.finger) < kState);
  REQUIRE(max_abs(device.value.trial.deformation, host.trial.deformation) < kState);
  REQUIRE(std::abs(device.value.trial.plastic - host.trial.plastic) < kState);
  REQUIRE(max_abs(device.value.tangent, host.tangent) < kTangent);
}

} // namespace

TEST_CASE("device J2 elastic step matches the host", "[constitutive][j2][device]") {
  if (!available()) SKIP("GPU unavailable");
  const J2Moduli moduli = j2_moduli();
  const J2State state = identity_j2_state();
  const Tensor2 deformation = uniaxial(1.01);
  const J2Evaluation host = evaluate_j2(moduli, state, deformation);
  REQUIRE(host.response.plastic == 0.0);
  reset_constitutive_integrations();
  const J2EvaluationResult device = launch_j2(moduli, state, deformation);
  REQUIRE(constitutive_integrations() == 0);
  require_j2(device, host);
}

TEST_CASE("device J2 plastic step matches the host", "[constitutive][j2][device]") {
  if (!available()) SKIP("GPU unavailable");
  const J2Moduli moduli = j2_moduli();
  const J2State state = identity_j2_state();
  const Tensor2 deformation = uniaxial(1.2);
  const J2Evaluation host = evaluate_j2(moduli, state, deformation);
  REQUIRE(host.response.plastic > 0.0);
  reset_constitutive_integrations();
  const J2EvaluationResult device = launch_j2(moduli, state, deformation);
  REQUIRE(constitutive_integrations() == 0);
  require_j2(device, host);
}

TEST_CASE("device J2 accepted yield state has both tangents",
          "[constitutive][j2][device]") {
  if (!available()) SKIP("GPU unavailable");
  const J2Moduli moduli = j2_moduli();
  const Tensor2 deformation = uniaxial(1.2);
  const J2State identity = identity_j2_state();
  SimoJ2Material material({}, 1, moduli, moduli);
  material.begin_trial();
  material.stage(0, deformation);
  material.accept();
  const J2State committed = material.history().committed(0);
  REQUIRE(committed.plastic > 0.0);

  const J2Evaluation loading = evaluate_j2(moduli, identity, deformation);
  const J2Evaluation unloading = evaluate_j2(moduli, committed, deformation);
  REQUIRE(max_abs(material.accepted_tangent(0), loading.tangent) == 0.0);
  REQUIRE(gap(unloading.tangent.action(basis(0)), loading.tangent.action(basis(0))) >
          0.1);

  reset_constitutive_integrations();
  const J2EvaluationResult device_loading = launch_j2(moduli, identity, deformation);
  const J2EvaluationResult device_unloading =
      launch_j2(moduli, committed, deformation);
  REQUIRE(constitutive_integrations() == 0);
  require_j2(device_loading, loading);
  REQUIRE(max_abs(device_loading.value.tangent, material.accepted_tangent(0)) <
          kTangent);
  require_j2(device_unloading, unloading);
  REQUIRE(gap(device_unloading.value.tangent.action(basis(0)),
              device_loading.value.tangent.action(basis(0))) > 0.1);
}

TEST_CASE("device J2 singular deformation returns status",
          "[constitutive][j2][device]") {
  if (!available()) SKIP("GPU unavailable");
  const J2Moduli moduli = j2_moduli();
  J2State singular = identity_j2_state();
  singular.deformation = Tensor2{};
  const Tensor2 deformation = uniaxial(1.01);
  const auto host = try_evaluate_j2(moduli, singular, deformation);
  REQUIRE(host.status == J2Status::Singular);
  reset_constitutive_integrations();
  const J2EvaluationResult device = launch_j2(moduli, singular, deformation);
  REQUIRE(constitutive_integrations() == 0);
  REQUIRE(device.status == host.status);
}

TEST_CASE("device FCC shear 0.15 matches the host", "[constitutive][fcc][device]") {
  if (!available()) SKIP("GPU unavailable");
  const FccSlipParameters parameters = slipping_fcc();
  const Tensor2 orientation = identity2();
  const auto state = fcc_identity_state(parameters);
  const Tensor2 deformation = simple_shear(0.15);
  const auto host = evaluate_fcc(parameters, orientation, state, deformation);
  reset_constitutive_integrations();
  const auto device = launch_fcc(parameters, orientation, state, deformation);
  REQUIRE(constitutive_integrations() == 0);
  REQUIRE(device.status == FccStatus::Success);
  REQUIRE(max_abs(device.value.update.piola, host.update.piola) < kState);
  REQUIRE(max_abs(device.value.update.state.plastic, host.update.state.plastic) <
          kState);
  REQUIRE(max_abs(device.value.update.state.velocity, host.update.state.velocity) <
          kState);
  for (int system = 0; system < 12; ++system) {
    REQUIRE(std::abs(device.value.update.state.resistance[system] -
                     host.update.state.resistance[system]) < kState);
    REQUIRE(std::abs(device.value.update.state.shear[system] -
                     host.update.state.shear[system]) < kState);
  }
  REQUIRE(max_abs(device.value.tangent, host.tangent) < kTangent);
}

TEST_CASE("device FCC shear 0.3 reports the resistance cap",
          "[constitutive][fcc][device]") {
  if (!available()) SKIP("GPU unavailable");
  const FccSlipParameters parameters = slipping_fcc();
  const Tensor2 orientation = identity2();
  const auto state = fcc_identity_state(parameters);
  const Tensor2 deformation = simple_shear(0.3);
  const auto host = try_evaluate_fcc(parameters, orientation, state, deformation);
  REQUIRE(host.status == FccStatus::ResistanceIterations);
  reset_constitutive_integrations();
  const auto device = launch_fcc(parameters, orientation, state, deformation);
  REQUIRE(constitutive_integrations() == 0);
  REQUIRE(device.status == host.status);
  std::string refusal;
  try {
    (void)evaluate_fcc(parameters, orientation, state, deformation);
    refusal = "no throw";
  } catch (const std::runtime_error &error) {
    refusal = error.what();
  }
  REQUIRE(refusal == "FCC slip resistance did not converge");
  REQUIRE(constitutive_integrations() == 1);
}

int main(int argc, char **argv) { return Catch::Session().run(argc, argv); }
