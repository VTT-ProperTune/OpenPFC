// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_constitutive_evaluation.cpp
 * @brief One-integration Piola response and the two J2 yield branches.
 *
 * Smooth states are checked with a central difference and both one-sided
 * differences. An accepted J2 yield state is checked one side at a time.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/mechanics/constitutive/crystal_plasticity.hpp>
#include <openpfc/mechanics/constitutive/saint_venant_kirchhoff.hpp>
#include <openpfc/mechanics/constitutive/simo_j2.hpp>

#include <cmath>
#include <string>

using Catch::Matchers::WithinAbs;
using pfc::finite_strain::constitutive_integrations;
using pfc::finite_strain::constitutive_response;
using pfc::finite_strain::evaluate_fcc;
using pfc::finite_strain::evaluate_j2;
using pfc::finite_strain::evaluate_saint_venant;
using pfc::finite_strain::fcc_identity_state;
using pfc::finite_strain::fcc_tangent_action;
using pfc::finite_strain::FccEvaluation;
using pfc::finite_strain::FccSlipMaterial;
using pfc::finite_strain::FccSlipParameters;
using pfc::finite_strain::FccSlipState;
using pfc::finite_strain::identity2;
using pfc::finite_strain::identity_j2_state;
using pfc::finite_strain::integrate_fcc;
using pfc::finite_strain::integrate_j2;
using pfc::finite_strain::IsotropicModuli;
using pfc::finite_strain::j2_tangent_action;
using pfc::finite_strain::J2Evaluation;
using pfc::finite_strain::J2Moduli;
using pfc::finite_strain::J2Response;
using pfc::finite_strain::J2State;
using pfc::finite_strain::reset_constitutive_integrations;
using pfc::finite_strain::SaintVenantKirchhoffMaterial;
using pfc::finite_strain::SimoJ2Material;
using pfc::finite_strain::tangent_action;
using pfc::finite_strain::Tensor2;

namespace {

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

bool same_j2(const J2State &left, const J2State &right) {
  if (left.plastic != right.plastic) {
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (left.finger(i, j) != right.finger(i, j) ||
          left.deformation(i, j) != right.deformation(i, j)) {
        return false;
      }
    }
  }
  return true;
}

bool same_fcc(const FccSlipState &left, const FccSlipState &right) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (left.plastic(i, j) != right.plastic(i, j) ||
          left.velocity(i, j) != right.velocity(i, j)) {
        return false;
      }
    }
  }
  for (int system = 0; system < 12; ++system) {
    if (left.resistance[system] != right.resistance[system] ||
        left.shear[system] != right.shear[system]) {
      return false;
    }
  }
  return true;
}

Tensor2 scaled_diff(const Tensor2 &later, const Tensor2 &earlier, double scale) {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = (later(i, j) - earlier(i, j)) * scale;
    }
  }
  return out;
}

} // namespace

TEST_CASE("one J2 evaluation supplies every tangent column", "[finite_strain][j2]") {
  const J2Moduli moduli = j2_moduli();
  const J2State committed = identity_j2_state();
  for (const double stretch : {1.01, 1.2}) {
    const Tensor2 deformation = uniaxial(stretch);
    const J2Response reference = integrate_j2(moduli, committed, deformation);
    reset_constitutive_integrations();
    const J2Evaluation evaluation = evaluate_j2(moduli, committed, deformation);
    REQUIRE(constitutive_integrations() == 1);
    for (int index = 0; index < 9; ++index) {
      (void)evaluation.tangent.action(basis(index));
    }
    REQUIRE(constitutive_integrations() == 1);
    REQUIRE(gap(evaluation.response.piola, reference.piola) == 0.0);
    for (int index = 0; index < 9; ++index) {
      const Tensor2 direction = basis(index);
      const Tensor2 from_action =
          j2_tangent_action(moduli, committed, deformation, direction);
      REQUIRE(gap(evaluation.tangent.action(direction), from_action) == 0.0);
    }
    REQUIRE(constitutive_integrations() == 10);
    REQUIRE(same_j2(committed, identity_j2_state()));
    REQUIRE(evaluation.trial.plastic == evaluation.response.plastic);
    REQUIRE(evaluation.trial.deformation(0, 0) == stretch);
  }
}

TEST_CASE("a repeated J2 evaluation reuses the committed state",
          "[finite_strain][j2]") {
  const J2Moduli moduli = j2_moduli();
  const Tensor2 deformation = uniaxial(1.2);
  SimoJ2Material material({}, 1, moduli, moduli);
  REQUIRE_FALSE(material.has_accepted_tangent(0));
  REQUIRE_THROWS_AS(material.accepted_tangent(0), std::logic_error);

  reset_constitutive_integrations();
  const J2Evaluation first = material.evaluate(0, deformation);
  const J2Evaluation second = material.evaluate(0, deformation);
  REQUIRE(constitutive_integrations() == 1);
  REQUIRE(gap(first.response.piola, second.response.piola) == 0.0);
  REQUIRE(material.stress(0, deformation)(0, 0) == first.response.piola(0, 0));
  REQUIRE(gap(material.tangent_action(0, deformation, basis(0)),
              first.tangent.action(basis(0))) == 0.0);
  REQUIRE(constitutive_integrations() == 1);
  REQUIRE(material.accumulated(0) == 0.0);

  material.begin_trial();
  material.stage(0, deformation);
  REQUIRE(material.history().trial(0).plastic == first.trial.plastic);
  REQUIRE(material.accumulated(0) == 0.0);
  material.reject();
  REQUIRE(material.accumulated(0) == 0.0);
  REQUIRE_FALSE(material.has_accepted_tangent(0));

  material.begin_trial();
  material.stage(0, deformation);
  material.accept();
  REQUIRE(material.accumulated(0) == first.trial.plastic);
  REQUIRE(material.has_accepted_tangent(0));
  const Tensor2 loading = material.accepted_tangent(0).action(basis(0));
  REQUIRE(gap(loading, material.tangent_action(0, deformation, basis(0))) == 0.0);

  const Tensor2 further = uniaxial(1.25);
  const J2Evaluation moved = material.evaluate(0, further);
  REQUIRE(gap(material.tangent_action(0, further, basis(1)),
              moved.tangent.action(basis(1))) == 0.0);
  REQUIRE(material.accumulated(0) == first.trial.plastic);

  material.begin_trial();
  material.stage(0, uniaxial(1.5));
  const double staged = material.history().trial(0).plastic;
  material.reject();
  REQUIRE(staged > material.accumulated(0));
  REQUIRE(material.accumulated(0) == first.trial.plastic);
}

TEST_CASE("an accepted J2 yield state has two tangents", "[finite_strain][j2]") {
  const J2Moduli moduli = j2_moduli();
  const Tensor2 deformation = uniaxial(1.2);
  const Tensor2 direction = basis(0);
  const double step = 1.0e-7;
  SimoJ2Material material({}, 1, moduli, moduli);
  material.begin_trial();
  material.stage(0, deformation);
  material.accept();
  const J2State committed = material.history().committed(0);
  const double yield = moduli.initial_yield + moduli.hardening * committed.plastic;
  const auto fresh_response = integrate_j2(moduli, committed, deformation);
  REQUIRE_THAT(fresh_response.equivalent - yield, WithinAbs(0.0, 1.0e-12));
  REQUIRE(committed.plastic > 0.0);

  Tensor2 forward = deformation;
  Tensor2 backward = deformation;
  forward(0, 0) += step;
  backward(0, 0) -= step;
  const Tensor2 stress = fresh_response.piola;
  const Tensor2 reload = scaled_diff(integrate_j2(moduli, committed, forward).piola,
                                     stress, 1.0 / step);
  const Tensor2 unload = scaled_diff(
      stress, integrate_j2(moduli, committed, backward).piola, 1.0 / step);
  const Tensor2 fresh = j2_tangent_action(moduli, committed, deformation, direction);
  const Tensor2 lagged = material.tangent_action(0, deformation, direction);
  const J2Evaluation evaluation = evaluate_j2(moduli, committed, deformation);
  REQUIRE(gap(evaluation.tangent.action(direction), fresh) == 0.0);
  REQUIRE(gap(material.accepted_tangent(0).action(direction), lagged) == 0.0);
  REQUIRE(gap(fresh, unload) < 1.0e-6);
  REQUIRE(gap(lagged, reload) < 1.0e-6);
  REQUIRE(gap(fresh, reload) > 0.1);
  REQUIRE(gap(lagged, unload) > 0.1);
  REQUIRE(gap(fresh, lagged) > 0.1);

  for (const double sample_stretch : {1.01, 1.2}) {
    const Tensor2 sample = uniaxial(sample_stretch);
    const J2State identity = identity_j2_state();
    const Tensor2 action =
        evaluate_j2(moduli, identity, sample).tangent.action(direction);
    Tensor2 plus = sample;
    Tensor2 minus = sample;
    plus(0, 0) += step;
    minus(0, 0) -= step;
    const Tensor2 at = integrate_j2(moduli, identity, sample).piola;
    const Tensor2 ahead = integrate_j2(moduli, identity, plus).piola;
    const Tensor2 behind = integrate_j2(moduli, identity, minus).piola;
    REQUIRE(gap(action, scaled_diff(ahead, behind, 0.5 / step)) < 1.0e-8);
    REQUIRE(gap(action, scaled_diff(ahead, at, 1.0 / step)) < 1.0e-6);
    REQUIRE(gap(action, scaled_diff(at, behind, 1.0 / step)) < 1.0e-6);
  }
}

TEST_CASE("one FCC evaluation supplies every tangent column",
          "[finite_strain][crystal]") {
  const FccSlipParameters parameters = slipping_fcc();
  const Tensor2 orientation = identity2();
  const FccSlipState committed = fcc_identity_state(parameters);
  const Tensor2 deformation = simple_shear(0.15);
  const auto reference =
      integrate_fcc(parameters, orientation, committed, deformation);
  reset_constitutive_integrations();
  const FccEvaluation evaluation =
      evaluate_fcc(parameters, orientation, committed, deformation);
  REQUIRE(constitutive_integrations() == 1);
  for (int index = 0; index < 9; ++index) {
    (void)evaluation.tangent.action(basis(index));
  }
  REQUIRE(constitutive_integrations() == 1);
  REQUIRE(gap(evaluation.update.piola, reference.piola) == 0.0);
  for (int index = 0; index < 9; ++index) {
    const Tensor2 direction = basis(index);
    const Tensor2 from_action = fcc_tangent_action(
        parameters, orientation, committed, deformation, direction);
    REQUIRE(gap(evaluation.tangent.action(direction), from_action) == 0.0);
  }
  REQUIRE(constitutive_integrations() == 10);
  REQUIRE(same_fcc(committed, fcc_identity_state(parameters)));
  REQUIRE(same_fcc(evaluation.update.state, reference.state));
}

TEST_CASE("FCC keeps the accepted tangent and a fresh linearization",
          "[finite_strain][crystal]") {
  const FccSlipParameters parameters = slipping_fcc();
  const Tensor2 orientation = identity2();
  const Tensor2 deformation = simple_shear(0.15);
  const Tensor2 direction = basis(1);
  const double step = 1.0e-6;
  const FccSlipState initial = fcc_identity_state(parameters);
  FccSlipMaterial material(1, parameters);
  REQUIRE_FALSE(material.has_accepted_tangent(0));
  material.begin_trial();
  material.stage(0, deformation);
  REQUIRE(same_fcc(material.committed_state(0), initial));
  material.reject();
  REQUIRE(same_fcc(material.committed_state(0), initial));
  REQUIRE_FALSE(material.has_accepted_tangent(0));

  material.begin_trial();
  material.stage(0, deformation);
  material.accept();
  const FccSlipState committed = material.committed_state(0);
  REQUIRE_FALSE(same_fcc(committed, initial));
  const Tensor2 pre_accept =
      fcc_tangent_action(parameters, orientation, initial, deformation, direction);
  REQUIRE(gap(material.accepted_tangent(0).action(direction), pre_accept) == 0.0);
  REQUIRE(gap(material.tangent_action(0, deformation, direction), pre_accept) ==
          0.0);

  reset_constitutive_integrations();
  const FccEvaluation fresh = material.evaluate(0, deformation);
  const FccEvaluation again = material.evaluate(0, deformation);
  REQUIRE(constitutive_integrations() == 1);
  REQUIRE(gap(fresh.update.piola, again.update.piola) == 0.0);
  REQUIRE(same_fcc(material.committed_state(0), committed));
  REQUIRE(gap(fresh.tangent.action(direction), pre_accept) > 1.0e-3);

  Tensor2 forward = deformation;
  Tensor2 backward = deformation;
  forward(0, 1) += step;
  backward(0, 1) -= step;
  const Tensor2 from_initial = scaled_diff(
      integrate_fcc(parameters, orientation, initial, forward).piola,
      integrate_fcc(parameters, orientation, initial, backward).piola, 0.5 / step);
  const Tensor2 from_committed = scaled_diff(
      integrate_fcc(parameters, orientation, committed, forward).piola,
      integrate_fcc(parameters, orientation, committed, backward).piola, 0.5 / step);
  REQUIRE(gap(pre_accept, from_initial) < 1.0e-6);
  REQUIRE(gap(fresh.tangent.action(direction), from_committed) < 1.0e-6);

  const Tensor2 small = simple_shear(1.0e-4);
  const double elastic_step = 1.0e-7;
  Tensor2 elastic_forward = small;
  Tensor2 elastic_backward = small;
  elastic_forward(0, 1) += elastic_step;
  elastic_backward(0, 1) -= elastic_step;
  const Tensor2 elastic_action =
      evaluate_fcc(parameters, orientation, initial, small)
          .tangent.action(direction);
  const Tensor2 elastic_difference = scaled_diff(
      integrate_fcc(parameters, orientation, initial, elastic_forward).piola,
      integrate_fcc(parameters, orientation, initial, elastic_backward).piola,
      0.5 / elastic_step);
  REQUIRE(gap(elastic_action, elastic_difference) < 1.0e-8);

  std::string refusal;
  try {
    const auto refused =
        integrate_fcc(parameters, orientation, initial, simple_shear(0.3));
    (void)refused;
  } catch (const std::runtime_error &error) {
    refusal = error.what();
  }
  REQUIRE(refusal == "FCC slip resistance did not converge");
  REQUIRE(same_fcc(material.committed_state(0), committed));
}

TEST_CASE("Saint-Venant evaluation stores the analytic tangent",
          "[finite_strain][constitutive]") {
  IsotropicModuli moduli;
  moduli.bulk = 1.0;
  moduli.shear = 0.5;
  const Tensor2 deformation = uniaxial(1.2);
  reset_constitutive_integrations();
  const auto evaluation = evaluate_saint_venant(deformation, moduli);
  REQUIRE(constitutive_integrations() == 0);
  REQUIRE(gap(evaluation.response.piola,
              constitutive_response(deformation, moduli).piola) == 0.0);
  Tensor2 mixed = basis(0);
  mixed(1, 2) = -0.3;
  mixed(2, 1) = 0.4;
  REQUIRE(gap(evaluation.tangent.action(mixed),
              tangent_action(deformation, mixed, moduli)) < 1.0e-12);
  SaintVenantKirchhoffMaterial material({}, 1, moduli, moduli);
  REQUIRE(gap(material.evaluate(0, deformation).response.piola,
              evaluation.response.piola) == 0.0);
  REQUIRE(constitutive_integrations() == 0);
}
