// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_INPUT_HPP
#define TUNGSTEN_INPUT_HPP

#include "tungsten_model.hpp"
#include "tungsten_params.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @brief Validate model configuration from json file.
 *
 * @param j json file
 */
void validate(const json &j) {
  if (!j.contains("n0") || !j.at("n0").is_number()) {
    throw std::runtime_error("Missing or invalid n0");
  }
  if (!j.contains("n_sol") || !j.at("n_sol").is_number()) {
    throw std::runtime_error("Missing or invalid n_sol");
  }
  if (!j.contains("n_vap") || !j.at("n_vap").is_number()) {
    throw std::runtime_error("Missing or invalid n_vap");
  }
  if (!j.contains("T") || !j.at("T").is_number()) {
    throw std::runtime_error("Missing or invalid T");
  }
  if (!j.contains("T0") || !j.at("T0").is_number()) {
    throw std::runtime_error("Missing or invalid T0");
  }
  if (!j.contains("Bx") || !j.at("Bx").is_number()) {
    throw std::runtime_error("Missing or invalid Bx");
  }
  if (!j.contains("alpha") || !j.at("alpha").is_number()) {
    throw std::runtime_error("Missing or invalid alpha");
  }
  if (!j.contains("alpha_farTol") || !j.at("alpha_farTol").is_number()) {
    throw std::runtime_error("Missing or invalid alpha_farTol");
  }
  if (!j.contains("alpha_highOrd") || !j.at("alpha_highOrd").is_number()) {
    throw std::runtime_error("Missing or invalid alpha_highOrd");
  }
  if (!j.contains("lambda") || !j.at("lambda").is_number()) {
    throw std::runtime_error("Missing or invalid lambda");
  }
  if (!j.contains("stabP") || !j.at("stabP").is_number()) {
    throw std::runtime_error("Missing or invalid stabP");
  }
  if (!j.contains("shift_u") || !j.at("shift_u").is_number()) {
    throw std::runtime_error("Missing or invalid shift_u");
  }
  if (!j.contains("shift_s") || !j.at("shift_s").is_number()) {
    throw std::runtime_error("Missing or invalid shift_s");
  }
  if (!j.contains("p2") || !j.at("p2").is_number()) {
    throw std::runtime_error("Missing or invalid p2");
  }
  if (!j.contains("p3") || !j.at("p3").is_number()) {
    throw std::runtime_error("Missing or invalid p3");
  }
  if (!j.contains("p4") || !j.at("p4").is_number()) {
    throw std::runtime_error("Missing or invalid p4");
  }
  if (!j.contains("q20") || !j.at("q20").is_number()) {
    throw std::runtime_error("Missing or invalid q20");
  }
  if (!j.contains("q21") || !j.at("q21").is_number()) {
    throw std::runtime_error("Missing or invalid q21");
  }
  if (!j.contains("q30") || !j.at("q30").is_number()) {
    throw std::runtime_error("Missing or invalid q30");
  }
  if (!j.contains("q31") || !j.at("q31").is_number()) {
    throw std::runtime_error("Missing or invalid q31");
  }
  if (!j.contains("q40") || !j.at("q40").is_number()) {
    throw std::runtime_error("Missing or invalid q40");
  }
}

/**
 * @brief Read model configuration from json file, under model/params.
 *
 * @param j json file
 * @param m model
 */
void from_json(const json &j, Tungsten &m) {
  validate(j);
  auto &p = m.params;
  double value;
  j.at("n0").get_to(value);
  p.set_n0(value);
  j.at("n_sol").get_to(value);
  p.set_n_sol(value);
  j.at("n_vap").get_to(value);
  p.set_n_vap(value);
  j.at("T").get_to(value);
  p.set_T(value);
  j.at("T0").get_to(value);
  p.set_T0(value);
  j.at("Bx").get_to(value);
  p.set_Bx(value);
  j.at("alpha").get_to(value);
  p.set_alpha(value);
  j.at("alpha_farTol").get_to(value);
  p.set_alpha_farTol(value);
  int int_value;
  j.at("alpha_highOrd").get_to(int_value);
  p.set_alpha_highOrd(int_value);
  j.at("lambda").get_to(value);
  p.set_lambda(value);
  j.at("stabP").get_to(value);
  p.set_stabP(value);
  j.at("shift_u").get_to(value);
  p.set_shift_u(value);
  j.at("shift_s").get_to(value);
  p.set_shift_s(value);
  j.at("p2").get_to(value);
  p.set_p2(value);
  j.at("p3").get_to(value);
  p.set_p3(value);
  j.at("p4").get_to(value);
  p.set_p4(value);
  j.at("q20").get_to(value);
  p.set_q20(value);
  j.at("q21").get_to(value);
  p.set_q21(value);
  j.at("q30").get_to(value);
  p.set_q30(value);
  j.at("q31").get_to(value);
  p.set_q31(value);
  j.at("q40").get_to(value);
  p.set_q40(value);
  // Derived parameters are automatically calculated when accessed via getters
}

#endif // TUNGSTEN_INPUT_HPP
