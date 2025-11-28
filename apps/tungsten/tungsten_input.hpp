// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_INPUT_HPP
#define TUNGSTEN_INPUT_HPP

#include "tungsten_model.hpp"
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
  j.at("n0").get_to(p.n0);
  j.at("n_sol").get_to(p.n_sol);
  j.at("n_vap").get_to(p.n_vap);
  j.at("T").get_to(p.T);
  j.at("T0").get_to(p.T0);
  j.at("Bx").get_to(p.Bx);
  j.at("alpha").get_to(p.alpha);
  j.at("alpha_farTol").get_to(p.alpha_farTol);
  j.at("alpha_highOrd").get_to(p.alpha_highOrd);
  p.tau = p.T / p.T0;
  j.at("lambda").get_to(p.lambda);
  j.at("stabP").get_to(p.stabP);
  j.at("shift_u").get_to(p.shift_u);
  j.at("shift_s").get_to(p.shift_s);
  j.at("p2").get_to(p.p2);
  j.at("p3").get_to(p.p3);
  j.at("p4").get_to(p.p4);
  p.p2_bar = p.p2 + 2 * p.shift_s * p.p3 + 3 * pow(p.shift_s, 2) * p.p4;
  p.p3_bar = p.shift_u * (p.p3 + 3 * p.shift_s * p.p4);
  p.p4_bar = pow(p.shift_u, 2) * p.p4;
  j.at("q20").get_to(p.q20);
  j.at("q21").get_to(p.q21);
  j.at("q30").get_to(p.q30);
  j.at("q31").get_to(p.q31);
  j.at("q40").get_to(p.q40);
  p.q20_bar = p.q20 + 2.0 * p.shift_s * p.q30 + 3.0 * pow(p.shift_s, 2) * p.q40;
  p.q21_bar = p.q21 + 2.0 * p.shift_s * p.q31;
  p.q30_bar = p.shift_u * (p.q30 + 3.0 * p.shift_s * p.q40);
  p.q31_bar = p.shift_u * p.q31;
  p.q40_bar = pow(p.shift_u, 2) * p.q40;
  p.q2_bar = p.q21_bar * p.tau + p.q20_bar;
  p.q3_bar = p.q31_bar * p.tau + p.q30_bar;
  p.q4_bar = p.q40_bar;
}

#endif // TUNGSTEN_INPUT_HPP
