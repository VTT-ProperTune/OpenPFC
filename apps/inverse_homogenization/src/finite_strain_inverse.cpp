// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file finite_strain_inverse.cpp
 * @brief CLI for the frozen 2-D experiment.
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <inverse_homogenization/finite_strain_inverse.hpp>

namespace {

void write_evidence(const std::string &path,
                    const pfc::apps::inverse::fs::Experiment484 &ex) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << std::setprecision(16);
  out << "{\n";
  out << "  \"schema\": \"openpfc-inverse-evidence/v1\",\n";
  out << "  \"openpfc_issue\": 57,\n";
  out << "  \"research_issue\": \"\",\n";
  out << "  \"grid\": \"16x16\",\n";
  out << "  \"model\": \"neo-hookean\",\n";
  out << "  \"all_converged\": " << (ex.all_converged ? "true" : "false") << ",\n";
  out << "  \"any_programmed_crossing\": "
      << (ex.any_programmed_crossing ? "true" : "false") << ",\n";
  out << "  \"designs\": [\n";
  for (std::size_t i = 0; i < ex.designs.size(); ++i) {
    const auto &d = ex.designs[i];
    out << "    {\"arm\": \"" << d.arm << "\", \"half\": " << d.half
        << ", \"K_small\": " << d.K_small
        << ", \"cross_found\": " << (d.cross_found ? "true" : "false")
        << ", \"eps_cross\": " << d.eps_cross << ", \"points\": [";
    for (std::size_t j = 0; j < d.points.size(); ++j) {
      const auto &p = d.points[j];
      out << "{\"F11\": " << p.F11 << ", \"F22\": " << p.F22
          << ", \"P11\": " << p.P11 << ", \"nu_t\": " << p.nu_t
          << ", \"converged\": " << (p.converged ? "true" : "false") << "}";
      if (j + 1 < d.points.size()) out << ", ";
    }
    out << "]}";
    if (i + 1 < ex.designs.size()) out << ",";
    out << "\n";
  }
  out << "  ],\n";
  out << "  \"interpretation\": \"" << ex.interpretation << "\"\n";
  out << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string evidence;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--evidence" && i + 1 < argc)
      evidence = argv[++i];
    else if (a.rfind("--evidence=", 0) == 0)
      evidence = a.substr(11);
    else if (a == "--help" || a == "-h") {
      std::cout << "openpfc_finite_strain_inverse [--evidence PATH]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      return 2;
    }
  }
  const auto ex = pfc::apps::inverse::fs::run_declared_484_experiment();
  std::cout << std::setprecision(8);
  std::cout << "FINITE_STRAIN_INVERSE_CHECKSUM"
            << " n=" << ex.designs.size()
            << " converged=" << (ex.all_converged ? 1 : 0)
            << " crossing=" << (ex.any_programmed_crossing ? 1 : 0) << "\n";
  for (const auto &d : ex.designs) {
    std::cout << d.arm << " half=" << d.half << " K=" << d.K_small
              << " cross=" << (d.cross_found ? d.eps_cross : -1.0);
    if (!d.points.empty())
      std::cout << " nu0=" << d.points.front().nu_t
                << " nu1=" << d.points.back().nu_t;
    std::cout << "\n";
  }
  std::cout << ex.interpretation << "\n";
  if (!evidence.empty()) {
    write_evidence(evidence, ex);
    std::cout << "wrote " << evidence << "\n";
  }
  return ex.all_converged ? 0 : 1;
}
