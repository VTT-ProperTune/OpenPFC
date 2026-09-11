// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli.hpp
 * @brief `--key=value` argument parsing for the two `alloy_dendrite_elastic`
 *        drivers.
 *
 * @details
 * Named options rather than positional ones, because a Stage-1 run has
 * fifteen knobs and a positional list of fifteen numbers is a bug waiting to
 * be typed. Unknown keys are an error, not a warning: silently ignoring
 * `--lamda=2` would produce a plausible-looking run at the default `lambda`
 * and a wrong conclusion, which is exactly the failure mode a verification
 * app must not have.
 */

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vlasov {

/// Parsed `--key=value` pairs, with typed lookup and unknown-key detection.
class Options {
public:
  Options(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string a(argv[i]);
      if (a == "-h" || a == "--help") {
        m_help = true;
        continue;
      }
      if (a.rfind("--", 0) != 0) {
        throw std::invalid_argument("unexpected positional argument: " + a);
      }
      const auto eq = a.find('=');
      if (eq == std::string::npos) {
        m_kv[a.substr(2)] = "1"; // bare flag
      } else {
        m_kv[a.substr(2, eq - 2)] = a.substr(eq + 1);
      }
    }
  }

  [[nodiscard]] bool help() const noexcept { return m_help; }

  /// Was @p key supplied on the command line? Lets a driver distinguish
  /// "left at its default" from "explicitly set to the default value",
  /// which matters when the default is itself derived from another option
  /// (`lambda_el` follows `lambda` unless it is asked for by name).
  [[nodiscard]] bool has(const std::string &key) const {
    return m_kv.find(key) != m_kv.end();
  }

  double real(const std::string &key, double dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.push_back(key);
    return std::atof(it->second.c_str());
  }
  int integer(const std::string &key, int dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.push_back(key);
    return std::atoi(it->second.c_str());
  }
  bool flag(const std::string &key, bool dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.push_back(key);
    return it->second != "0" && it->second != "false";
  }
  std::string text(const std::string &key, const std::string &dflt) {
    const auto it = m_kv.find(key);
    if (it == m_kv.end()) {
      return dflt;
    }
    m_used.push_back(key);
    return it->second;
  }

  /// Throw if any supplied key was never queried. Call after all lookups.
  void require_all_consumed() const {
    std::string bad;
    for (const auto &[k, v] : m_kv) {
      bool found = false;
      for (const auto &u : m_used) {
        if (u == k) {
          found = true;
          break;
        }
      }
      if (!found) {
        bad += (bad.empty() ? "" : ", ") + k;
      }
    }
    if (!bad.empty()) {
      throw std::invalid_argument("unknown option(s): " + bad);
    }
  }

private:
  std::map<std::string, std::string> m_kv;
  std::vector<std::string> m_used;
  bool m_help{false};
};

} // namespace vlasov
