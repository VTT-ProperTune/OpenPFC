// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

using namespace pfc;

class ConstantIC : public FieldModifier {
public:
  explicit ConstantIC(const std::string &field_name, double value) : value_(value) {
    set_field_name(field_name);
  }

  void apply(pfc::field::FieldOutput<double> field, const Domain & /*domain*/, const Box3i & /*box*/,
             double /*t*/) override {
    for (double &elem : field) {
      elem = value_;
    }
  }

private:
  double value_;
};

TEST_CASE("FieldModifier integration: constant IC",
          "[integration][field][modifier]") {
  auto domain = pfc::domain::create(pfc::Int3{16, 16, 16});
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);
  ConstantIC ic("density", 0.25);
  ic.apply(psi, domain, box, 0.0);
  REQUIRE_FALSE(psi.empty());
  bool values_match = true;
  for (const auto &v : psi) {
    values_match &= v == Catch::Approx(0.25).margin(1e-12);
  }
  REQUIRE(values_match);
}
