// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>

// start snippet modifier
class AffineFill : public pfc::FieldModifier {
  double offset_;
  pfc::Real3 slope_;

public:
  AffineFill(double offset, pfc::Real3 slope) : offset_(offset), slope_(slope) {}
  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double) override {
    pfc::field::apply(field, domain, box, [&](const pfc::Real3 &x) {
      return offset_ + slope_[0] * x[0] + slope_[1] * x[1] + slope_[2] * x[2];
    });
  }
};
// end snippet modifier

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto config = initialization::input(argc, argv);
    const auto domain = initialization::domain(config);
    const auto partition = pfc::decomposition::create(domain, comm.size());
    auto field = pfc::data::field_from_subdomain<double>(partition, comm.rank(), 0);
    // start snippet catalog
    auto catalog = pfc::ui::make_builtin_field_modifier_catalog();
    catalog.register_modifier("affine", [](const nlohmann::json &parameters) {
      const double offset = parameters.at("offset").get<double>();
      const auto slope = parameters.at("slope").get<pfc::Real3>();
      if (!std::isfinite(offset) ||
          !std::ranges::all_of(slope, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("Affine parameters must be finite");
      return std::make_unique<AffineFill>(offset, slope);
    });
    const auto specification =
        config.value("initial_condition", nlohmann::json{{"type", "affine"},
                                                         {"offset", 1},
                                                         {"slope", {.2, -.1, .05}}});
    auto modifier = catalog.create_modifier(
        specification.at("type").get<std::string>(), specification);
    pfc::apply_field_modifier(*modifier, field, 0);
    // end snippet catalog
    auto output = initialization::output(comm.rank());
    initialization::write(output, specification.at("type").get<std::string>(),
                          field);
    if (comm.rank() == 0)
      std::cout << "Applied " << specification.at("type") << '\n';
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
