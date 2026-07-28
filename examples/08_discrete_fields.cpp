// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/frontend/utils/array_to_string.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;
using namespace pfc::data;
using namespace pfc::utils;

/**
 * \example 08_discrete_fields.cpp
 *
 * The `pfc::data::Field<T>` is the canonical field container for distributed
 * field data. It makes it easy to apply modifications to data, define initial
 * conditions and boundary conditions for simulations.
 *
 * Field combines geometry information (coordinate system) with data storage,
 * making it aware of which part of domain decomposition it represents and its
 * relationship to physical coordinates and discretization.
 *
 * In example 07, we manually decomposed arrays and calculated coordinate systems.
 * This example shows how the Field API simplifies this by integrating domain
 * decomposition information directly into the field structure.
 *
 * It's possible to add field modifier $$f(x,y,z) = 1 + x + y^2$$ using
 * anonymous functions or classes with `operator()` overloading.
 */

class Modifier {
private:
  double constant = 1.0;

public:
  double operator()(double x, double y, double z) const {
    return constant + x + y * y + 0.0 * z;
  }
};

int main() {
  auto domain = domain::create({16, 8, 1});
  std::cout << "Domain: " << domain << std::endl;
  auto decomposition = decomposition::create(domain, 4);
  std::cout << "Decomposition: " << decomposition << std::endl;

  // Construct fields from decomposition using the factory
  auto field1 = pfc::data::field_from_subdomain<double>(decomposition, 0, 0);
  auto field2 = pfc::data::field_from_subdomain<double>(decomposition, 1, 0);
  auto field3 = pfc::data::field_from_subdomain<double>(decomposition, 2, 0);
  auto field4 = pfc::data::field_from_subdomain<double>(decomposition, 3, 0);

  std::cout << "\nField 1:" << std::endl;
  std::cout << "  Owned box: [" << field1.box().low[0] << "," << field1.box().low[1] << ","
            << field1.box().low[2] << "] to [" << field1.box().high[0] << ","
            << field1.box().high[1] << "," << field1.box().high[2] << "]" << std::endl;
  std::cout << "\nField 2:" << std::endl;
  std::cout << "  Owned box: [" << field2.box().low[0] << "," << field2.box().low[1] << ","
            << field2.box().low[2] << "] to [" << field2.box().high[0] << ","
            << field2.box().high[1] << "," << field2.box().high[2] << "]" << std::endl;
  std::cout << "\nField 3:" << std::endl;
  std::cout << "  Owned box: [" << field3.box().low[0] << "," << field3.box().low[1] << ","
            << field3.box().low[2] << "] to [" << field3.box().high[0] << ","
            << field3.box().high[1] << "," << field3.box().high[2] << "]" << std::endl;
  std::cout << "\nField 4:" << std::endl;
  std::cout << "  Owned box: [" << field4.box().low[0] << "," << field4.box().low[1] << ","
            << field4.box().low[2] << "] to [" << field4.box().high[0] << ","
            << field4.box().high[1] << "," << field4.box().high[2] << "]" << std::endl;

  // Define functions that are applied to fields. Field::apply() takes a callable
  // that accepts physical coordinates (x, y, z) as separate arguments.
  Modifier func1;

  // Alternative: lambda taking separate coordinates
  auto func2 = [](double x, double y, double z) { return 1.0 + x + y * y; };

  // Alternative: lambda taking separate coordinates
  auto func3 = [](double x, double y, double z) { return 1.0 + x + y * y + 0.0 * z; };

  // Apply functions to the four sub-domains. Field API handles coordinate
  // transforms correctly for each part of the domain.
  field1.apply(func1);
  field2.apply(func2);
  field3.apply(func3);
  field4.apply([](double x, double y, double z) { return 1.0 + x + y * y + 0.0 * z; });

  // Keep in mind that in general, one would define only one decomposition and
  // thus one field for each MPI process. It requires extra work to determine
  // which MPI process contains a specific coordinate (x, y, z), potentially
  // involving MPI traffic.
  auto probe = [&](double x, double y) {
    std::array<Field<double>, 4> fields{field1, field2, field3, field4};
    const std::array<double, 3> coords = {x, y, 0.0};
    int field_num = 0;
    for (auto &field : fields) {
      // Check if coordinate is within field's physical bounds
      const auto &origin = field.domain().origin;
      const auto &spacing = field.domain().spacing;
      const auto &local_size = field.local_size();

      // Calculate physical bounds
      double x_low = origin[0] + field.box().low[0] * spacing[0];
      double x_high = origin[0] + field.box().high[0] * spacing[0] + spacing[0];
      double y_low = origin[1] + field.box().low[1] * spacing[1];
      double y_high = origin[1] + field.box().high[1] * spacing[1] + spacing[1];

      bool inbounds = (coords[0] >= x_low && coords[0] < x_high &&
                       coords[1] >= y_low && coords[1] < y_high);

      if (inbounds) {
        // Map coordinate to local indices
        int i = static_cast<int>((coords[0] - origin[0]) / spacing[0]) - field.box().low[0];
        int j = static_cast<int>((coords[1] - origin[1]) / spacing[1]) - field.box().low[1];
        int k = 0; // z is fixed at 0

        if (i >= 0 && i < local_size[0] && j >= 0 && j < local_size[1]) {
          std::cout << "Coordinate " << array_to_string(coords)
                    << " found from sub-domain #" << field_num << std::endl;
          std::cout << "Value at " << array_to_string(coords) << " is "
                    << field(i, j, k) << std::endl;
        }
      }
      field_num++;
    }
  };
  probe(4.0,
        2.0); // gives 9, since 1 + 4 + 2 * 2 = 9, found from first sub-domain
  probe(12.0,
        6.0); // gives 49, since 1 + 12 + 6 * 6 = 49, found from last sub-domain
  std::cout << func3(4.0, 2.0, 0.0) << std::endl;
  std::cout << func3(12.0, 6.0, 0.0) << std::endl;

  return 0;
}
