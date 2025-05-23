// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/discrete_field.hpp>
#include <openpfc/factory/decomposition_factory.hpp>
#include <openpfc/utils.hpp>

using namespace pfc;
using namespace pfc::utils;

/**
 * \example 08_discrete_fields.cpp
 *
 * Arrays are already quite useful for applying modification to data. Our aim is
 * to do this as easily as possible, because in general, users want to define
 * several different initial conditions and/or boundary conditions for their
 * simulations. By combining the information from World and Decomposition with
 * Array, we can construct "coordinate-aware arrays", which are aware which part
 * of domain decomposition they represent (via the information provided by size
 * of offset from Decomposition), as well as their relative relation to physical
 * coordinate system (origin and discretization from World).
 *
 * In example 07, it was shown how to use multidimensional arrays to manually
 * "decompose" one bigger array to two smaller ones, manually calculate
 * coordinate system and fill arrays with some data based on physical
 * coordinates. This example reimplements 07 using DiscreteField.
 *
 * It's possible to add field modifier $$f(x,y,z) = 1 + x + y^2$$ using
 * anonymous function like before, yet more flexible way to work around would be
 * use a class with `operator()` overloading.
 */

class Modifier {
private:
  double constant = 1.0;

public:
  double operator()(double x, double y, double z) const {
    return constant + x + y * y + 0.0 * z;
  }
};

auto create_field(const pfc::Decomposition &decomp, int field_num) {
  auto subworld = get_subworld(decomp, field_num);
  auto size = get_size(subworld);
  auto lower = get_lower(subworld);
  auto origin = get_origin(subworld);
  auto spacing = get_spacing(subworld);
  return DiscreteField<double, 3>(size, lower, origin, spacing);
}

int main() {
  auto world = world::create({16, 8, 1});
  std::cout << world << std::endl;
  auto decomposition = decomposition::create(world, 4);

  std::cout << decomposition << std::endl;

  auto field1 = create_field(decomposition, 0);
  auto field2 = create_field(decomposition, 1);
  auto field3 = create_field(decomposition, 2);
  auto field4 = create_field(decomposition, 3);
  std::cout << field1 << std::endl;
  std::cout << field2 << std::endl;
  std::cout << field3 << std::endl;
  std::cout << field4 << std::endl;

  // Define function that is applied to fields. Can can be callable which
  // returns type T and takes std::array<double, D> as input argument.
  // Alternatively, specializations are made for D=2 and D=3, so those also
  // works, taking input arguments `double x, double y` or `double x, double y,
  // double z`.
  Modifier func1;

  // This would be alternative way ...
  auto func2 = [](const std::array<double, 3> &coords) {
    auto [x, y, z] = coords;
    return 1.0 + x + y * y;
  };

  // ... or even this
  auto func3 = [](auto x, auto y, auto z) { return 1.0 + x + y * y + 0.0 * z; };

  // Here, we apply some function to four different "sub-domains" of a single
  // field. Coordinate transforms and knowledge of offsets and dimensions make
  // sure that function gets applied correctly to each part of the domain.
  field1.apply(func1);
  field2.apply(func2);
  field3.apply(func3);
  field4.apply([](auto x, auto y, auto z) { return 1.0 + x + y * y + 0.0 * z; });

  // Keep on mind, that in general, one would define only one decomposition and
  // thus one "field" for each MPI process. Thus it's hard to say, given some
  // spesific coordinate (x, y, z), in which MPI process it stays, and some
  // extra work to find it needs to be done, potentially involving MPI traffic.
  auto probe = [&](double x, double y) {
    std::array<DiscreteField<double, 3>, 4> fields{field1, field2, field3, field4};
    const std::array<double, 3> coords = {x, y, 0.0};
    int field_num = 0;
    for (auto &field : fields) {
      if (field.inbounds(coords)) {
        std::cout << "Coordinate " << array_to_string(coords)
                  << " found from sub-domain #" << field_num << std::endl;
        std::cout << "Value at " << array_to_string(coords) << " is "
                  << field.interpolate(coords) << std::endl;
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
