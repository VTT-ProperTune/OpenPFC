// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <iostream>
#include <openpfc/frontend/utils/array_to_string.hpp>
#include <openpfc/frontend/utils/typename.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

/**
 * \example 07_array.cpp
 *
 * OpenPFC implements a basic multidimensional field to make it easier to work
 * with grid data. The new `pfc::data::Field<T>` is the canonical field container
 * that replaces the legacy `Array<T,D>` and `DiscreteField<T,D>` types.
 *
 * The field API makes it possible to work with decomposed domains by storing
 * geometry information (origin, spacing) alongside the data. Fields are aware
 * of which part of domain decomposition they represent and can map between
 * grid indices and physical coordinates. This helps when working with discrete
 * fields of data that are decomposed across multiple machines using MPI.
 *
 * In this example, we introduce two fields representing different parts of a
 * decomposed domain, demonstrating field construction, indexing, and coordinate
 * mapping. We use a 2D simulation extended to 3D (with size 1 in the third dimension)
 * to preserve the educational intent while using the 3D field API.
 */

using namespace pfc;
using namespace pfc::data;

template <typename T> struct SecondOrderTensor {
  std::array<T, 9> data;

  friend std::ostream &operator<<(std::ostream &os,
                                  const SecondOrderTensor<T> &tensor) {
    os << std::string("SecondOrderTensor<") + TypeName<T>::get().data() + ">"
       << utils::array_to_string(tensor.data);
    return os;
  }
};

int main() {
  int Lx = 16;
  int Ly = 8;
  int Lz = 1; // Extended to 3D for Field API

  // Create a domain for the whole simulation
  auto domain = domain::create({Lx, Ly, Lz});

  // "Process 0" contains the first part of domain
  Field<double> field0(domain, Box3i::from_bounds({0, 0, 0}, {Lx / 2 - 1, Ly - 1, 0}), 0);
  // "Process 1" contains the second part of domain
  Field<double> field1(domain, Box3i::from_bounds({Lx / 2, 0, 0}, {Lx - 1, Ly - 1, 0}), 0);

  std::cout << "Field 0 (process 0):" << std::endl;
  std::cout << "  Owned box: [" << field0.box().low[0] << "," << field0.box().low[1] << ","
            << field0.box().low[2] << "] to [" << field0.box().high[0] << ","
            << field0.box().high[1] << "," << field0.box().high[2] << "]" << std::endl;
  std::cout << "  Local size: [" << field0.local_size()[0] << "," << field0.local_size()[1]
            << "," << field0.local_size()[2] << "]" << std::endl;

  std::cout << "Field 1 (process 1):" << std::endl;
  std::cout << "  Owned box: [" << field1.box().low[0] << "," << field1.box().low[1] << ","
            << field1.box().low[2] << "] to [" << field1.box().high[0] << ","
            << field1.box().high[1] << "," << field1.box().high[2] << "]" << std::endl;
  std::cout << "  Local size: [" << field1.local_size()[0] << "," << field1.local_size()[1]
            << "," << field1.local_size()[2] << "]" << std::endl;

  // Accessing fields:
  field0(0, 0, 0) = 1.0;
  field1(0, 0, 0) = 2.0;  // Local index (0,0,0) corresponds to global (8,0,0)
  std::cout << "First item of field0: " << field0(0, 0, 0) << std::endl;
  std::cout << "First item of field1: " << field1(0, 0, 0) << std::endl;

  // We can get access to underlying data with linear index, if needed.
  // Field stores its data in a contiguous buffer, and linear indexing works
  // across the entire owned domain:
  std::cout << "First item of field0: " << field0.data()[0] << std::endl;
  std::cout << "First item of field1: " << field1.data()[0] << std::endl;

  // In practice, one might want to apply some function to fields, where each
  // field contains different part of domain. This can be done with `apply`,
  // which takes a function of physical coordinates (x, y, z):
  field0.apply([](double x, double y, double /*z*/) {
    return 1.0 + x + y * y;
  });
  field1.apply([](double x, double y, double /*z*/) {
    return 1.0 + x + y * y;
  });

  std::cout << "First item of field0: " << field0(0, 0, 0)
            << std::endl; // 1.0 + 0.0 + 0.0 * 0.0 = 1
  std::cout << "First item of field1: " << field1(0, 0, 0)
            << std::endl; // 1.0 + 8.0 + 0.0 * 0.0 = 9

  // Another way to fill and modify field would be to use STL algorithms.
  // We can use for_each_owned to iterate over all cells:
  field0.for_each_owned([&](int i, int j, int k) {
    const auto coords = field0.coords(i, j, k);
    field0(i, j, k) = 1.0 + coords[0] + coords[1] * coords[1];
  });
  field1.for_each_owned([&](int i, int j, int k) {
    const auto coords = field1.coords(i, j, k);
    field1(i, j, k) = 1.0 + coords[0] + coords[1] * coords[1];
  });

  // Fields can also hold more complex objects. Here we have field of tensors:
  auto tensor_domain = domain::create({2, 2, 1});
  Field<SecondOrderTensor<double>> tensors(
      tensor_domain, Box3i::from_bounds({0, 0, 0}, {1, 1, 0}), 0);
  std::cout << "Field of tensors:" << std::endl;
  tensors(1, 1, 0) = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  // Iterate over all owned cells with for_each_owned
  tensors.for_each_owned([&](int i, int j, int k) {
    std::cout << "Local index [" << i << "," << j << "," << k << "] => "
              << tensors(i, j, k) << std::endl;
  });

  return 0;
}
