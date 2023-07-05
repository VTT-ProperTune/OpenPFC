#include <array>
#include <openpfc/utils/multi_index.hpp>

using namespace pfc::utils;

/**
 * \example 06_multi_index.cpp
 *
 * The following example demonstrates how the auxiliary class MultiIndex can
 * facilitate index-related calculations. As the name suggests, MultiIndex
 * implements a way to refer to multiple indices instead of a single linear
 * index, where each index corresponds to a dimension. Typically, there are
 * three dimensions.
 */
int main() {
  // To construct MultiIndex, we need to define offset and size. This constructs
  // MultiIndex which size if 2 in every dimension and offset is one.
  std::array<int, 3> size = {2, 2, 2};
  std::array<int, 3> offset = {1, 1, 1};
  MultiIndex index(size, offset);

  std::cout << index << std::endl;

  // Using MultiIndex through iterator:
  for (auto it = index.begin(); it != index.end(); ++it) {
    std::cout << it << ", linear index = " << it.get_linear_index() << std::endl;
  }
  // Result is
  /*
  {1, 1, 1}, linear index = 0
  {2, 1, 1}, linear index = 1
  {1, 2, 1}, linear index = 2
  {2, 2, 1}, linear index = 3
  {1, 1, 2}, linear index = 4
  {2, 1, 2}, linear index = 5
  {1, 2, 2}, linear index = 6
  {2, 2, 2}, linear index = 7
  */

  // Alternatively, using convenient range-based for loop:
  for (const auto &indices : index) {
    std::cout << "{" << indices[0] << ", " << indices[1] << ", " << indices[2] << "}" << std::endl;
  }

  // Filling vector using MultiIndex:
  std::array<int, 8> data;
  for (auto it = index.begin(); it != index.end(); ++it) {
    const auto [i, j, k] = *it;
    data[it] = 2 * i + j + 3 * k;
  }

  std::cout << "data = [";
  for (const auto &v : data) std::cout << v << ((&v != &data.back()) ? ", " : "]\n");

  // Iterator can be started from custom position
  std::cout << "Linear index of {2, 1, 2} = " << (int)index.from({2, 1, 2}) << std::endl;

  // Filling two-dimensional data, "matrix"
  std::array<int, 25> arr{};
  MultiIndex<2> index2d({5, 5}, {3, 3});
  std::array<int, 2> start_index{4, 4};
  std::array<int, 2> end_index{6, 6};
  for (int j = start_index[1]; j <= end_index[1]; ++j) {
    for (int i = start_index[0]; i <= end_index[0]; ++i) {
      arr[index2d.to_linear({i, j})] = 1;
    }
  }
  // Print the array as a matrix
  for (size_t i = 0; i < arr.size(); ++i) {
    std::cout << arr[i] << ' ';
    if ((i + 1) % 5 == 0) {
      std::cout << '\n';
    }
  }
  // The result is
  /*
       3 4 5 6 7  <-- indices in second dimension
       ---------
    3 |0 0 0 0 0
    4 |0 1 1 1 0
    5 |0 1 1 1 0
    6 |0 1 1 1 0
    7 |0 0 0 0 0
    ^
    indices in first dimension
   */
  return 0;
}
