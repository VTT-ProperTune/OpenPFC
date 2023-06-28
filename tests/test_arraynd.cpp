#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
using namespace Catch::Matchers;

#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <vector>

template <typename T, size_t D> class array {
protected:
  std::vector<T> data;
  std::array<int, D> size;
  std::array<int, D> offset;

public:
  array(const std::array<int, D> &dimensions, const std::array<int, D> &offsets) : size(dimensions), offset(offsets) {
    int totalSize = 1;
    for (int dim : size) {
      totalSize *= dim;
    }
    data.resize(totalSize);
  }

  bool inbounds(const std::array<int, D> &indices) const {
    for (size_t i = 0; i < D; ++i) {
      if (indices[i] < offset[i] || indices[i] >= offset[i] + size[i]) {
        return false;
      }
    }
    return true;
  }

  std::vector<T> &get_data() { return data; }

  T &operator()(const std::array<int, D> &indices) { return data[calculate_index(indices)]; }

  T &operator[](const std::array<int, D> &indices) { return operator()(indices); }

  T &operator[](int idx) { return data[idx]; }

  size_t calculate_index(const std::array<int, D> &indices) const {
    auto itSize = size.rbegin();
    auto itOffset = offset.rbegin();
    auto itIndex = indices.rbegin();
    int idx = *itIndex - *itOffset;
    ++itSize;
    ++itOffset;
    ++itIndex;

    while (itSize != size.rend()) {
      idx = (idx * (*itSize)) + (*itIndex - *itOffset);
      ++itSize;
      ++itOffset;
      ++itIndex;
    }

    return idx;
  }

  std::array<int, D> calculate_index_inverse(size_t idx) const {
    std::array<int, D> indices;
    for (size_t i = 0; i < D; ++i) {
      indices[i] = (idx % size[i]) + offset[i];
      idx /= size[i];
    }
    return indices;
  }

  std::array<int, D> get_size() const { return size; }

  std::array<int, D> get_offset() const { return offset; }

  void apply(const std::function<T(std::array<int, D>)> &&func) {
    std::array<int, D> indices = offset;
    for (size_t idx = 0; idx < data.size(); idx++) {
      data[idx] = std::invoke(std::forward<const std::function<T(std::array<int, D>)>>(func), indices);
      for (size_t i = 0; i < D; i++) {
        indices[i] += 1;
        if (indices[i] < offset[i] + size[i]) break;
        indices[i] = offset[i];
      }
    }
  }
};

/*
template <typename T> class array<T, 3> : public std::vector<T> {
private:
  int Lx, Ly, Lz;
  int i0, j0, k0;

public:
  array(const std::array<int, 3> &dimensions, const std::array<int, 3> &offsets)
      : std::vector<T>(dimensions[0] * dimensions[1] * dimensions[2]), Lx(dimensions[0]), Ly(dimensions[1]),
        Lz(dimensions[2]), i0(offsets[0]), j0(offsets[1]), k0(offsets[2]) {
    if constexpr (std::is_same_v<T, bool>) std::fill(this->begin(), this->end(), false);
  }

  int calculate_index(int i, int j, int k) const { return (i - i0) + (j - j0) * Lx + (k - k0) * Lx * Ly; }
  T &operator()(int i, int j, int k) { return (*this)[calculate_index(i, j, k)]; }
  T &operator[](int idx) { return std::vector<T>::operator[](idx); }
  std::array<int, 3> get_size() const { return {Lx, Ly, Lz}; }
  T &operator[](const std::initializer_list<int> &indices) { return operator()(indices); }
  int calculate_index(const std::array<int, 3> &indices) const {
    return calculate_index(indices[0], indices[1], indices[2]);
  }
  T &operator()(const std::array<int, 3> &indices) { return (*this)(indices[0], indices[1], indices[2]); }
};
*/

TEST_CASE("array1d") {
  // array with dimension 5, offset 2
  array<int, 1> arr({5}, {2});

  SECTION("Test setting and accessing elements using linear indexing") {
    arr[0] = 1;
    arr[1] = 2;
    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);
    REQUIRE(arr.get_data()[0] == 1);
    REQUIRE(arr.get_data()[1] == 2);
  }

  SECTION("Test setting and accessing elements using custom indexing") {
    arr({2}) = 1;
    arr({3}) = 2;
    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);
  }

  SECTION("Test in-bounds") {
    // 0 1 2 3 4 5 6
    //     ^
    //     0 1 2 3 4
    REQUIRE_FALSE(arr.inbounds({1}));
    REQUIRE(arr.inbounds({2}));
    REQUIRE(arr.inbounds({6}));
    REQUIRE_FALSE(arr.inbounds({7}));
  }

  SECTION("Test get_size()") {
    REQUIRE(arr.get_size()[0] == 5);
  }

  SECTION("Test apply()") {
    auto func = [](const std::array<int, 1> &indices) -> int { return 2 * indices[0]; };
    arr.apply(func);
    REQUIRE(arr[0] == 4);
    REQUIRE(arr[1] == 6);
    REQUIRE(arr[2] == 8);
    REQUIRE(arr[3] == 10);
    REQUIRE(arr[4] == 12);
  }
}

TEST_CASE("array2d") {
  array<int, 2> arr({2, 3}, {1, 2});
  /**
   *    2  3  4
   * 1  x  x  x
   * 2  x  x  x
   */

  SECTION("Test setting and accessing elements using custom / linear indexing") {
    arr({1, 2}) = 1;
    arr({2, 2}) = 2;
    arr({1, 3}) = 3;
    arr({2, 3}) = 4;
    arr({1, 4}) = 5;
    arr({2, 4}) = 6;
    // for (int i = 0; i < 6; i++) std::cout << arr[i] << std::endl;
    for (int i = 0; i < 6; i++) REQUIRE(arr[i] == i + 1);
  }

  SECTION("Test apply()") {
    auto func = [](const std::array<int, 2> &indices) -> int { return indices[0] + indices[1]; };
    arr.apply(func);
    REQUIRE(arr[0] == 1 + 2);
    REQUIRE(arr[1] == 2 + 2);
    REQUIRE(arr[2] == 1 + 3);
    REQUIRE(arr[3] == 2 + 3);
    REQUIRE(arr[4] == 1 + 4);
    REQUIRE(arr[5] == 2 + 4);
  }
}

template <typename T, size_t D> class DiscreteField : public array<T, D> {
private:
  std::array<double, D> origin;
  std::array<double, D> discretization;

public:
  DiscreteField(const std::array<int, D> &dimensions, const std::array<int, D> &offsets,
                const std::array<double, D> &fieldOrigin, const std::array<double, D> &fieldDiscretization)
      : array<T, D>(dimensions, offsets), origin(fieldOrigin), discretization(fieldDiscretization) {}

  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  iterator begin() { return this->data.begin(); }
  iterator end() { return this->data.end(); }
  const_iterator begin() const { return this->data.begin(); }
  const_iterator end() const { return this->data.end(); }

  std::array<double, D> mapIndicesToCoordinates(const std::array<int, D> &indices) const {
    std::array<double, D> coordinates;
    for (size_t i = 0; i < D; ++i) {
      coordinates[i] = origin[i] + indices[i] * discretization[i];
    }
    return coordinates;
  }

  std::array<int, D> mapCoordinatesToIndices(const std::array<double, D> &coordinates) const {
    std::array<int, D> indices;
    for (size_t i = 0; i < D; ++i) {
      indices[i] = static_cast<int>(std::round((coordinates[i] - origin[i]) / discretization[i]));
    }
    return indices;
  }

  bool inbounds(const std::array<double, D> &coordinates) const {
    std::array<int, D> indices = mapCoordinatesToIndices(coordinates);
    return array<T, D>::inbounds(indices);
  }

  T &operator()(const std::array<double, D> &coordinates) {
    std::array<int, D> indices = mapCoordinatesToIndices(coordinates);
    return array<T, D>::operator()(indices);
  }

  void apply(const std::function<T(std::array<double, D>)> &&func) {
    std::array<int, D> indices = this->offset;
    std::array<double, D> coordinates = {0};
    for (size_t idx = 0; idx < this->data.size(); idx++) {
      for (size_t i = 0; i < D; ++i) coordinates[i] = origin[i] + indices[i] * discretization[i];
      this->data[idx] = std::invoke(std::forward<const std::function<T(std::array<double, D>)>>(func), coordinates);
      for (size_t i = 0; i < D; i++) {
        indices[i] += 1;
        if (indices[i] < this->offset[i] + this->size[i]) break;
        indices[i] = this->offset[i];
      }
    }
  }
};

TEST_CASE("DiscreteField1D") {
  int Lx = 5;
  int i0 = -2;
  double x0 = 1.0;
  double dx = 2.0;
  DiscreteField<int, 1> field({Lx}, {i0}, {x0}, {dx});

  SECTION("Accessing elements using indices") {
    field({0}) = 1;
    REQUIRE(field({0}) == 1);
  }

  SECTION("Accessing elements using coordinates") {
    field({2.0}) = 1;
    REQUIRE(field({1.9}) == 0);
    REQUIRE(field({2.0}) == 1);
    REQUIRE(field({2.1}) == 1);
  }

  SECTION("Test apply()") {
    auto func = [](const std::array<double, 1> &coords) -> int { return static_cast<int>(coords[0]); };
    field.apply(func);
    for (int i = 0; i < Lx; i++) REQUIRE(field[i] == -3 + 2 * i);
  }
}
