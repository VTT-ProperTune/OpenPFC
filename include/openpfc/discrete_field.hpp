// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_DISCRETE_FIELD_HPP
#define PFC_DISCRETE_FIELD_HPP

#include "array.hpp"
#include "utils/show.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>

namespace pfc {

/**
 * @brief A class representing a discrete field.
 *
 * @tparam T The type of elements in the field.
 * @tparam D The dimensionality of the field.
 */
template <typename T, size_t D> class DiscreteField {
private:
  Array<T, D> m_array; /**< Multidimensional array containing data. */
  const std::array<double, D> m_origin; /**< The origin of the field. */
  const std::array<double, D>
      m_discretization; /**< The discretization of the field. */
  const std::array<double, D>
      m_coords_low; /**< The lower bound of coordinates. */
  const std::array<double, D>
      m_coords_high; /**< The upper bound of coordinates. */

  /**
   * @brief Calculate lower bounding box of this field.
   *
   */
  std::array<double, D> calculate_coords_low(const std::array<int, D> &offset) {
    std::array<double, D> coords_low;
    for (size_t i = 0; i < D; i++)
      coords_low[i] = m_origin[i] + offset[i] * m_discretization[i];
    return coords_low;
  }

  /**
   * @brief Calculate upper bounding box of this field.
   *
   */
  std::array<double, D> calculate_coords_high(const std::array<int, D> &offset,
                                              const std::array<int, D> &size) {
    std::array<double, D> coords_high;
    for (size_t i = 0; i < D; i++)
      coords_high[i] =
          m_origin[i] + (offset[i] + size[i]) * m_discretization[i];
    return coords_high;
  }

public:
  /**
   * @brief Constructs a DiscreteField object with the specified dimensions,
   * offsets, origin, and discretization.
   *
   * @param dimensions The dimensions of the field.
   * @param offsets The offsets of the field.
   * @param origin The origin of the field.
   * @param discretization The discretization of the field.
   */
  DiscreteField(const std::array<int, D> &dimensions,
                const std::array<int, D> &offsets,
                const std::array<double, D> &origin,
                const std::array<double, D> &discretization)
      : m_array(dimensions, offsets), m_origin(origin),
        m_discretization(discretization),
        m_coords_low(calculate_coords_low(offsets)),
        m_coords_high(calculate_coords_high(offsets, dimensions)) {}

  /**
   * @brief Constructs a DiscreteField from an Decomposition object.
   *
   * @param decomp The Decomposition object.
   */
  DiscreteField(const Decomposition &decomp)
      : DiscreteField(decomp.get_inbox_size(), decomp.get_inbox_offset(),
                      decomp.get_world().get_origin(),
                      decomp.get_world().get_spacing()) {}

  const std::array<double, D> &get_origin() const { return m_origin; }
  const std::array<double, D> &get_discretization() const {
    return m_discretization;
  }
  const std::array<double, D> &get_coords_low() const { return m_coords_low; }
  const std::array<double, D> &get_coords_high() const { return m_coords_high; }

  Array<T, D> &get_array() { return m_array; }
  const Array<T, D> &get_array() const { return m_array; }
  const MultiIndex<D> &get_index() { return get_array().get_index(); }
  const MultiIndex<D> &get_index() const { return get_array().get_index(); }
  std::vector<T> &get_data() { return get_array().get_data(); }

  /**
   * @brief Returns the element at the specified index.
   *
   * @param indices multi-dimensional indices
   * @return T&
   */
  T &operator[](const std::array<int, D> &indices) {
    return get_array()[indices];
  }

  /**
   * @brief Returns the element at the specified index.
   *
   * @param idx
   * @return T&
   */
  T &operator[](size_t idx) { return get_array()[idx]; }

  /**
   * @brief Maps indices to coordinates in the field.
   *
   * @param indices The indices to map.
   * @return The corresponding coordinates.
   */
  std::array<double, D>
  map_indices_to_coordinates(const std::array<int, D> &indices) const {
    std::array<double, D> coordinates;
    for (size_t i = 0; i < D; ++i) {
      coordinates[i] = m_origin[i] + indices[i] * m_discretization[i];
    }
    return coordinates;
  }

  /**
   * @brief Maps coordinates to indices in the field.
   *
   * @param coordinates The coordinates to map.
   * @return The corresponding indices.
   */
  std::array<int, D>
  map_coordinates_to_indices(const std::array<double, D> &coordinates) const {
    std::array<int, D> indices;
    for (size_t i = 0; i < D; ++i) {
      indices[i] = static_cast<int>(
          std::round((coordinates[i] - m_origin[i]) / m_discretization[i]));
    }
    return indices;
  }

  /**
   * @brief Checks if the given coordinates are within the bounds of the field.
   *
   * @param coords The coordinates to check.
   * @return True if the coordinates are within the bounds, false otherwise.
   */
  bool inbounds(const std::array<double, D> &coords) {
    for (size_t i = 0; i < D; i++) {
      if (m_coords_low[i] > coords[i] || coords[i] >= m_coords_high[i])
        return false;
    }
    return true;
  }

  /**
   * @brief Performs nearest neighbor interpolation at the specified coordinates
   * in the field.
   *
   * @param coordinates The coordinates for interpolation.
   * @return A reference to the interpolated element.
   */
  T &interpolate(const std::array<double, D> &coordinates) {
    return get_array()[(map_coordinates_to_indices(coordinates))];
  }

  using FunctionND = std::function<T(std::array<double, D>)>;
  using Function1D = std::function<T(double)>;
  using Function2D = std::function<T(double, double)>;
  using Function3D = std::function<T(double, double, double)>;

  /**
   * @brief Applies the given function to each element of the field.
   *
   * The function must be invocable with std::array<double, D> and return a type
   * convertible to T.
   *
   * @tparam FunctionND The type of the function.
   * @param func The function to apply.
   */
  void apply(FunctionND &&func) {
    static_assert(
        std::is_convertible_v<
            std::invoke_result_t<FunctionND, std::array<double, D>>, T>,
        "Func must be invocable with std::array<double, D> and return a type "
        "convertible to T");
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      element = std::invoke(std::forward<FunctionND>(func),
                            map_indices_to_coordinates(*index_iterator));
      ++index_iterator;
    }
  }

  /**
   * @brief Applies the given 1D function to each element of the field.
   *
   * The function must be invocable with (double) and return a type convertible
   * toT.
   *
   * @tparam Function1D The type of the 1D function.
   * @param func The 1D function to apply.
   */
  void apply(Function1D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto coords = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function1D>(func), coords[0]);
      ++index_iterator;
    }
  }

  /**
   * @brief Applies the given 2D function to each element of the field.
   *
   * The function must be invocable with (double, double) and return a type
   * convertible toT.
   *
   * @tparam Function2D The type of the 2D function.
   * @param func The 2D function to apply.
   */
  void apply(Function2D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto [x, y] = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function2D>(func), x, y);
      ++index_iterator;
    }
  }

  /**
   * @brief Applies the given 3D function to each element of the field.
   *
   * The function must be invocable with (double, double, double) and return a
   * type convertible to T.
   *
   * @tparam Function3D The type of the 3D function.
   * @param func The 3D function to apply.
   */
  void apply(Function3D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto [x, y, z] = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function3D>(func), x, y, z);
      ++index_iterator;
    }
  }

  /**
   * @brief Convert DiscreteField<T, D> to std::vector<T>
   *
   * @return A reference to the underlying data.
   */
  operator std::vector<T> &() { return get_data(); }
  operator std::vector<T> &() const { return get_data(); }

  /**
   * @brief Get the size of the field.
   *
   * @return const std::array<int, D>&
   */
  const std::array<int, D> &get_size() const { return get_index().get_size(); }

  const std::array<int, D> &get_offset() const {
    return get_index().get_begin();
  }

  void set_data(const std::vector<T> &data) { get_array().set_data(data); }

  /**
   * @brief Outputs the discrete field to the specified output stream.
   *
   * @param os The output stream.
   * @param field The discrete field to output.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os,
                                  const DiscreteField<T, D> &field) {
    const Array<T, D> &array = field.get_array();
    const MultiIndex<D> &index = array.get_index();
    os << "DiscreteField<" << TypeName<T>::get() << "," << D
       << ">(begin = " << utils::array_to_string(index.get_begin())
       << ", end = " << utils::array_to_string(index.get_end())
       << ", size = " << utils::array_to_string(index.get_size())
       << ", linear_size = " << index.get_linear_size()
       << ", origin = " << utils::array_to_string(field.get_origin())
       << ", discretization = "
       << utils::array_to_string(field.get_discretization())
       << ", coords_low = " << utils::array_to_string(field.get_coords_low())
       << ", coords_high = " << utils::array_to_string(field.get_coords_high());
    return os;
  }
};

/**
 * @brief Apply function to discrete field.
 *
 * @param field The discrete field.
 * @param func The function to apply.
 *
 */
template <typename T, size_t D, typename Function>
void apply(DiscreteField<T, D> &field, Function &&func) {
  field.apply(std::forward<Function>(func));
}

template <typename T, size_t D> void show(DiscreteField<T, D> &field) {
  utils::show(field.get_array().get_data(), field.get_index().get_size(),
              field.get_index().get_begin());
}

} // namespace pfc

#endif
