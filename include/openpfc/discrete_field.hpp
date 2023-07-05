#ifndef PFC_DISCRETE_FIELD_HPP
#define PFC_DISCRETE_FIELD_HPP

#include "utils.hpp"
#include "utils/array.hpp"
#include "utils/typename.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>

namespace pfc {
namespace utils {

template <typename T, size_t D> class DiscreteField : public Array<T, D> {
private:
  std::array<double, D> m_origin;
  std::array<double, D> m_discretization;
  std::array<double, D> m_coords_low;
  std::array<double, D> m_coords_high;

public:
  DiscreteField(const std::array<int, D> &dimensions, const std::array<int, D> &offsets,
                const std::array<double, D> &origin, const std::array<double, D> &discretization)
      : utils::Array<T, D>(dimensions, offsets), m_origin(origin), m_discretization(discretization) {
    for (size_t i = 0; i < D; i++) {
      m_coords_low[i] = m_origin[i] + offsets[i] * m_discretization[i];
      m_coords_high[i] = m_origin[i] + (offsets[i] + dimensions[i]) * m_discretization[i];
    }
  }

  using Array<T, D>::get_index;
  using Array<T, D>::get_data;

  std::array<double, D> map_indices_to_coordinates(const std::array<int, D> &indices) const {
    std::array<double, D> coordinates;
    for (size_t i = 0; i < D; ++i) {
      coordinates[i] = m_origin[i] + indices[i] * m_discretization[i];
    }
    return coordinates;
  }

  std::array<int, D> map_coordinates_to_indices(const std::array<double, D> &coordinates) const {
    std::array<int, D> indices;
    for (size_t i = 0; i < D; ++i) {
      indices[i] = static_cast<int>(std::round((coordinates[i] - m_origin[i]) / m_discretization[i]));
    }
    return indices;
  }

  bool inbounds(const std::array<double, D> &coords) {
    for (size_t i = 0; i < D; i++) {
      // if (!(m_coords_low[i] <= coords[i] < m_coords_high[i])) return false;
      if (m_coords_low[i] > coords[i] || coords[i] >= m_coords_high[i]) return false;
    }
    return true;
  }

  T &interpolate(const std::array<double, D> &coordinates) {
    return Array<T, D>::operator[](map_coordinates_to_indices(coordinates));
  }

  using FunctionND = std::function<T(std::array<double, D>)>;

  void apply(FunctionND &&func) {
    static_assert(std::is_convertible_v<std::invoke_result_t<FunctionND, std::array<double, D>>, T>,
                  "Func must be invocable with std::array<double, D> and return a type convertible to T");
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      element = std::invoke(std::forward<FunctionND>(func), map_indices_to_coordinates(*index_iterator));
      ++index_iterator;
    }
  }

  using Function2D = std::function<T(double, double)>;

  void apply(Function2D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto [x, y] = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function2D>(func), x, y);
      ++index_iterator;
    }
  }

  using Function3D = std::function<T(double, double, double)>;

  void apply(Function3D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto [x, y, z] = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function3D>(func), x, y, z);
      ++index_iterator;
    }
  }

  /**
   * @brief Outputs the discrete field to the specified output stream.
   *
   * @param os The output stream.
   * @param field The discrete field to output.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const DiscreteField<T, D> &field) {
    auto begin = field.get_index().get_offset();
    auto size = field.get_index().get_size();
    decltype(begin) end;
    size_t linear_size = 1;
    for (size_t i = 0; i < D; i++) {
      end[i] = begin[i] + size[i] - 1;
      linear_size *= size[i];
    }
    os << "DiscreteField<" << TypeName<T>::get() << "," << D << ">(begin = " << array_to_string(begin)
       << ", end = " << array_to_string(end) << ", size = " << array_to_string(size)
       << ", linear_size = " << linear_size << ", origin = " << array_to_string(field.m_origin)
       << ", discretization = " << array_to_string(field.m_discretization)
       << ", coords_low = " << array_to_string(field.m_coords_low)
       << ", coords_high = " << array_to_string(field.m_coords_high);
    return os;
  }
};

} // namespace utils
} // namespace pfc

#endif
