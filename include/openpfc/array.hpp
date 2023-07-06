#ifndef PFC_ARRAY_HPP
#define PFC_ARRAY_HPP

#include "multi_index.hpp"
#include "utils/typename.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <typeinfo>
#include <vector>

namespace pfc {

template <typename T, size_t D> class Array {

protected:
  MultiIndex<D> index;
  std::vector<T> data;

public:
  Array(const std::array<int, D> &dimensions, const std::array<int, D> &offsets = {0}) : index(dimensions, offsets) {
    int totalSize = 1;
    for (int dim : index.get_size()) totalSize *= dim;
    data.resize(totalSize);
  }

  typename std::vector<T>::iterator begin() { return data.begin(); }
  typename std::vector<T>::iterator end() { return data.end(); }
  typename std::vector<T>::const_iterator begin() const { return data.begin(); }
  typename std::vector<T>::const_iterator end() const { return data.end(); }

  const MultiIndex<D> &get_index() const { return index; }

  std::vector<T> &get_data() { return data; }

  T &operator[](const std::array<int, D> &indices) { return operator[](index.to_linear(indices)); }

  T &operator[](int idx) { return data.operator[](idx); }

  T &operator()(const std::array<int, D> &indices) { return operator[](indices); }

  std::array<int, D> get_size() const { return index.get_size(); }

  std::array<int, D> get_offset() const { return index.get_offset(); }

  bool inbounds(const std::array<int, D> &indices) { return index.inbounds(indices); }

  template <typename Func> void apply(Func &&func) {
    static_assert(std::is_convertible_v<std::invoke_result_t<Func, std::array<int, D>>, T>,
                  "Func must be invocable with std::array<int, D> and return a type convertible to T");
    auto it = index.begin();
    for (T &element : get_data()) element = std::invoke(std::forward<Func>(func), *(it++));
  }

  /**
   * @brief Outputs the array to the specified output stream.
   *
   * @param os The output stream.
   * @param array The array to output.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const Array<T, D> &array) {
    auto begin = array.get_index().get_offset();
    auto size = array.get_index().get_size();
    decltype(begin) end;
    size_t linear_size = 1;
    for (size_t i = 0; i < D; i++) {
      end[i] = begin[i] + size[i] - 1;
      linear_size *= size[i];
    }
    os << "Array with " << D << " dimensions and type " << TypeName<T>::get() << " (indices from {";
    for (size_t i = 0; i < D - 1; i++) os << begin[i] << ",";
    os << begin[D - 1] << "} to {";
    for (size_t i = 0; i < D - 1; i++) os << end[i] << ",";
    os << end[D - 1] << "}, size {";
    for (size_t i = 0; i < D - 1; i++) os << size[i] << ",";
    os << size[D - 1] << "}, linear size " << linear_size << ")";
    return os;
  }
};

} // namespace pfc

#endif
