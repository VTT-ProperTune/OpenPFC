// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_ARRAY_HPP
#define PFC_ARRAY_HPP

#include "core/decomposition.hpp"
#include "fft.hpp"
#include "multi_index.hpp"
#include "utils/array_to_string.hpp"
#include "utils/show.hpp"
#include "utils/typename.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace pfc {

template <typename T, size_t D> class Array {

private:
  const MultiIndex<D> index;
  std::vector<T> data;

  /**
   * @brief Construct a new Array object from FFT, using outbox, if
   * is_complex<T> = true.
   *
   * @param fft
   */
  Array(const FFT &fft, std::true_type)
      : index(get_outbox(fft).size, get_outbox(fft).low) {}

  /**
   * @brief Construct a new Array object from FFT, using inbox, if
   * is_complex<T> = false.
   *
   * @param decomp
   */
  Array(const FFT &fft, std::false_type)
      : index(get_inbox(fft).size, get_inbox(fft).low) {}

  // Custom type trait to check if a type is complex
  template <typename U> struct is_complex : std::false_type {};
  template <typename U> struct is_complex<std::complex<U>> : std::true_type {};

public:
  /**
   * @brief Constructs an Array object with the specified dimensions and
   * offsets.
   *
   * @param dimensions The dimensions of the array.
   * @param offsets The offsets of the array.
   */
  Array(const std::array<int, D> &dimensions,
        const std::array<int, D> &offsets = {0})
      : index(dimensions, offsets) {
    data.resize(index.get_linear_size());
  }

  /**
   * @brief Constructs an Array object from Decomposition object. Array
   * dimension and offset depends from the type T of array. If the type of array
   * is double, i.e. T = double, then inbox_size and inbox_offset is used. If
   * the type of array is complex, i.e. T = std::complex<double>, then
   * outbox_size and oubox_offset is used.
   *
   * @param decomp The Decomposition object.
   */
  Array(const Decomposition &decomp) : Array(decomp, is_complex<T>()) {
    data.resize(index.get_linear_size());
  }

  typename std::vector<T>::iterator begin() { return data.begin(); }
  typename std::vector<T>::iterator end() { return data.end(); }
  typename std::vector<T>::const_iterator begin() const { return data.begin(); }
  typename std::vector<T>::const_iterator end() const { return data.end(); }

  /**
   * @brief Get the index object
   *
   * @return const MultiIndex<D>&
   */
  const MultiIndex<D> &get_index() const { return index; }

  /**
   * @brief Get the data object
   *
   * @return std::vector<T>&
   */
  std::vector<T> &get_data() { return data; }

  T &operator[](const std::array<int, D> &indices) {
    return operator[](index.to_linear(indices));
  }

  T &operator[](int idx) { return data.operator[](idx); }

  T &operator()(const std::array<int, D> &indices) { return operator[](indices); }

  /**
   * @brief Get the size object
   *
   * @return std::array<int, D>
   */
  std::array<int, D> get_size() const { return index.get_size(); }

  /**
   * @brief Get the offset object
   *
   * @return std::array<int, D>
   */
  std::array<int, D> get_offset() const { return index.get_offset(); }

  /**
   * @brief Checks if the specified indices are in bounds.
   *
   * @param indices The indices to check.
   * @return true
   * @return false
   */
  bool inbounds(const std::array<int, D> &indices) {
    return index.inbounds(indices);
  }

  /**
   * @brief Applies the specified function to each element of the array.
   *
   * @tparam Func
   * @param func A function that takes std::array<int, D> as an argument and
   * returns a type convertible to T.
   */
  template <typename Func> void apply(Func &&func) {
    static_assert(
        std::is_convertible_v<std::invoke_result_t<Func, std::array<int, D>>, T>,
        "Func must be invocable with std::array<int, D> and return a type "
        "convertible to T");
    auto it = index.begin();
    for (T &element : get_data())
      element = std::invoke(std::forward<Func>(func), *(it++));
  }

  /**
   * @brief Convert Array<T, D> to std::vector<T>.
   *
   * @return A reference to underlying data.
   */
  operator std::vector<T> &() { return data; }

  void set_data(const std::vector<T> &data) {
    if (data.size() != index.get_linear_size()) {
      throw std::runtime_error("Dimension mismatch, set_data failed");
    }
    this->data = std::move(data);
  }

  /**
   * @brief Outputs the array to the specified output stream.
   *
   * @param os The output stream.
   * @param array The array to output.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const Array<T, D> &array) {
    const MultiIndex<D> &index = array.get_index();
    os << "Array<" << TypeName<T>::get() << "," << D
       << ">(begin = " << utils::array_to_string(index.get_begin())
       << ", end = " << utils::array_to_string(index.get_end())
       << ", size = " << utils::array_to_string(index.get_size())
       << ", linear_size = " << index.get_linear_size() << ")";
    return os;
  }
};

template <typename T, size_t D> void show(Array<T, D> &array) {
  utils::show(array.get_data(), array.get_index().get_size(),
              array.get_index().get_begin());
}

} // namespace pfc

#endif
