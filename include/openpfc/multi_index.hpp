#ifndef PFC_MULTI_INDEX_HPP
#define PFC_MULTI_INDEX_HPP

#include <array>
#include <iostream>
#include <vector>

namespace pfc {

/**
 * @brief MultiIndex class for iterating over multi-dimensional indices.
 *
 * The MultiIndex class provides a convenient way to iterate over multi-dimensional indices.
 * It supports iterating over a range defined by an offset and size in each dimension.
 * The class also provides conversion functions between linear and multi-dimensional indices.
 *
 * @tparam D The number of dimensions.
 */
template <size_t D> class MultiIndex {
private:
  const std::array<int, D> m_begin; ///< The offset of the range in each dimension.
  const std::array<int, D> m_size;  ///< The size of the range in each dimension.
  const std::array<int, D> m_end;   ///< The end index of the range in each dimension.
  const size_t m_linear_begin;      ///< The linear index corresponding to the beginning of the range.
  const size_t m_linear_end;        ///< The linear index corresponding to the end of the range.
  const size_t m_linear_size;       ///< The total number of elements in the range.

  std::array<int, D> calculate_end(std::array<int, D> begin, std::array<int, D> size) {
    std::array<int, D> end;
    for (size_t i = 0; i < D; i++) end[i] = begin[i] + size[i] - 1;
    return end;
  }

public:
  /**
   * @brief Constructs a MultiIndex object with the specified offset and size in each dimension.
   *
   * @param size The size of the range in each dimension.
   * @param offset The offset of the range in each dimension (default: 0)
   */
  MultiIndex(std::array<int, D> size, std::array<int, D> offset = std::array<int, D>())
      : m_begin(offset), m_size(size), m_end(calculate_end(m_begin, m_size)), m_linear_begin(to_linear(m_begin)),
        m_linear_end(to_linear(m_end)), m_linear_size(m_linear_end - m_linear_begin + 1) {}

  const std::array<int, D> &get_begin() const { return m_begin; }
  const std::array<int, D> &get_size() const { return m_size; }
  const std::array<int, D> &get_end() const { return m_end; }
  const size_t &get_linear_begin() const { return m_linear_begin; }
  const size_t &get_linear_size() const { return m_linear_size; }
  const size_t &get_linear_end() const { return m_linear_end; }

  /**
   * @brief Converts a multi-dimensional index to its corresponding linear index.
   *
   * @param indices The multi-dimensional indices.
   * @return The linear index corresponding to the given indices.
   */
  size_t to_linear(const std::array<int, D> &indices) const {
    size_t linear_index = indices[0] - m_begin[0];
    size_t multiplier = 1;
    for (size_t i = 1; i < D; ++i) {
      multiplier *= m_size[i - 1];
      linear_index += (indices[i] - m_begin[i]) * multiplier;
    }
    return linear_index;
  }

  /**
   * @brief Converts a linear index to its corresponding multi-dimensional indices.
   *
   * @param idx The linear index.
   * @return The multi-dimensional indices corresponding to the given linear index.
   */
  std::array<int, D> to_multi(size_t idx) const {
    std::array<int, D> indices;
    for (size_t i = 0; i < D; ++i) {
      indices[i] = (idx % m_size[i]) + m_begin[i];
      idx /= m_size[i];
    }
    return indices;
  }

  /**
   * @brief Checks whether given indices are in bounds or not.
   *
   * @param indices Array of indices.
   * @return true if in bounds, false otherwise
   */
  bool inbounds(const std::array<int, D> &indices) const {
    for (size_t i = 0; i < D; ++i) {
      if (indices[i] < m_begin[i] || indices[i] > m_end[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Outputs the index to the specified output stream.
   *
   * @param os The output stream.
   * @param index The index to output.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const MultiIndex<D> &index) {
    os << "MultiIndex set with " << D << " dimensions (indices from {";
    for (size_t i = 0; i < D - 1; i++) os << index.m_begin[i] << ",";
    os << index.m_begin[D - 1] << "} to {";
    for (size_t i = 0; i < D - 1; i++) os << index.m_end[i] << ",";
    os << index.m_end[D - 1] << "}, size {";
    for (size_t i = 0; i < D - 1; i++) os << index.m_size[i] << ",";
    os << index.m_size[D - 1] << "}, linear size " << index.m_linear_size << ")";
    return os;
  }

  /**
   * @brief Iterator class for iterating over multi-dimensional indices.
   */
  class Iterator {
  private:
    std::array<int, D> m_indices;    ///< Current multi-dimensional indices.
    const MultiIndex &m_multi_index; ///< Reference to the MultiIndex object.
    size_t m_linear_index;           ///< Current linear index.

  public:
    /**
     * @brief Constructs an Iterator object with the specified indices and MultiIndex object.
     *
     * @param indices The initial multi-dimensional indices.
     * @param multi_index Reference to the MultiIndex object.
     */
    Iterator(std::array<int, D> indices, const MultiIndex &multi_index)
        : m_indices(indices), m_multi_index(multi_index) {
      m_linear_index = m_multi_index.to_linear(indices);
    }

    /**
     * @brief Constructs an Iterator object with the specified linear index and MultiIndex object.
     *
     * @param linear_index The initial linear index.
     * @param multi_index Reference to the MultiIndex object.
     */
    Iterator(int linear_index, const MultiIndex &multi_index)
        : m_multi_index(multi_index), m_linear_index(linear_index) {
      m_indices = m_multi_index.to_multi(linear_index);
    }

    /**
     * @brief Dereferences the iterator to obtain the current multi-dimensional indices.
     *
     * @return A reference to the current multi-dimensional indices.
     */
    std::array<int, D> &operator*() { return m_indices; }

    /**
     * @brief Advances the iterator to the next position.
     *
     * @return Reference to the updated iterator.
     */
    Iterator &operator++() {
      m_linear_index++;
      for (size_t i = 0; i < D; i++) {
        m_indices[i]++;
        if (m_indices[i] > m_multi_index.m_end[i]) {
          m_indices[i] = m_multi_index.m_begin[i];
        } else {
          break;
        }
      }
      return *this;
    }

    /**
     * @brief Advances the iterator to the next position (post-increment).
     *
     * @return A copy of the iterator before the increment.
     */
    Iterator operator++(int) {
      Iterator copy(*this);
      ++(*this);
      return copy;
    }

    /**
     * @brief Subtraction operator for the Iterator class.
     *
     * This operator subtracts the specified number of positions from the current iterator position.
     * It returns a new iterator pointing to the element at the updated position.
     *
     * @param n The number of positions to subtract from the current iterator position.
     * @return A new Iterator object pointing to the element at the updated position.
     */
    Iterator operator-(int n) const { return Iterator(m_linear_index - n, m_multi_index); }

    /**
     * @brief Compares two iterators for inequality.
     *
     * @param other The iterator to compare.
     * @return True if the iterators are not equal, false otherwise.
     */
    bool operator!=(const Iterator &other) const { return m_linear_index != other.m_linear_index; }

    /**
     * @brief Compares two iterators for equality.
     *
     * @param other The iterator to compare.
     * @return True if the iterators are equal, false otherwise.
     */
    bool operator==(const Iterator &other) const { return m_linear_index == other.m_linear_index; }

    /**
     * @brief Conversion operator to obtain the current linear index.
     *
     * @return The current linear index.
     */
    operator size_t() const { return m_linear_index; }

    /**
     * @brief Conversion operator to obtain the current multi-dimensional indices.
     *
     * @return The current multi-dimensional indices.
     */
    operator std::array<int, D>() const { return m_indices; }

    /**
     * @brief Returns the current linear index.
     *
     * @return The current linear index.
     */
    size_t get_linear_index() const { return m_linear_index; }

    /**
     * @brief Outputs the current multi-dimensional indices to the specified output stream.
     *
     * @param os The output stream.
     * @param it The iterator to output.
     * @return Reference to the output stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const Iterator &it) {
      os << "{";
      for (size_t i = 0; i < D - 1; i++) os << it.m_indices[i] << ", ";
      os << it.m_indices[D - 1] << "}";
      return os;
    }
  };

  friend class Iterator;

  /**
   * @brief Returns an iterator pointing to the beginning of the range.
   *
   * @return An iterator pointing to the beginning of the range.
   */
  Iterator begin() { return Iterator(m_linear_begin, *this); }

  /**
   * @brief Returns an iterator pointing to the end of the range.
   *
   * @return An iterator pointing to the end of the range.
   */
  Iterator end() { return Iterator(m_linear_end + 1, *this); }

  /**
   * @brief Returns a const iterator pointing to the beginning of the range.
   *
   * @return A const iterator pointing to the beginning of the range.
   */
  Iterator begin() const { return Iterator(m_linear_begin, *this); }

  /**
   * @brief Returns a const iterator pointing to the end of the range.
   *
   * @return A const iterator pointing to the end of the range.
   */
  Iterator end() const { return Iterator(m_linear_end + 1, *this); }

  /**
   * @brief Returns an iterator starting from the specified multi-dimensional indices.
   *
   * @param from The multi-dimensional indices to start from.
   * @return An iterator starting from the specified multi-dimensional indices.
   */
  Iterator from(std::array<int, D> from) { return Iterator(from, *this); }

  /**
   * @brief Returns an iterator starting from the specified linear index.
   *
   * This function creates an iterator starting from the specified linear index. The linear index
   * is converted to the corresponding multi-dimensional indices, and the iterator is constructed
   * using these indices and the current MultiIndex object.
   *
   * @param from The linear index to start from.
   * @return An iterator starting from the specified linear index.
   */
  Iterator from(size_t from) { return Iterator(from, *this); }
};

} // namespace pfc

#endif
