#include "multi_index.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

namespace pfc {
namespace utils {

template <typename T, size_t D> class array {
protected:
  MultiIndex<D> index;
  std::vector<T> data;

public:
  array(const std::array<int, D> &dimensions, const std::array<int, D> &offsets = {0}) : index(dimensions, offsets) {
    int totalSize = 1;
    for (int dim : index.get_size()) {
      totalSize *= dim;
    }
    data.resize(totalSize);
  }

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
};

} // namespace utils
} // namespace pfc
