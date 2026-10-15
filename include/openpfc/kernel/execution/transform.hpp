// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file transform.hpp
 * @brief Elementwise `out[i] = f(a[i], b[i], ...)` over one to four buffers.
 *
 * @details
 * `transform(out, f, a, ...)` applies a user functor at every index of
 * `out`. Inputs and output may have different element types (real or
 * `std::complex`), and every input must have the size of `out`. The output
 * may alias an input: element `i` is read before it is written.
 *
 * This header defines the host (`CPUTag`) path. The CUDA and HIP overloads in
 * `openpfc/runtime/gpu/transform_gpu.hpp` take the same arguments and run the
 * same functor in a device kernel, so the functor must be trivially copyable,
 * carry its constants by value, and mark its call operator `OPENPFC_HD`
 * (`kernel/data/host_device.hpp`):
 *
 * @code
 * struct Reaction {
 *   double l1, l2, c;
 *   OPENPFC_HD double operator()(double grad, double phi, double phi0) const {
 *     return grad + l1 * (phi - phi0) + l2 * phi * (phi * phi - 1.0) + c;
 *   }
 * };
 * pfc::core::transform(out, Reaction{0.1, 0.5, 0.0}, grad, phi, phi0);
 * @endcode
 */

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>

namespace pfc::core {

/**
 * @brief A functor `transform` can run on any backend: trivially copyable
 *        and callable with one element of each input, returning a value
 *        convertible to the output element type.
 */
template <typename F, typename Out, typename... In>
concept ElementwiseFunctor =
    (sizeof...(In) >= 1 && sizeof...(In) <= 4) && std::is_trivially_copyable_v<F> &&
    std::invocable<const F &, const In &...> &&
    std::convertible_to<std::invoke_result_t<const F &, const In &...>, Out>;

namespace detail {

inline void check_transform_sizes(std::size_t out_size,
                                  std::initializer_list<std::size_t> in_sizes) {
  std::size_t arg = 0;
  for (const std::size_t n : in_sizes) {
    ++arg;
    if (n != out_size) {
      throw std::invalid_argument(
          "transform: input " + std::to_string(arg) + " has " + std::to_string(n) +
          " elements but out has " + std::to_string(out_size));
    }
  }
}

} // namespace detail

/// Host `out[i] = f(in[i]...)` for every `i` in `[0, out.size())`.
template <typename Out, typename F, typename... In>
  requires ElementwiseFunctor<F, Out, In...>
void transform(DataBuffer<backend::CPUTag, Out> &out, F f,
               const DataBuffer<backend::CPUTag, In> &...in) {
  detail::check_transform_sizes(out.size(), {in.size()...});
  Out *o = out.data();
  const std::size_t n = out.size();
  for (std::size_t i = 0; i < n; ++i) {
    o[i] = static_cast<Out>(f(in.data()[i]...));
  }
}

} // namespace pfc::core
