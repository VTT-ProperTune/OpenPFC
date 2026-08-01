// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file cuda_span_compat.hpp
 * @brief Minimal span polyfill for CUDA compatibility (C++17)
 *
 * This provides basic span functionality for CUDA compilation when std::span
 * is not available (C++20 only). Only implements what's needed by databuffer.hpp
 */

#pragma once

#include <cstddef>
#include <type_traits>

namespace std {

template<typename T>
class cuda_span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using index_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    constexpr cuda_span() noexcept : data_(nullptr), size_(0) {}
    constexpr cuda_span(pointer ptr, index_type count) noexcept : data_(ptr), size_(count) {}
    constexpr cuda_span(pointer first, pointer last) noexcept : data_(first), size_(static_cast<index_type>(last - first)) {}

    template<size_t N>
    constexpr cuda_span(element_type (&arr)[N]) noexcept : data_(arr), size_(N) {}

    [[nodiscard]] constexpr pointer data() const noexcept { return data_; }
    [[nodiscard]] constexpr index_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] constexpr reference operator[](index_type idx) const { return data_[idx]; }
    [[nodiscard]] constexpr reference front() const { return data_[0]; }
    [[nodiscard]] constexpr reference back() const { return data_[size_ - 1]; }

    [[nodiscard]] constexpr pointer begin() const noexcept { return data_; }
    [[nodiscard]] constexpr pointer end() const noexcept { return data_ + size_; }

private:
    pointer data_;
    index_type size_;
};

} // namespace std

// For compatibility: alias cuda_span to span when std::span is not available
#if __cplusplus < 202002L
namespace std {
    template<typename T>
    using span = cuda_span<T>;
}
#endif