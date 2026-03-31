// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file array_to_string.hpp
 * @brief Convert std::array to string representation
 *
 * @details
 * Implementation lives in kernel/detail; this header re-exports under
 * pfc::utils for compatibility.
 */

#ifndef PFC_UTILS_ARRAY_TO_STRING_HPP
#define PFC_UTILS_ARRAY_TO_STRING_HPP

#include <openpfc/kernel/detail/array_format.hpp>

namespace pfc::utils {

using detail::array_to_string;

} // namespace pfc::utils

#endif
