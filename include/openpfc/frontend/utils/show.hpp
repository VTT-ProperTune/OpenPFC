// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file show.hpp
 * @brief Pretty-print 2D/3D arrays to console
 *
 * @details
 * Implementation lives in kernel/detail; this header re-exports under
 * pfc::utils for compatibility.
 */

#ifndef PFC_UTILS_SHOW
#define PFC_UTILS_SHOW

#include <openpfc/kernel/detail/array_format.hpp>

namespace pfc {
namespace utils {

using detail::show;

} // namespace utils
} // namespace pfc

#endif
