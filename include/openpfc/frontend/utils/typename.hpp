// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file typename.hpp
 * @brief Get human-readable type names at runtime
 *
 * @details
 * Implementation lives in kernel/detail; this header re-exports the API for
 * frontend and legacy includes.
 *
 * @code
 * #include <openpfc/frontend/utils/typename.hpp>
 * std::cout << pfc::TypeName<double>::get() << std::endl;
 * @endcode
 */

#ifndef PFC_TYPENAME_HPP
#define PFC_TYPENAME_HPP

#include <openpfc/kernel/detail/typename.hpp>

#endif
