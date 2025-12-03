// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_all.hpp
 * @brief Convenience header including all SparseVector functionality
 *
 * Includes:
 * - sparse_vector.hpp: Core SparseVector class
 * - sparse_vector_ops.hpp: Gather and scatter operations
 * - exchange.hpp: MPI exchange operations
 *
 * @code
 * #include <openpfc/core/sparse_vector_all.hpp>
 *
 * // Now you have access to:
 * // - core::SparseVector
 * // - gather, scatter
 * // - exchange::send, exchange::receive
 * // - exchange::send_data, exchange::receive_data
 * @endcode
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <openpfc/core/exchange.hpp>
#include <openpfc/core/sparse_vector.hpp>
#include <openpfc/core/sparse_vector_ops.hpp>
