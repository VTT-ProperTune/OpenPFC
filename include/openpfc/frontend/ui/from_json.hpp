// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/from_json.hpp
 * @brief Umbrella header for JSON deserialization of OpenPFC UI types
 *
 * @details
 * Implementation is split for readability and faster incremental builds:
 * - `from_json_fwd.hpp` — primary `from_json<T>` declaration
 * - `from_json_log.hpp` — `set_from_json_log_rank` / `get_from_json_log_rank`
 *   and loggers used by parsers
 * - `from_json_heffte.hpp` — HeFFTe `plan_options` overlay and specialization
 * - `from_json_fft_backend.hpp` — `fft::Backend` specialization
 * - `from_json_world_time.hpp` — `World` and `Time` specializations
 * - `from_json_field_modifiers.hpp` — built-in IC/BC `from_json` overloads and
 *   base `Model` params stub
 *
 * Include this header for the full surface area (same as before the split).
 * GPU stack helpers that only need the HeFFTe overlay may include
 * `from_json_heffte.hpp` directly.
 *
 * Logging from these parsers uses `set_from_json_log_rank` /
 * `get_from_json_log_rank` so messages can include an MPI rank prefix; `App::main`
 * sets the rank at startup.
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_FROM_JSON_HPP
#define PFC_UI_FROM_JSON_HPP

#include <openpfc/frontend/ui/from_json_fft_backend.hpp>
#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/frontend/ui/from_json_heffte.hpp>
#include <openpfc/frontend/ui/from_json_log.hpp>
#include <openpfc/frontend/ui/from_json_world_time.hpp>

#endif // PFC_UI_FROM_JSON_HPP
