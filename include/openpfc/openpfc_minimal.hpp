// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file openpfc_minimal.hpp
 * @brief Minimal convenience header: kernel and runtime only (no frontend)
 *
 * @details
 * Use this for minimal applications that do not need frontend features (UI,
 * logging, extra I/O helpers, binary_writer, utils). Includes kernel and
 * runtime public API only. For the full API including frontend, use
 * openpfc.hpp.
 *
 * @see openpfc.hpp for full API including frontend
 * @see docs/architecture.md for package structure
 */

#ifndef PFC_OPENPFC_MINIMAL_HPP
#define PFC_OPENPFC_MINIMAL_HPP

#include "kernel/data/array.hpp"
#include "kernel/data/constants.hpp"
#include "kernel/data/discrete_field.hpp"
#include "kernel/data/model_types.hpp"
#include "kernel/data/multi_index.hpp"
#include "kernel/data/world.hpp"
#include "kernel/decomposition/decomposition.hpp"
#include "kernel/decomposition/decomposition_factory.hpp"
#include "kernel/fft/fft.hpp"
#include "kernel/mpi/mpi.hpp"
#include "kernel/simulation/binary_reader.hpp"
#include "kernel/simulation/boundary_conditions/fixed_bc.hpp"
#include "kernel/simulation/boundary_conditions/moving_bc.hpp"
#include "kernel/simulation/field_modifier.hpp"
#include "kernel/simulation/initial_conditions/constant.hpp"
#include "kernel/simulation/initial_conditions/file_reader.hpp"
#include "kernel/simulation/initial_conditions/random_seeds.hpp"
#include "kernel/simulation/initial_conditions/seed.hpp"
#include "kernel/simulation/initial_conditions/seed_grid.hpp"
#include "kernel/simulation/initial_conditions/single_seed.hpp"
#include "kernel/simulation/model.hpp"
#include "kernel/simulation/results_writer.hpp"
#include "kernel/simulation/simulator.hpp"
#include "kernel/simulation/time.hpp"

#endif // PFC_OPENPFC_MINIMAL_HPP
