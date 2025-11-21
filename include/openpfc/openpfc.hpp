// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file openpfc.hpp
 * @brief Main convenience header that includes all OpenPFC public API components
 *
 * @details
 * This is the primary include file for OpenPFC users. Including this single
 * header provides access to all public API components:
 * - Core infrastructure (World, Decomposition, FFT)
 * - Physics models (Model base class)
 * - Simulation orchestration (Simulator, Time)
 * - Initial conditions (Constant, Seed, FileReader, RandomSeeds, SeedGrid, etc.)
 * - Boundary conditions (FixedBC, MovingBC)
 * - I/O (ResultsWriter, BinaryReader)
 * - Utilities (Array, MultiIndex, DiscreteField, utils)
 *
 * Most users should start with this header for maximum convenience:
 * @code
 * #include <openpfc/openpfc.hpp>
 *
 * int main(int argc, char** argv) {
 *     pfc::mpi::Environment env(argc, argv);
 *     // Your simulation code here
 * }
 * @endcode
 *
 * For faster compilation times, advanced users can include specific headers:
 * @code
 * #include <openpfc/core/world.hpp>
 * #include <openpfc/model.hpp>
 * #include <openpfc/simulator.hpp>
 * @endcode
 *
 * This file is part of the Core API module, serving as the main entry point
 * for all OpenPFC functionality.
 *
 * @see core/world.hpp for computational domain setup
 * @see model.hpp for physics model implementation
 * @see simulator.hpp for running simulations
 * @see docs/getting_started/ for tutorials and examples
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_OPENPFC_HPP
#define PFC_OPENPFC_HPP

#include "array.hpp"
#include "binary_reader.hpp"
#include "boundary_conditions/fixed_bc.hpp"
#include "boundary_conditions/moving_bc.hpp"
#include "constants.hpp"
#include "core/decomposition.hpp"
#include "core/world.hpp"
#include "discrete_field.hpp"
#include "factory/decomposition_factory.hpp"
#include "fft.hpp"
#include "field_modifier.hpp"
#include "initial_conditions/constant.hpp"
#include "initial_conditions/file_reader.hpp"
#include "initial_conditions/random_seeds.hpp"
#include "initial_conditions/seed.hpp"
#include "initial_conditions/seed_grid.hpp"
#include "initial_conditions/single_seed.hpp"
#include "model.hpp"
#include "mpi.hpp"
#include "multi_index.hpp"
#include "results_writer.hpp"
#include "simulator.hpp"
#include "time.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "utils/show.hpp"

#endif // PFC_OPENPFC_HPP
