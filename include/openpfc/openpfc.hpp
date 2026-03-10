// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file openpfc.hpp
 * @brief Main convenience header that includes all OpenPFC public API components
 *
 * @details
 * This is the primary include file for OpenPFC users. Including this single
 * header provides access to all public API components:
 * - Kernel: data (World, Field, etc.), decomposition, execution, FFT, simulation
 * - Runtime: MPI, backends (via kernel interfaces)
 * - Frontend: utils, UI (when included)
 *
 * For faster compilation times, include specific headers from kernel/, runtime/, or frontend/.
 *
 * @see docs/architecture.md for package structure (kernel / runtime / frontend)
 * @see kernel/data/world.hpp for computational domain setup
 * @see kernel/simulation/model.hpp for physics model implementation
 * @see kernel/simulation/simulator.hpp for running simulations
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_OPENPFC_HPP
#define PFC_OPENPFC_HPP

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
#include "kernel/simulation/field_modifier.hpp"
#include "kernel/simulation/model.hpp"
#include "kernel/simulation/results_writer.hpp"
#include "kernel/simulation/simulator.hpp"
#include "kernel/simulation/time.hpp"
#include "kernel/simulation/boundary_conditions/fixed_bc.hpp"
#include "kernel/simulation/boundary_conditions/moving_bc.hpp"
#include "kernel/simulation/initial_conditions/constant.hpp"
#include "kernel/simulation/initial_conditions/file_reader.hpp"
#include "kernel/simulation/initial_conditions/random_seeds.hpp"
#include "kernel/simulation/initial_conditions/seed.hpp"
#include "kernel/simulation/initial_conditions/seed_grid.hpp"
#include "kernel/simulation/initial_conditions/single_seed.hpp"
#include "frontend/utils/utils.hpp"
#include "frontend/utils/show.hpp"

#endif // PFC_OPENPFC_HPP
