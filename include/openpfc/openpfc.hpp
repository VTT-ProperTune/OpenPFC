// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
 * For faster compilation times, include specific headers from kernel/, runtime/, or
 * frontend/.
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

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/utils/show.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/data/array.hpp>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/discrete_field.hpp>
#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/data/multi_index.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>
#include <openpfc/kernel/simulation/boundary_conditions/fixed_bc.hpp>
#include <openpfc/kernel/simulation/boundary_conditions/moving_bc.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/initial_conditions/file_reader.hpp>
#include <openpfc/kernel/simulation/initial_conditions/random_seeds.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed_grid.hpp>
#include <openpfc/kernel/simulation/initial_conditions/single_seed.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

#endif // PFC_OPENPFC_HPP
