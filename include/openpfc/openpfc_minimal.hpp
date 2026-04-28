// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file openpfc_minimal.hpp
 * @brief Minimal convenience header: kernel and minimal runtime (no frontend)
 *
 * @details
 * Use this for minimal applications that do not need frontend features (UI,
 * logging, extra I/O helpers, binary_writer, utils). Includes kernel and a
 * minimal runtime set (HeFFTe adapter for FFT/decomposition). For CUDA/HIP,
 * include the corresponding runtime headers. For the full API including
 * frontend, use openpfc.hpp.
 *
 * @see openpfc.hpp for full API including frontend
 * @see docs/architecture.md for package structure
 */

#ifndef PFC_OPENPFC_MINIMAL_HPP
#define PFC_OPENPFC_MINIMAL_HPP

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
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

// Minimal runtime (CPU FFT uses HeFFTe; adapter shared by decomposition/FFT)
#include <openpfc/runtime/common/heffte_adapter.hpp>

#endif // PFC_OPENPFC_MINIMAL_HPP
