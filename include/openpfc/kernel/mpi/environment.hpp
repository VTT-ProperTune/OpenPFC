// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file environment.hpp
 * @brief RAII wrapper for MPI initialization and finalization
 *
 * @details
 * This header provides the environment class, which manages MPI
 * initialization (MPI_Init) and finalization (MPI_Finalize) using
 * RAII (Resource Acquisition Is Initialization) pattern.
 *
 * The environment class ensures:
 * - MPI_Init is called on construction
 * - MPI_Finalize is called on destruction
 * - Query MPI state (initialized(), finalized())
 * - Get processor name
 *
 * @code
 * #include <openpfc/kernel/mpi/environment.hpp>
 *
 * int main(int argc, char** argv) {
 *     pfc::mpi::environment env;
 *     std::cout << "Running on: " << env.processor_name() << std::endl;
 *     // MPI_Finalize called automatically when env goes out of scope
 *     return 0;
 * }
 * @endcode
 *
 * @see mpi/communicator.hpp for communicator wrapper
 * @see mpi.hpp for top-level MPI utilities
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_MPI_ENVIRONMENT_HPP
#define PFC_MPI_ENVIRONMENT_HPP

#include <array>
#include <mpi.h>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <string>

namespace pfc::mpi {

class environment {
public:
  environment(const environment &) = delete;
  environment &operator=(const environment &) = delete;
  environment(environment &&) = delete;
  environment &operator=(environment &&) = delete;

  inline environment();
  inline ~environment() noexcept;
  static inline std::string processor_name();
  static inline bool initialized();
  static inline bool finalized();
};

inline environment::environment() {
  int err = MPI_Init(nullptr, nullptr);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Init");
}

inline environment::~environment() noexcept {
  // Fail-closed cleanup policy: never throw from a destructor (would risk
  // std::terminate during unwinding). Log + MPI_Abort on a finalize error.
  pfc::mpi::abort_on_mpi_error(MPI_Finalize(), "MPI_Finalize");
}

inline std::string environment::processor_name() {
  std::array<char, MPI_MAX_PROCESSOR_NAME> name{};
  int resultlen = 0;
  MPI_Get_processor_name(name.data(), &resultlen);
  return {name.data(), static_cast<std::size_t>(resultlen)};
}

inline bool environment::initialized() {
  int flag = 0;
  MPI_Initialized(&flag);
  return {flag != 0};
}

inline bool environment::finalized() {
  int flag = 0;
  MPI_Finalized(&flag);
  return {flag != 0};
}

} // namespace pfc::mpi

#endif // PFC_MPI_ENVIRONMENT_HPP
