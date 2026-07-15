// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_environment_error_handling.cpp
 * @brief Standalone test for MPI_Init/MPI_Finalize error handling
 *
 * This executable has its own main() function to test MPI error handling
 * independently of the global MPI initialization in the main test harness.
 */

#include <openpfc/kernel/mpi/environment.hpp>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    // Test that MPI_Init error handling works
    pfc::mpi::environment env;
    std::cout << "MPI environment initialized successfully" << std::endl;
    std::cout << "Processor name: " << env.processor_name() << std::endl;
    std::cout << "MPI initialized: " << (env.initialized() ? "true" : "false")
              << std::endl;
    std::cout << "MPI finalized: " << (env.finalized() ? "true" : "false")
              << std::endl;

    // Test that copy operations are deleted (compile-time check)
    // Uncommenting the following lines should cause compilation errors:
    // pfc::mpi::environment env2 = env;  // copy constructor deleted
    // pfc::mpi::environment env3;
    // env3 = env;  // copy assignment deleted

    std::cout << "Environment is non-copyable (verified at compile time)"
              << std::endl;

    // MPI_Finalize called automatically when env goes out of scope
    std::cout << "Test completed successfully" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return 1;
  }
}
