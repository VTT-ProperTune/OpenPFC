// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_reporter.hpp
 * @brief Memory usage reporting utilities for parallel simulations
 *
 * @details
 * This header provides utilities for reporting memory allocation in MPI-parallel
 * phase-field simulations. It helps track and analyze memory usage across:
 * - User application data (model fields, operators)
 * - FFT library workspace (HeFFTe allocations)
 * - System-wide memory statistics
 *
 * The reporter provides per-rank, total, and per-voxel memory metrics,
 * helping users understand memory scaling and optimize domain decomposition.
 *
 * ## Key Features
 * - Separate reporting of application vs FFT memory
 * - Global memory aggregation via MPI reduction
 * - Per-voxel memory calculation for scalability analysis
 * - Optional system memory detection (total available RAM)
 * - Logging via OpenPFC's core logger
 *
 * ## Usage Pattern
 * @code
 * #include <openpfc/utils/memory_reporter.hpp>
 *
 * // After model initialization
 * size_t model_mem = model.get_allocated_memory_bytes();
 * size_t fft_mem = fft.get_allocated_memory_bytes();
 *
 * pfc::utils::MemoryUsage usage{model_mem, fft_mem};
 * pfc::utils::report_memory_usage(usage, world, logger, MPI_COMM_WORLD);
 * @endcode
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include "openpfc/core/world.hpp"
#include "openpfc/core/world_queries.hpp"
#include "openpfc/logging.hpp"
#include "openpfc/mpi.hpp"
#include "openpfc/utils.hpp"
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <sstream>
#include <string>

namespace pfc {
namespace utils {

/**
 * @brief Memory usage statistics for a single MPI rank
 *
 * Holds memory allocation data separated into:
 * - Application memory: User-defined fields, operators, auxiliary arrays
 * - FFT memory: Workspace allocated by HeFFTe for transform operations
 */
struct MemoryUsage {
  size_t m_application_bytes = 0; ///< Memory allocated by user application (bytes)
  size_t m_fft_bytes = 0;         ///< Memory allocated by FFT library (bytes)

  /**
   * @brief Calculate total memory usage
   * @return Sum of application and FFT memory in bytes
   */
  size_t total_bytes() const noexcept { return m_application_bytes + m_fft_bytes; }
};

/**
 * @brief Attempt to read total system memory from /proc/meminfo
 *
 * Reads the MemTotal field from /proc/meminfo (Linux systems).
 * Returns 0 if file cannot be read or parsed.
 *
 * @return Total system memory in bytes, or 0 if unavailable
 *
 * @note Only works on Linux systems with /proc filesystem
 * @note Returns physical RAM, not necessarily all available to MPI
 */
inline size_t get_system_memory_bytes() noexcept {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo) return 0;

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.find("MemTotal:") == 0) {
      // Format: "MemTotal:       16384000 kB"
      std::istringstream iss(line);
      std::string label;
      size_t mem_kb;
      iss >> label >> mem_kb;
      return mem_kb * 1024; // Convert kB to bytes
    }
  }
  return 0;
}

/**
 * @brief Format bytes as human-readable string (KB, MB, GB, TB)
 *
 * @param bytes Memory size in bytes
 * @return Formatted string (e.g., "1.23 GB")
 */
inline std::string format_bytes(size_t bytes) {
  const double KB = 1024.0;
  const double MB = KB * 1024.0;
  const double GB = MB * 1024.0;
  const double TB = GB * 1024.0;

  std::ostringstream oss;
  oss.precision(2);
  oss << std::fixed;

  if (bytes >= TB) {
    oss << (bytes / TB) << " TB";
  } else if (bytes >= GB) {
    oss << (bytes / GB) << " GB";
  } else if (bytes >= MB) {
    oss << (bytes / MB) << " MB";
  } else if (bytes >= KB) {
    oss << (bytes / KB) << " KB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

/**
 * @brief Report memory usage to logger
 *
 * Logs detailed memory statistics including:
 * - Per-rank memory (application + FFT breakdown)
 * - Total memory across all ranks
 * - Memory per voxel (for scalability analysis)
 * - System memory percentage (if available)
 *
 * Only rank 0 performs logging after MPI reduction of memory data.
 *
 * @param usage Memory usage for this rank
 * @param world Simulation domain (for voxel count)
 * @param logger OpenPFC logger instance
 * @param comm MPI communicator for global reduction
 *
 * @note MPI collective operation - all ranks must call
 *
 * @example
 * ```cpp
 * pfc::Logger logger{pfc::LogLevel::Info, rank};
 * pfc::utils::MemoryUsage usage{model_memory, fft_memory};
 * pfc::utils::report_memory_usage(usage, world, logger, MPI_COMM_WORLD);
 * ```
 *
 * Example output:
 * ```
 * [INFO] Memory Usage Report:
 * [INFO]   Rank 0 - Application: 1.23 GB, FFT: 456.78 MB, Total: 1.68 GB
 * [INFO]   Global Total: 6.72 GB (4 ranks)
 * [INFO]   Per Voxel: 1.57 KB/voxel (4194304 voxels)
 * [INFO]   System Memory: 64.00 GB, Usage: 10.5%
 * ```
 */
template <typename WorldType>
inline void report_memory_usage(const MemoryUsage &usage, const WorldType &world,
                                const Logger &logger,
                                MPI_Comm comm = MPI_COMM_WORLD) {
  int rank = pfc::mpi::get_comm_rank(comm);
  int num_ranks = pfc::mpi::get_comm_size(comm);

  // Gather memory data to rank 0
  size_t local_mem[3] = {usage.m_application_bytes, usage.m_fft_bytes,
                         usage.total_bytes()};
  size_t global_mem[3] = {0, 0, 0};

  MPI_Reduce(local_mem, global_mem, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);

  // Only rank 0 logs the report
  if (rank == 0) {
    log_info(logger, "");
    log_info(logger, "=== Memory Usage Report ===");

    // Per-rank memory
    std::ostringstream rank_msg;
    rank_msg << "  Rank 0 - Application: " << format_bytes(usage.m_application_bytes)
             << ", FFT: " << format_bytes(usage.m_fft_bytes)
             << ", Total: " << format_bytes(usage.total_bytes());
    log_info(logger, rank_msg.str());

    // Global total
    std::ostringstream global_msg;
    global_msg << "  Global Total: " << format_bytes(global_mem[2]) << " ("
               << num_ranks << " ranks)";
    log_info(logger, global_msg.str());

    // Per-voxel memory
    auto [Nx, Ny, Nz] = get_size(world);
    size_t total_voxels = static_cast<size_t>(Nx) * Ny * Nz;
    double mem_per_voxel = static_cast<double>(global_mem[2]) / total_voxels;

    std::ostringstream voxel_msg;
    voxel_msg << "  Per Voxel: " << format_bytes(static_cast<size_t>(mem_per_voxel))
              << "/voxel (" << total_voxels << " voxels)";
    log_info(logger, voxel_msg.str());

    // System memory (if available)
    size_t system_mem = get_system_memory_bytes();
    if (system_mem > 0) {
      double usage_pct = (static_cast<double>(global_mem[2]) / system_mem) * 100.0;
      std::ostringstream sys_msg;
      sys_msg << "  System Memory: " << format_bytes(system_mem)
              << ", Usage: " << std::fixed << std::setprecision(1) << usage_pct
              << "%";
      log_info(logger, sys_msg.str());
    }

    log_info(logger, "===========================");
    log_info(logger, "");
  }
}

} // namespace utils
} // namespace pfc
