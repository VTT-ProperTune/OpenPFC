// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file publish.hpp
 * @brief Atomic, MPI-collective publication of accepted checkpoint bundles
 *
 * @details
 * `publish_checkpoint_directory` is the **one** checkpoint publisher. It stages
 * `<final_dir>.publishing/` (rank 0 creates it), lets the caller write the
 * accepted field bricks into `<staging>/fields/` through a collective callback
 * (normally `brick_io.hpp::write_real_brick_mpi`), writes versioned
 * `CheckpointMetadata` on rank 0, agrees success with `MPI_Allreduce`, and only
 * then renames staging to `final_dir`. Incomplete writes are never visible as
 * a loadable checkpoint; on failure staging is removed on rank 0.
 *
 * Publish only accepted owned field cells. Do not place stage buffers, FFT
 * plans, operator caches, or stepper rollback scratch into a bundle. Time is
 * caller-owned: fill metadata from `Time::get_current()` / `get_increment()`.
 *
 * Staging and final paths must share a parent on the same filesystem so
 * `std::filesystem::rename` of the directory is atomic. Every rank of @p comm
 * must call this function (it is collective).
 *
 * @see checkpoint_metadata.hpp
 * @see brick_io.hpp
 * @see docs/development/checkpoint_publish.md
 */

#ifndef OPENPFC_KERNEL_CHECKPOINT_PUBLISH_HPP
#define OPENPFC_KERNEL_CHECKPOINT_PUBLISH_HPP

#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>

#include <mpi.h>

#include "openpfc/kernel/checkpoint/checkpoint_metadata.hpp"
#include "openpfc/kernel/mpi/mpi_io_helpers.hpp"

namespace pfc::checkpoint {

/**
 * @brief Result of a publish attempt (identical on every rank).
 */
struct PublishOutcome {
  bool ok{false};
  std::string message;
  std::filesystem::path final_path;
};

[[nodiscard]] inline PublishOutcome
make_publish_ok(std::filesystem::path final_path) {
  return PublishOutcome{.ok = true,
                        .message = {},
                        .final_path = std::move(final_path)};
}

[[nodiscard]] inline PublishOutcome
make_publish_failed(std::string message) {
  return PublishOutcome{.ok = false,
                        .message = std::move(message),
                        .final_path = {}};
}

/**
 * @brief Collective callback that writes every field brick into @p fields_dir.
 *
 * Called on every rank with the same staging path. Throw to abort the publish;
 * the exception message becomes `PublishOutcome::message` on that rank.
 */
using FieldsWriter = std::function<void(const std::filesystem::path &fields_dir)>;

namespace detail {

inline void best_effort_remove_all(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

} // namespace detail

/**
 * @brief Atomically publish a checkpoint directory bundle (MPI-collective).
 *
 * Layout after success:
 * @code
 * <final_dir>/
 *   metadata.json
 *   fields/<field_id>.bin
 * @endcode
 *
 * @param final_dir    Destination directory (must not already exist).
 * @param meta         Versioned metadata (time/increment filled by caller).
 * @param comm         Communicator whose ranks all call this function.
 * @param write_fields Collective brick writer (see @ref FieldsWriter).
 * @return `make_publish_ok(final_dir)` on success on every rank; otherwise a
 *         failed outcome. On failure staging is removed and `final_dir` is not
 *         left present.
 */
[[nodiscard]] inline PublishOutcome
publish_checkpoint_directory(const std::filesystem::path &final_dir,
                             const CheckpointMetadata &meta, MPI_Comm comm,
                             const FieldsWriter &write_fields) {
  namespace fs = std::filesystem;
  int rank = 0;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank publish");
  const fs::path staging = fs::path(final_dir.string() + ".publishing");

  auto agree_min = [&](int local) {
    int global = 0;
    pfc::mpi::throw_on_mpi_error(
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm),
        "MPI_Allreduce publish");
    return global;
  };
  auto barrier = [&] {
    pfc::mpi::throw_on_mpi_error(MPI_Barrier(comm), "MPI_Barrier publish");
  };

  // 1. Refuse to overwrite an existing bundle (rank 0 decides, all agree).
  int exists_flag = 0;
  if (rank == 0) {
    exists_flag = fs::exists(final_dir) ? 1 : 0;
  }
  pfc::mpi::throw_on_mpi_error(MPI_Bcast(&exists_flag, 1, MPI_INT, 0, comm),
                               "MPI_Bcast publish exists");
  if (exists_flag != 0) {
    return make_publish_failed("checkpoint final path already exists: " +
                               final_dir.string());
  }

  // 2. Rank 0 prepares a clean staging directory.
  std::string message;
  int local_ok = 1;
  if (rank == 0) {
    try {
      detail::best_effort_remove_all(staging);
      if (fs::exists(staging)) {
        throw std::runtime_error("could not clear leftover staging directory: " +
                                 staging.string());
      }
      fs::create_directories(staging / "fields");
    } catch (const std::exception &ex) {
      local_ok = 0;
      message = std::string("publish staging failed: ") + ex.what();
    }
  }
  if (agree_min(local_ok) == 0) {
    if (rank == 0) {
      detail::best_effort_remove_all(staging);
    }
    barrier();
    return make_publish_failed(message.empty() ? "publish staging failed on rank 0"
                                               : message);
  }
  barrier();

  // 3. Collective field bricks + rank-0 metadata.
  try {
    write_fields(staging / "fields");
    if (rank == 0) {
      std::ofstream out(staging / "metadata.json", std::ios::out | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("failed to open metadata.json for write");
      }
      out << to_json(meta).dump(2) << '\n';
      out.flush();
      if (!out) {
        throw std::runtime_error("failed while writing metadata.json");
      }
    }
  } catch (const std::exception &ex) {
    local_ok = 0;
    message = std::string("publish failed: ") + ex.what();
  } catch (...) {
    local_ok = 0;
    message = "publish failed: unknown exception";
  }
  if (agree_min(local_ok) == 0) {
    if (rank == 0) {
      detail::best_effort_remove_all(staging);
      detail::best_effort_remove_all(final_dir);
    }
    barrier();
    return make_publish_failed(
        message.empty() ? "publish failed on another rank" : message);
  }

  // 4. Atomic rename on rank 0, then everyone sees the final bundle.
  if (rank == 0) {
    std::error_code rename_ec;
    fs::rename(staging, final_dir, rename_ec);
    if (rename_ec) {
      local_ok = 0;
      message = "rename staging to final failed: " + rename_ec.message();
    } else if (!fs::exists(final_dir) || !fs::is_directory(final_dir)) {
      local_ok = 0;
      message = "final checkpoint directory missing after rename";
    }
    if (local_ok == 0) {
      detail::best_effort_remove_all(staging);
      detail::best_effort_remove_all(final_dir);
    }
  }
  if (agree_min(local_ok) == 0) {
    barrier();
    return make_publish_failed(message.empty() ? "publish rename failed on rank 0"
                                               : message);
  }
  barrier();
  return make_publish_ok(final_dir);
}

} // namespace pfc::checkpoint

#endif
