// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file publish.hpp
 * @brief Atomic filesystem publication of accepted checkpoint bundles
 *
 * @details
 * `publish_checkpoint_directory` stages versioned `CheckpointMetadata` plus
 * injectable accepted field bricks under `<final_dir>.publishing/`, then
 * renames that staging directory to `final_dir` so incomplete writes are never
 * visible as a loadable checkpoint.
 *
 * Publish only accepted owned field cells. Do not place stage buffers, FFT
 * plans, operator caches, or stepper in-memory rollback scratch
 * (`EulerStepper` `m_u_checkpoint` and similar) into `PublishedFieldBrick`
 * spans. Time is caller-owned: fill metadata from `Time::get_current()` /
 * `get_increment()`.
 *
 * Field bricks are written with `std::ofstream` binary I/O (no frontend
 * `BinaryWriter`). A future adapter from #166 `FieldPayload` may live in a
 * separate header; this API accepts `PublishedFieldBrick` spans so Catch2
 * doubles work without that sibling.
 *
 * Staging and final paths must share a parent on the same filesystem so
 * `std::filesystem::rename` of the directory is atomic.
 *
 * @see checkpoint_metadata.hpp
 * @see docs/development/checkpoint_publish.md
 */

#ifndef OPENPFC_KERNEL_CHECKPOINT_PUBLISH_HPP
#define OPENPFC_KERNEL_CHECKPOINT_PUBLISH_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>

#include "openpfc/kernel/checkpoint/checkpoint_metadata.hpp"

namespace pfc::checkpoint {

/**
 * @brief Injectable accepted-field brick for checkpoint publication.
 *
 * Tests and drivers build these from owned buffers (e.g. `std::vector<double>`
 * via `std::as_bytes`) without depending on #166 payload carriers.
 */
struct PublishedFieldBrick {
  std::string id;
  std::string dtype{"float64"};
  std::array<int, 3> extents{};
  std::span<const std::byte> bytes;
  std::string coordinate_order{"fortran"};
};

/**
 * @brief Result of a publish attempt.
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
 * @brief Optional hook invoked immediately before each field file write.
 *
 * Tests may throw to force a mid-publish failure after metadata (and earlier
 * fields) have been staged.
 */
using PublishWriteHook =
    std::function<void(std::size_t field_index, const PublishedFieldBrick &)>;

namespace detail {

[[nodiscard]] inline std::size_t
expected_float64_bytes(const std::array<int, 3> &extents) {
  if (extents[0] <= 0 || extents[1] <= 0 || extents[2] <= 0) {
    return 0;
  }
  const auto nx = static_cast<std::size_t>(extents[0]);
  const auto ny = static_cast<std::size_t>(extents[1]);
  const auto nz = static_cast<std::size_t>(extents[2]);
  return nx * ny * nz * sizeof(double);
}

inline void best_effort_remove_all(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

} // namespace detail

/**
 * @brief Atomically publish a checkpoint directory bundle.
 *
 * Layout after success:
 * @code
 * <final_dir>/
 *   metadata.json
 *   fields/<field_id>.bin
 * @endcode
 *
 * @param final_dir Destination directory (must not already exist).
 * @param meta Versioned metadata (time/increment filled by caller).
 * @param fields Accepted field bricks only (injectable / test doubles OK).
 * @param before_field_write Optional pre-write hook (tests use to force fail).
 * @return `make_publish_ok(final_dir)` on success; failed outcome otherwise.
 *         On failure, staging is removed and `final_dir` is not left present.
 */
[[nodiscard]] inline PublishOutcome publish_checkpoint_directory(
    const std::filesystem::path &final_dir, const CheckpointMetadata &meta,
    std::span<const PublishedFieldBrick> fields,
    PublishWriteHook before_field_write = {}) {
  namespace fs = std::filesystem;

  if (fs::exists(final_dir)) {
    return make_publish_failed("checkpoint final path already exists: " +
                               final_dir.string());
  }

  const fs::path staging = fs::path(final_dir.string() + ".publishing");

  try {
    if (fs::exists(staging)) {
      detail::best_effort_remove_all(staging);
      if (fs::exists(staging)) {
        return make_publish_failed(
            "could not clear leftover staging directory: " + staging.string());
      }
    }

    fs::create_directories(staging / "fields");

    {
      const fs::path meta_path = staging / "metadata.json";
      std::ofstream out(meta_path, std::ios::out | std::ios::trunc);
      if (!out) {
        detail::best_effort_remove_all(staging);
        return make_publish_failed("failed to open metadata.json for write");
      }
      out << to_json(meta).dump(2) << '\n';
      out.flush();
      if (!out) {
        detail::best_effort_remove_all(staging);
        return make_publish_failed("failed while writing metadata.json");
      }
    }

    for (std::size_t i = 0; i < fields.size(); ++i) {
      const PublishedFieldBrick &brick = fields[i];
      if (before_field_write) {
        before_field_write(i, brick);
      }
      if (brick.id.empty()) {
        detail::best_effort_remove_all(staging);
        return make_publish_failed("field brick id must be non-empty");
      }
      if (brick.dtype == "float64") {
        const std::size_t expected =
            detail::expected_float64_bytes(brick.extents);
        if (expected == 0 || brick.bytes.size() != expected) {
          detail::best_effort_remove_all(staging);
          return make_publish_failed(
              "field brick byte length does not match float64 extents for id=" +
              brick.id);
        }
      } else {
        detail::best_effort_remove_all(staging);
        return make_publish_failed("unsupported field dtype: " + brick.dtype);
      }

      const fs::path field_path = staging / "fields" / (brick.id + ".bin");
      std::ofstream out(field_path, std::ios::binary | std::ios::trunc);
      if (!out) {
        detail::best_effort_remove_all(staging);
        return make_publish_failed("failed to open field file for write: " +
                                   field_path.string());
      }
      out.write(reinterpret_cast<const char *>(brick.bytes.data()),
                static_cast<std::streamsize>(brick.bytes.size()));
      out.flush();
      if (!out) {
        detail::best_effort_remove_all(staging);
        return make_publish_failed("failed while writing field file: " +
                                   field_path.string());
      }
    }

    std::error_code rename_ec;
    fs::rename(staging, final_dir, rename_ec);
    if (rename_ec) {
      detail::best_effort_remove_all(staging);
      return make_publish_failed("rename staging to final failed: " +
                                 rename_ec.message());
    }
    if (!fs::exists(final_dir) || !fs::is_directory(final_dir)) {
      detail::best_effort_remove_all(staging);
      detail::best_effort_remove_all(final_dir);
      return make_publish_failed(
          "final checkpoint directory missing after rename");
    }

    return make_publish_ok(final_dir);
  } catch (const std::exception &ex) {
    detail::best_effort_remove_all(staging);
    detail::best_effort_remove_all(final_dir);
    return make_publish_failed(std::string("publish failed: ") + ex.what());
  } catch (...) {
    detail::best_effort_remove_all(staging);
    detail::best_effort_remove_all(final_dir);
    return make_publish_failed("publish failed: unknown exception");
  }
}

} // namespace pfc::checkpoint

#endif
