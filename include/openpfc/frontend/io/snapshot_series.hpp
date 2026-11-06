// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file snapshot_series.hpp
 * @brief Solver-independent scalar-field snapshots.
 *
 * A driver registers fields and calls `write(step, time)`. The series owns
 * the output directory, the owned-cell pack, the writer, and the time-series
 * manifest. It does not know which stepper produced the field, and it does
 * not parse application configuration.
 *
 * The registered field is not owned. It must outlive every `write`. Device
 * fields are read through `with_host_read`, so a device backend is not
 * special-cased here. Call `close()` on the successful path so a writer,
 * manifest, or XDMF failure is reported. The destructor also calls
 * `close()` and swallows that error; it does not throw. `BinaryWriter`
 * frees its MPI filetype in `close()` and aborts from its destructor if
 * that free fails.
 */

#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/io/xdmf_binary_series.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>

namespace pfc::io {

enum class SnapshotFormat { Binary, Vtk };

/// How many output opportunities elapse between saves. `every` counts calls
/// the driver chooses, not a stepper's internal substeps. A non-positive
/// count is rejected.
struct SnapshotCadence {
  int every{1};

  explicit SnapshotCadence(int every_sample) : every(every_sample) {
    if (every < 1) {
      throw std::invalid_argument(
          "snapshot cadence: every sample count must be positive");
    }
  }

  [[nodiscard]] bool due(int sample_ordinal) const {
    if (sample_ordinal < 0) {
      throw std::invalid_argument("snapshot cadence: sample ordinal must be >= 0");
    }
    return sample_ordinal % every == 0;
  }
};

/// File ordinal and the step/time rows already written. The caller stores
/// this and passes it back to `restore`. The series does not search the
/// output directory.
struct SnapshotProgress {
  int next_frame{0};
  std::vector<int> steps;
  std::vector<double> times;
};

struct SnapshotSeriesOptions {
  std::filesystem::path directory;
  std::string prefix;
  MPI_Comm comm{MPI_COMM_NULL};
};

class SnapshotSeries {
public:
  SnapshotSeries(const Domain &domain, const Box3i &owned,
                 SnapshotSeriesOptions options)
      : m_domain(domain), m_owned(owned), m_options(std::move(options)) {
    if (m_options.comm == MPI_COMM_NULL) {
      throw std::invalid_argument("snapshot series: pass an explicit communicator");
    }
    MPI_Comm_rank(m_options.comm, &m_rank);
  }

  SnapshotSeries(const SnapshotSeries &) = delete;
  SnapshotSeries &operator=(const SnapshotSeries &) = delete;

  ~SnapshotSeries() {
    try {
      close();
    } catch (...) {
    }
  }

  /// Pack @p field's owned cells. Halos are not written.
  template <typename Space>
  void add_field(std::string name, data::Field<double, Space> &field,
                 std::string pattern = {},
                 SnapshotFormat format = SnapshotFormat::Binary) {
    if (field.box() != m_owned) {
      throw std::invalid_argument("snapshot series: '" + name +
                                  "' is not on the series owned box");
    }
    if (pattern.empty()) {
      if (m_options.directory.empty() || m_options.prefix.empty()) {
        throw std::invalid_argument(
            "snapshot series: a field needs a path or a directory and prefix");
      }
      pattern = (m_options.directory / (m_options.prefix + "_" + name + "_%04d.bin"))
                    .string();
      format = SnapshotFormat::Binary;
    }
    add_packed(std::move(name), std::move(pattern), format,
               [&field](std::vector<double> &out) { pack_owned(field, out); });
  }

  /// @p pack writes the owned brick, x-fastest, with no halo cells.
  void add_field(std::string name, std::string pattern, SnapshotFormat format,
                 std::function<void(std::vector<double> &)> pack) {
    if (!pack) {
      throw std::invalid_argument("snapshot series: '" + name + "' has no source");
    }
    add_packed(std::move(name), std::move(pattern), format, std::move(pack));
  }

  [[nodiscard]] bool has_field(std::string_view name) const {
    for (const Slot &slot : m_slots) {
      if (slot.name == name) return true;
    }
    return false;
  }

  [[nodiscard]] bool empty() const noexcept { return m_slots.empty(); }
  [[nodiscard]] int frames() const noexcept { return m_index; }

  void set_cadence(SnapshotCadence cadence) { m_cadence = cadence; }

  /// True when @p sample_ordinal is a save. Requires `set_cadence`.
  [[nodiscard]] bool due(int sample_ordinal) const {
    if (!m_cadence) {
      throw std::invalid_argument("snapshot series: set a cadence before due()");
    }
    return m_cadence->due(sample_ordinal);
  }

  /// `write` when `due(sample_ordinal)` and at least one field is registered.
  bool write_if_due(int sample_ordinal, int step, double time) {
    if (!due(sample_ordinal) || empty()) return false;
    write(step, time);
    return true;
  }

  [[nodiscard]] SnapshotProgress progress() const {
    return SnapshotProgress{m_index, m_steps, m_times};
  }

  /// Continue numbering at @p saved.next_frame and keep the earlier rows.
  /// Call it before the first `write` on this series.
  void restore(SnapshotProgress saved) {
    if (m_closed) {
      throw std::invalid_argument("snapshot series: restore after close");
    }
    if (m_index != 0 || !m_steps.empty() || !m_times.empty()) {
      throw std::invalid_argument("snapshot series: restore before the first write");
    }
    if (saved.next_frame < 0 ||
        static_cast<int>(saved.steps.size()) != saved.next_frame ||
        saved.times.size() != saved.steps.size()) {
      throw std::invalid_argument("snapshot series: progress next_frame must match "
                                  "the saved steps and times");
    }
    m_index = saved.next_frame;
    m_steps = std::move(saved.steps);
    m_times = std::move(saved.times);
  }
  [[nodiscard]] const std::vector<int> &steps() const noexcept { return m_steps; }
  [[nodiscard]] const std::vector<double> &times() const noexcept { return m_times; }

  /// File ordinal is the number of `write` calls. @p step and @p time go in the
  /// manifest, not into the filename.
  void write(int step, double time) {
    if (m_closed) {
      throw std::invalid_argument("snapshot series: write after close");
    }
    if (m_slots.empty()) return;
    for (Slot &slot : m_slots) {
      int local_ok = 1;
      std::string error;
      try {
        if (!slot.writer->writes_real()) {
          throw std::invalid_argument("snapshot series: '" + slot.name +
                                      "' writer does not support real fields");
        }
        slot.pack(m_packed);
      } catch (const std::exception &ex) {
        local_ok = 0;
        error = ex.what();
      } catch (...) {
        local_ok = 0;
        error = "snapshot series: pack failed for '" + slot.name + "'";
      }
      agree_(local_ok, error,
             "snapshot series: a peer failed while packing '" + slot.name + "'");
      slot.writer->write(m_index, pfc::field::FieldView<double>(m_packed));
    }
    m_steps.push_back(step);
    m_times.push_back(time);
    ++m_index;
  }

  /// Rank 0 writes the manifest and, for binary frames, one XDMF series.
  /// Every rank throws if that write fails. The failing rank keeps its
  /// error text; the others report a collective failure.
  void close() {
    if (m_closed) return;
    for (auto &slot : m_slots) {
      if (!slot.writer) continue;
      std::string error;
      int local_ok = 1;
      try {
        slot.writer->close();
      } catch (const std::exception &ex) {
        local_ok = 0;
        error = ex.what();
      } catch (...) {
        local_ok = 0;
        error = "snapshot series: writer close failed for '" + slot.name + "'";
      }
      agree_(local_ok, error,
             "snapshot series: a peer failed while closing '" + slot.name + "'");
    }
    m_closed = true;
    if (m_slots.empty()) return;
    std::string error;
    if (m_rank == 0) {
      try {
        write_manifest_();
      } catch (const std::exception &ex) {
        error = ex.what();
      } catch (...) {
        error = "snapshot series: manifest write failed";
      }
    }
    int failed = error.empty() ? 0 : 1;
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&failed, 1, MPI_INT, 0, m_options.comm),
                                 "snapshot series: MPI_Bcast");
    if (failed != 0) {
      if (!error.empty()) throw std::runtime_error(error);
      throw std::runtime_error(
          "snapshot series: manifest or XDMF write failed on another rank");
    }
  }

private:
  struct Slot {
    std::string name;
    SnapshotFormat format{SnapshotFormat::Binary};
    std::string pattern;
    std::unique_ptr<ResultsWriter> writer;
    std::function<void(std::vector<double> &)> pack;
  };

  void agree_(int local_ok, const std::string &error, const std::string &peer) {
    int global_ok = 0;
    pfc::mpi::throw_on_mpi_error(
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_options.comm),
        "snapshot series: MPI_Allreduce");
    if (global_ok != 0) return;
    if (local_ok == 0 && !error.empty()) throw std::runtime_error(error);
    throw std::runtime_error(peer);
  }

  template <typename Space>
  static void pack_owned(data::Field<double, Space> &field,
                         std::vector<double> &out) {
    const auto n = field.box().size;
    const int nx = n[0];
    const int ny = n[1];
    const int nz = n[2];
    const int halo = field.storage_halo();
    const auto npx = static_cast<std::size_t>(nx + 2 * halo);
    const auto npy = static_cast<std::size_t>(ny + 2 * halo);
    const auto hw = static_cast<std::size_t>(halo);
    out.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
               static_cast<std::size_t>(nz));
    field.with_host_read([&](const double *data, std::size_t) {
      std::size_t q = 0;
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            const std::size_t src = (static_cast<std::size_t>(i) + hw) +
                                    (static_cast<std::size_t>(j) + hw) * npx +
                                    (static_cast<std::size_t>(k) + hw) * npx * npy;
            out[q++] = data[src];
          }
        }
      }
    });
  }

  void add_packed(std::string name, std::string pattern, SnapshotFormat format,
                  std::function<void(std::vector<double> &)> pack) {
    if (pattern.empty()) {
      throw std::invalid_argument("snapshot series: '" + name + "' has no path");
    }
    ensure_parent_(std::filesystem::path(pattern));
    std::unique_ptr<ResultsWriter> writer;
    if (format == SnapshotFormat::Vtk) {
      auto vtk = std::make_unique<VTKWriter>(pattern, m_options.comm);
      vtk->set_field_name(name);
      writer = std::move(vtk);
    } else {
      writer = std::make_unique<BinaryWriter>(pattern, m_options.comm);
    }
    if (!writer->writes_real()) {
      throw std::invalid_argument("snapshot series: writer for '" + name +
                                  "' does not support real fields");
    }
    apply_writer_domain(*writer, m_domain, m_owned);
    m_slots.push_back(Slot{std::move(name), format, std::move(pattern),
                           std::move(writer), std::move(pack)});
  }

  void ensure_parent_(const std::filesystem::path &pattern) {
    const std::filesystem::path parent = pattern.parent_path();
    if (parent.empty()) return;
    int failed = 0;
    if (m_rank == 0) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec) failed = 1;
    }
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&failed, 1, MPI_INT, 0, m_options.comm),
                                 "snapshot series: MPI_Bcast");
    pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_options.comm),
                                 "snapshot series: MPI_Barrier");
    if (failed != 0) {
      throw std::runtime_error("snapshot series: cannot create output directory '" +
                               parent.string() + "'");
    }
  }

  void write_manifest_() const {
    const std::filesystem::path path = manifest_path_();
    const auto n = domain::get_size(m_domain);
    const auto spacing = domain::get_spacing(m_domain);
    const auto origin = domain::get_origin(m_domain);
    nlohmann::json fields = nlohmann::json::array();
    nlohmann::json patterns = nlohmann::json::array();
    for (const Slot &slot : m_slots) {
      fields.push_back(slot.name);
      patterns.push_back(slot.pattern);
    }
    nlohmann::json doc = {{"prefix", m_options.prefix},
                          {"nx", n[0]},
                          {"ny", n[1]},
                          {"nz", n[2]},
                          {"dx", spacing[0]},
                          {"dy", spacing[1]},
                          {"dz", spacing[2]},
                          {"origin", {origin[0], origin[1], origin[2]}},
                          {"order", "fortran"},
                          {"dtype", "float64"},
                          {"fields", std::move(fields)},
                          {"steps", m_steps},
                          {"times", m_times},
                          {"patterns", std::move(patterns)}};
    write_text_file(path, doc.dump(2) + '\n');
    write_xdmf_();
  }

  void write_xdmf_() const {
    if (m_times.empty()) return;
    const std::filesystem::path xdmf = xdmf_path_();
    const auto base =
        xdmf.parent_path().empty() ? std::filesystem::path(".") : xdmf.parent_path();
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> rels;
    for (const Slot &slot : m_slots) {
      if (slot.format != SnapshotFormat::Binary) continue;
      names.push_back(slot.name);
      std::vector<std::string> frames;
      frames.reserve(m_times.size());
      for (int frame = 0; frame < m_index; ++frame) {
        const auto file =
            std::filesystem::path(utils::format_with_number(slot.pattern, frame));
        const auto rel = file.lexically_relative(base);
        if (rel.empty()) {
          throw std::runtime_error("snapshot series: cannot express '" +
                                   file.string() + "' relative to '" +
                                   xdmf.string() + "'");
        }
        frames.push_back(rel.generic_string());
      }
      rels.push_back(std::move(frames));
    }
    if (names.empty()) return;
    const auto n = domain::get_size(m_domain);
    const auto spacing = domain::get_spacing(m_domain);
    const auto origin = domain::get_origin(m_domain);
    const double dz = n[2] > 1 ? spacing[2] : 1.0;
    write_xdmf_binary_series(xdmf.string(),
                             BinarySeriesGeometry{.nx = n[0],
                                                  .ny = n[1],
                                                  .nz = n[2],
                                                  .x0 = origin[0],
                                                  .y0 = origin[1],
                                                  .z0 = origin[2],
                                                  .dx = spacing[0],
                                                  .dy = spacing[1],
                                                  .dz = dz},
                             names, rels, m_times);
  }

  [[nodiscard]] std::filesystem::path manifest_path_() const {
    if (!m_options.directory.empty() && !m_options.prefix.empty()) {
      return m_options.directory / (m_options.prefix + "_manifest.json");
    }
    const auto parent = std::filesystem::path(m_slots.front().pattern).parent_path();
    return (parent.empty() ? std::filesystem::path(".") : parent) /
           "snapshots_manifest.json";
  }

  [[nodiscard]] std::filesystem::path xdmf_path_() const {
    if (!m_options.directory.empty() && !m_options.prefix.empty()) {
      return m_options.directory / (m_options.prefix + ".xdmf");
    }
    const auto parent = std::filesystem::path(m_slots.front().pattern).parent_path();
    return (parent.empty() ? std::filesystem::path(".") : parent) / "snapshots.xdmf";
  }

  Domain m_domain;
  Box3i m_owned;
  SnapshotSeriesOptions m_options;
  int m_rank{0};
  int m_index{0};
  bool m_closed{false};
  std::optional<SnapshotCadence> m_cadence;
  std::vector<Slot> m_slots;
  std::vector<int> m_steps;
  std::vector<double> m_times;
  std::vector<double> m_packed;
};

} // namespace pfc::io
