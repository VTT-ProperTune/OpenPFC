// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_snapshots.hpp
 * @brief Optional `fields[]` snapshot output for the standalone science drivers.
 *
 * Shared by the science drivers that do *not* run on
 * `SpectralETDSession`: `surface_diffusion_anisotropic` and
 * `ehd_film_nonlinear`. Both own their own `main()` because their physics is
 * not expressible as a reciprocal-space symbol, and both consequently missed
 * out on the one thing the session gives every other application for free —
 * a field writer.
 *
 * @details
 * That gap had a concrete cost: the science presets of those two apps could
 * only ever be read through their diagnostics CSV. A CSV can say that the
 * anisotropic anneal ends up with three quarters of its spectral energy in
 * \f$k_y\f$, or that a compliant plate dents four times deeper than a stiff
 * one, but it cannot show *what the surface looks like* while doing so. The
 * report's field figures need the field.
 *
 * The JSON spelling is deliberately the same one the session-based presets of
 * the same two applications already use, so a reader moving between
 * `smoothing.json` and `nanosurface_isotropic.json` does not meet a second
 * convention:
 *
 * ```json
 * "fields": [ { "name": "h", "data": "results/<app>/<case>_%04d.vti" } ]
 * ```
 *
 * Only the subset these drivers can honour is supported: one entry, naming
 * the driver's single evolving field, written to a `.vti` path. Anything else
 * (a second entry, an unknown field name) is rejected loudly rather than
 * silently ignored — a preset that believes it is writing output and is not
 * is worse than one that fails.
 *
 * Omitting `fields[]` entirely keeps the previous behaviour: diagnostics
 * only, no file-system traffic. That matters because these presets are also
 * run from the test suite, where the CSV is the assertion and the snapshots
 * would be dead weight.
 *
 * Snapshot indices count saves, not steps, exactly as
 * `SpectralETDSession::write_results` does: the file written at \f$t = 0\f$ is
 * `_0000`, the one written after the first `saveat` interval is `_0001`, and
 * so on. A figure script can therefore recover the time of frame \f$i\f$ as
 * \f$i \times \mathtt{saveat}\f$ without parsing anything.
 */

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>

namespace pfc::apps {

/**
 * @brief Build a `.vti` snapshot writer from an optional `fields[]` entry.
 *
 * Collective on @p comm: rank 0 creates the output directory and every rank
 * waits for it, so a preset may name a `results/...` path that does not exist
 * yet (the diagnostics CSV in the same drivers behaves the same way).
 *
 * @param cfg         the parsed case JSON; `fields` is optional
 * @param field_name  the name of the driver's evolving field, e.g. `"h"`
 * @param field       that field, used only for its geometry
 * @param comm        communicator the writer is collective over
 * @return the writer, or `nullptr` when the case asks for no field output
 *
 * @throws std::invalid_argument if `fields[]` is malformed, holds more than
 *         one entry, or names a field this driver does not own
 */
inline std::unique_ptr<pfc::VTKWriter>
make_field_snapshot_writer(const nlohmann::json &cfg, const std::string &field_name,
                           const pfc::data::Field<double> &field,
                           MPI_Comm comm = MPI_COMM_WORLD) {
  if (!cfg.contains("fields")) return nullptr;
  const auto &fields = cfg.at("fields");
  if (!fields.is_array())
    throw std::invalid_argument("fields: must be an array of {name, data} objects");
  if (fields.empty()) return nullptr;
  if (fields.size() > 1)
    throw std::invalid_argument(
        "fields: this driver evolves a single field, so at most one entry is "
        "meaningful");

  const auto &entry = fields.front();
  if (!entry.is_object() || !entry.contains("data"))
    throw std::invalid_argument("fields[0]: expected an object with a `data` path");
  const std::string name = entry.value("name", field_name);
  if (name != field_name)
    throw std::invalid_argument("fields[0].name: this driver owns only '" +
                                field_name + "', got '" + name + "'");

  const std::filesystem::path path = entry.at("data").get<std::string>();
  if (path.extension() != ".vti")
    throw std::invalid_argument(
        "fields[0].data: only `.vti` output is implemented here, got '" +
        path.string() + "'");

  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0 && path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
  MPI_Barrier(comm);

  auto writer = std::make_unique<pfc::VTKWriter>(path.string(), comm);
  writer->set_field_name(name);
  // Origin and spacing must be set before `set_domain`, which validates the
  // three against each other.
  const auto &origin = field.origin();
  const auto &spacing = field.spacing();
  writer->set_origin({origin[0], origin[1], origin[2]});
  writer->set_spacing({spacing[0], spacing[1], spacing[2]});
  pfc::apply_writer_domain(*writer, field);
  return writer;
}

/**
 * @brief Write one snapshot of @p field, or do nothing if @p writer is null.
 *
 * The null case is the point: a driver can call this unconditionally at every
 * save without wrapping it in a test for whether the case requested output.
 *
 * @param writer     may be `nullptr`
 * @param increment  save index, not step index (see the file-level notes)
 */
inline void write_field_snapshot(pfc::VTKWriter *writer, int increment,
                                 const pfc::data::Field<double> &field) {
  if (writer == nullptr) return;
  const pfc::field::FieldView<double> view(field.data(), field.size(),
                                           field.box().size, field.spacing(),
                                           field.origin());
  writer->write(increment, view);
}

} // namespace pfc::apps
