// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_output.hpp
 * @brief Snapshots of the coupled state to raw MPI-IO bricks, with a manifest.
 *
 * @details
 * ## Why this exists at all
 *
 * Every other diagnostic in this application is a *number*: a tip velocity, a
 * selection parameter, a conservation residual. Numbers are what verification
 * needs, and for a long while they were the only output. They are not what a
 * reader needs to believe that the thing being measured is a dendrite, or to
 * see where in the morphology the elastic stress actually lives. A coupled
 * thermo-solutal-elastic run has four fields and a stress tensor, and the
 * interesting statement -- "the eigenstrain is carried by the solute-rich
 * solid, so the stress concentrates behind the tip and in the grooves between
 * arms, not at the tip itself" -- is a statement about a picture.
 *
 * ## Format, and why raw rather than VTI
 *
 * Each field goes to its own headerless Fortran-ordered `double` brick through
 * `pfc::BinaryWriter`, i.e. one collective `MPI_File_write_all`. The grid
 * shape lives out of band in a JSON manifest written once by rank 0. That is
 * the format `docs/report/figures/field_io.py` already reads (`read_raw`), and
 * -- unlike the single-piece `.vti` path the other report figures use -- it is
 * correct at any rank count, which matters because the 3-D cases here are not
 * single-rank runs. The manifest is what makes the brick self-describing; a
 * `.bin` without one is a file nobody can read six months later.
 *
 * ## What is written, and when
 *
 * `phi`, `U` and `theta` always. When the elastic coupling is on, also
 * `f_el` (the energy density of equation (6)), `dfel_dphi` (the driving-force
 * term of equation (7), i.e. exactly what feeds back into equation (2)), and
 * two stress invariants: the hydrostatic `p = tr(sigma)/3` and the von Mises
 * equivalent. The invariants rather than the six components, because a
 * dilatational eigenstrain makes the component figures six views of one
 * scalar, and because the hydrostatic part is the one that couples to
 * composition.
 *
 * Snapshots are taken on the *diagnostic sample* grid, thinned by
 * @ref FieldOutputConfig::every. Writing on a separate cadence would put the
 * snapshot at a time no CSV row records, and then the figure and the
 * time series could not be read against one another.
 *
 * ## The halo
 *
 * The phase-field fields are padded (`FDPaddedCPUStack`, halo `order/2`); the
 * elastic fields are flat. Both are packed into a contiguous owned-cell buffer
 * before the write, because `BinaryWriter` requires exactly the local brick
 * product and the padded field's `vec()` is longer than that. Writing the
 * padded vector directly would produce a file that is the right size only at
 * `order = 2` and is silently sheared everywhere else.
 *
 * @see docs/reference/binary_field_io_spec.md
 * @see docs/report/figures/field_io.py for the reader
 */

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/frontend/io/binary_writer.hpp>

namespace alloy_dendrite {

/// Where and how often to snapshot. Empty @ref dir disables everything.
struct FieldOutputConfig {
  std::string dir;
  /// Write every `every`-th diagnostic sample. 1 = every sample.
  int every{1};
};

/**
 * @brief Packs owned cells and writes them as raw bricks, plus a manifest.
 *
 * One instance per run, constructed only when output is wanted. The
 * constructor creates nothing on disk: `BinaryWriter` is collective and is
 * built per file, so a run that never reaches a snapshot leaves no artefacts.
 */
class FieldSnapshotWriter {
public:
  FieldSnapshotWriter(FieldOutputConfig cfg, std::string run_id,
                      const std::array<int, 3> &global,
                      const std::array<int, 3> &local,
                      const std::array<int, 3> &offset, double dx, int rank,
                      MPI_Comm comm)
      : m_cfg(std::move(cfg)), m_run(std::move(run_id)), m_global(global),
        m_local(local), m_offset(offset), m_dx(dx), m_rank(rank), m_comm(comm),
        m_count(static_cast<std::size_t>(local[0]) *
                static_cast<std::size_t>(local[1]) *
                static_cast<std::size_t>(local[2])) {
    m_buf.resize(m_count);
  }

  [[nodiscard]] bool active() const noexcept { return !m_cfg.dir.empty(); }

  /// True when diagnostic sample @p sample (0-based) is a snapshot sample.
  [[nodiscard]] bool due(int sample) const noexcept {
    return active() && (sample % std::max(1, m_cfg.every)) == 0;
  }

  /**
   * @brief Write one field.
   *
   * @param name  Field name; becomes part of the file name and the manifest.
   * @param idx   Snapshot index, zero-padded to four digits in the file name.
   * @param f     Any field with `for_each_owned` and `operator()(i,j,k)`.
   *
   * Collective over the writer's communicator: every rank must call it with
   * the same @p name and @p idx.
   */
  template <typename FieldT>
  void write(const std::string &name, int idx, const FieldT &f) {
    std::size_t n = 0;
    // for_each_owned visits owned cells in the field's own storage order,
    // which is i fastest. That is the Fortran order BinaryWriter's subarray
    // filetype expects, so the pack is a straight append rather than an
    // index computation -- and if that order ever changed, every brick would
    // be transposed rather than subtly wrong, which is the failure mode one
    // actually notices.
    const_cast<FieldT &>(f).for_each_owned(
        [&](int i, int j, int k) { m_buf[n++] = f(i, j, k); });
    write_buffer_(name, idx);
  }

  /// Write a field derived cell-by-cell from a symmetric tensor's components.
  template <typename SymFields, typename Fn>
  void write_from_sym(const std::string &name, int idx, const SymFields &s,
                      Fn &&reduce) {
    for (std::size_t q = 0; q < m_count; ++q) {
      double c[6];
      for (int a = 0; a < 6; ++a) {
        c[a] = s[static_cast<std::size_t>(a)].data()[q];
      }
      m_buf[q] = reduce(c);
    }
    write_buffer_(name, idx);
  }

  /// Record that snapshot @p idx was taken at time @p t.
  void note_time(double t) { m_times.push_back(t); }

  /**
   * @brief Write the manifest. Rank 0 only; call once, after the run.
   *
   * Without it the bricks are unreadable, so it is written even if the run
   * ended early -- the times vector says how many snapshots actually landed.
   */
  void write_manifest(const std::vector<std::string> &fields) const {
    if (!active() || m_rank != 0) return;
    const std::string path = m_cfg.dir + "/" + m_run + "_manifest.json";
    std::FILE *fp = std::fopen(path.c_str(), "w");
    if (fp == nullptr) return;
    std::fprintf(fp, "{\n  \"run_id\": \"%s\",\n", m_run.c_str());
    std::fprintf(fp, "  \"nx\": %d,\n  \"ny\": %d,\n  \"nz\": %d,\n", m_global[0],
                 m_global[1], m_global[2]);
    std::fprintf(fp, "  \"dx\": %.17g,\n", m_dx);
    std::fprintf(fp, "  \"order\": \"fortran\",\n  \"dtype\": \"float64\",\n");
    std::fprintf(fp, "  \"fields\": [");
    for (std::size_t i = 0; i < fields.size(); ++i) {
      std::fprintf(fp, "%s\"%s\"", i ? ", " : "", fields[i].c_str());
    }
    std::fprintf(fp, "],\n  \"times\": [");
    for (std::size_t i = 0; i < m_times.size(); ++i) {
      std::fprintf(fp, "%s%.10g", i ? ", " : "", m_times[i]);
    }
    std::fprintf(fp, "],\n  \"pattern\": \"%s_{field}_{index:04d}.bin\"\n}\n",
                 m_run.c_str());
    std::fclose(fp);
  }

private:
  void write_buffer_(const std::string &name, int idx) {
    char tail[64];
    std::snprintf(tail, sizeof(tail), "_%s_%04d.bin", name.c_str(), idx);
    pfc::BinaryWriter w(m_cfg.dir + "/" + m_run + tail, m_comm);
    w.set_domain(m_global, m_local, m_offset);
    w.write(0, pfc::field::FieldView<double>(m_buf));
  }

  FieldOutputConfig m_cfg;
  std::string m_run;
  std::array<int, 3> m_global{};
  std::array<int, 3> m_local{};
  std::array<int, 3> m_offset{};
  double m_dx{1.0};
  int m_rank{0};
  MPI_Comm m_comm{MPI_COMM_WORLD};
  std::size_t m_count{0};
  std::vector<double> m_buf;
  std::vector<double> m_times;
};

} // namespace alloy_dendrite
