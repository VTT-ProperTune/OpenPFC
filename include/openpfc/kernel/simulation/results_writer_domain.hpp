// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file results_writer_domain.hpp
 * @brief Map `Domain` + owned `Box3i` (or a `Field`) onto
 * `ResultsWriter::set_domain`
 *
 * @details
 * File writers need global size, local size, and local offset. Those come from
 * the field's geometry (`Domain` + owned `Box3i`), not from an FFT inbox;
 * prefer the `Field` overload so the writer and the data it receives share one
 * geometry source.
 *
 * The `Domain` also carries the physical spacing and origin, and those are
 * forwarded too. They used to be dropped, so every `.vti` written through a
 * JSON session claimed `Spacing="1 1 1"` no matter what `dx` the run used —
 * invisible for the `dx = 1` presets and wrong by 4x for a `dx = 0.25` one.
 * Geometry that the writer already knows how to record should not have to be
 * set a second time by hand at every call site.
 */

#include <array>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>

namespace pfc {

inline void apply_writer_domain(ResultsWriter &writer, const Domain &domain,
                                const Box3i &owned) {
  const auto gs = domain::get_size(domain);
  writer.set_domain({gs[0], gs[1], gs[2]},
                    {owned.size[0], owned.size[1], owned.size[2]},
                    {owned.low[0], owned.low[1], owned.low[2]});
  const auto origin = domain::get_origin(domain);
  const auto spacing = domain::get_spacing(domain);
  writer.set_geometry({origin[0], origin[1], origin[2]},
                      {spacing[0], spacing[1], spacing[2]});
}

template <typename T, typename MemorySpace = HostSpace>
inline void apply_writer_domain(ResultsWriter &writer,
                                const data::Field<T, MemorySpace> &field) {
  apply_writer_domain(writer, field.domain(), field.box());
}

} // namespace pfc
