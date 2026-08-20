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
 * the field's geometry, not from an FFT inbox. Gen-1 `Simulator` still passes
 * the FFT real-space box as the owned box until Model fields are
 * `pfc::data::Field`.
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
}

template <typename T, typename MemorySpace = HostSpace>
inline void apply_writer_domain(ResultsWriter &writer,
                                const data::Field<T, MemorySpace> &field) {
  apply_writer_domain(writer, field.domain(), field.box());
}

} // namespace pfc
