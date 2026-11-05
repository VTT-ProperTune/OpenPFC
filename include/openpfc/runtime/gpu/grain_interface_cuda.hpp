// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file grain_interface_cuda.hpp
 * @brief One-rank CUDA copy of a compact grain interface.
 *
 * The device holds the grain-id field and the narrow band, not one
 * signed-distance field per grain. HIP has the same kernel entry and is
 * not executed here.
 */

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/field/grain_interface.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>

namespace pfc::field::grain_interface {

struct GrainDeviceProfile {
  std::size_t device_bytes{0};
  std::size_t host_to_device_bytes{0};
  std::size_t device_to_host_bytes{0};
  int kernel_launches{0};
};

struct GrainDeviceCheck {
  GrainDeviceProfile profile{};
  std::vector<int> cell_count;
  int mismatches{0};
};

/// Resident grain ids and band. Buffers are allocated once.
class GrainInterfaceCuda {
public:
  explicit GrainInterfaceCuda(const GrainInterface &proto)
      : m_n(proto.grain_id.size()), m_band(proto.band.size()),
        m_grains(static_cast<int>(proto.cell_count.size())), m_nx(proto.grid.nx),
        m_ny(proto.grid.ny), m_nz(proto.grid.nz), m_ids(m_n), m_band_buf(m_band),
        m_counts(static_cast<std::size_t>(std::max(m_grains, 0))), m_mismatch(1) {
    if (m_grains < 1 || m_n == 0)
      throw std::invalid_argument("grain device buffer needs grains and cells");
  }

  GrainInterfaceCuda(const GrainInterfaceCuda &) = delete;
  GrainInterfaceCuda &operator=(const GrainInterfaceCuda &) = delete;

  [[nodiscard]] const GrainDeviceProfile &profile() const noexcept {
    return m_profile;
  }

  /// Upload, recount, and check the band. A throw leaves the profile at zero
  /// transfers.
  [[nodiscard]] GrainDeviceCheck measure(const GrainInterface &in) {
    m_profile = {};
    if (in.grain_id.size() != m_n || in.band.size() != m_band ||
        static_cast<int>(in.cell_count.size()) != m_grains || in.grid.nx != m_nx ||
        in.grid.ny != m_ny || in.grid.nz != m_nz)
      throw std::invalid_argument(
          "grain interface does not match the device buffer");
    for (int id : in.grain_id) {
      if (id < 0 || id >= m_grains)
        throw std::invalid_argument("grain id is outside the seed set");
    }
    for (const BandSite &site : in.band) {
      if (site.index < 0 || static_cast<std::size_t>(site.index) >= m_n)
        throw std::invalid_argument("grain band index is outside the grid");
    }
    m_profile.device_bytes = resident_device_bytes(m_n, m_band, m_grains);
    m_ids.copy_from_host(in.grain_id);
    m_profile.host_to_device_bytes += m_n * sizeof(int);
    if (m_band > 0) {
      m_band_buf.copy_from_host(in.band);
      m_profile.host_to_device_bytes += m_band * sizeof(BandSite);
    }
    m_counts.zero();
    m_mismatch.zero();
    grain_count_cuda_impl(m_ids.data(), m_counts.data(), m_n);
    int launches = 1;
    if (m_band > 0) {
      grain_band_cuda_impl(m_ids.data(), m_band_buf.data(), static_cast<int>(m_band),
                           m_nx, m_ny, m_nz, m_mismatch.data());
      ++launches;
    }
    m_profile.kernel_launches = launches;
    GrainDeviceCheck out;
    out.cell_count.resize(static_cast<std::size_t>(m_grains));
    m_counts.copy_to_host(out.cell_count.data(), out.cell_count.size());
    m_mismatch.copy_to_host(&out.mismatches, 1);
    m_profile.device_to_host_bytes =
        static_cast<std::size_t>(m_grains) * sizeof(int) + sizeof(int);
    out.profile = m_profile;
    return out;
  }

private:
  std::size_t m_n;
  std::size_t m_band;
  int m_grains;
  int m_nx;
  int m_ny;
  int m_nz;
  GrainDeviceProfile m_profile;
  pfc::core::DataBuffer<pfc::backend::CUDATag, int> m_ids;
  pfc::core::DataBuffer<pfc::backend::CUDATag, BandSite> m_band_buf;
  pfc::core::DataBuffer<pfc::backend::CUDATag, int> m_counts;
  pfc::core::DataBuffer<pfc::backend::CUDATag, int> m_mismatch;
};

} // namespace pfc::field::grain_interface
