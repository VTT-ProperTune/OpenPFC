// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_operations.hpp
 * @brief Padded-field blocks for `gmres`, `cg` and `newton`.
 *
 * `FieldVector` is one or more padded `data::Field`s on the same owned box.
 * `FieldOperations` is the solver contract (`make`, `copy`, `scale`, `axpy`,
 * `dot`, `dots`, `norm`) on that block. Algebra that does not care about
 * neighbours touches the whole padded buffer. `dot` / `norm` / `dots` sum
 * owned cells only, optionally weighted by a per-component mask (1 free, 0
 * pinned), and issue one `MPI_Allreduce` for the whole call.
 *
 * A stencil must not exchange the transient vectors the solver passes in.
 * `FieldHalo` keeps a scratch block and its `comm::HaloExchange`; the action
 * calls `halo(x)` and reads the returned fields. Direction sets such as
 * `Full2D` / `Full3D` are the caller's `HaloExchangeOptions`.
 *
 * Host, CUDA and HIP. Device dots copy one scalar (or one batch) back to the
 * host before the collective, the same way `solvers::gpu::VectorOperations`
 * does. This is not a second halo or reduction runtime.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/solvers/vector_operations.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/field_operations_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/vector_operations.hpp>
#endif

namespace pfc::solvers {

template <class MemorySpace> class FieldVector {
public:
  using FieldT = data::Field<double, MemorySpace>;

  FieldVector() = default;
  explicit FieldVector(std::vector<FieldT> fields) : m_fields(std::move(fields)) {}
  FieldVector(FieldVector &&) noexcept = default;
  FieldVector &operator=(FieldVector &&) noexcept = default;
  FieldVector(const FieldVector &o) { assign(o); }
  FieldVector &operator=(const FieldVector &o) {
    if (this != &o) assign(o);
    return *this;
  }

  [[nodiscard]] int count() const noexcept {
    return static_cast<int>(m_fields.size());
  }
  /// Owned cells across every field. Halo cells are not part of the length.
  [[nodiscard]] std::size_t size() const noexcept {
    if (m_fields.empty()) return 0;
    const auto n = m_fields[0].local_size();
    return m_fields.size() * static_cast<std::size_t>(n[0]) * n[1] * n[2];
  }
  FieldT &operator[](int i) { return m_fields.at(static_cast<std::size_t>(i)); }
  const FieldT &operator[](int i) const {
    return m_fields.at(static_cast<std::size_t>(i));
  }

private:
  void assign(const FieldVector &o) {
    bool same = m_fields.size() == o.m_fields.size();
    for (std::size_t c = 0; same && c < m_fields.size(); ++c)
      same = m_fields[c].size() == o.m_fields[c].size() &&
             m_fields[c].storage_halo() == o.m_fields[c].storage_halo();
    if (!same) {
      m_fields.clear();
      m_fields.reserve(o.m_fields.size());
      for (const auto &f : o.m_fields)
        m_fields.emplace_back(f.domain(), f.box(), f.storage_halo());
    }
    for (std::size_t c = 0; c < m_fields.size(); ++c) {
      const auto &src = o.m_fields[c];
      auto &dst = m_fields[c];
      if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
        std::copy(src.data(), src.data() + src.size(), dst.data());
      } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
        if (pfc::gpuMemcpy(dst.data(), src.data(), src.size() * sizeof(double),
                           pfc::gpuMemcpyDeviceToDevice) != pfc::gpuSuccess)
          throw std::runtime_error("FieldVector: device copy failed");
        dst.note_device_write();
#endif
      }
    }
  }

  std::vector<FieldT> m_fields;
};

template <class MemorySpace> class FieldOperations {
public:
  using Vector = FieldVector<MemorySpace>;
  using FieldT = typename Vector::FieldT;

  ReductionMetrics reductions;
  std::size_t scalar_transfer_bytes = 0;

  FieldOperations(const Domain &domain, const Box3i &box, int count, int halo,
                  MPI_Comm comm = MPI_COMM_NULL)
      : m_domain(domain), m_box(box), m_count(count), m_halo(halo), m_comm(comm),
        m_masks(static_cast<std::size_t>(count), nullptr) {
    if (count < 1)
      throw std::invalid_argument("FieldOperations: count must be >= 1");
    if (halo < 0) throw std::invalid_argument("FieldOperations: halo must be >= 0");
  }

  /// Weight component `i` in `dot` / `norm` / `dots`. Null clears the weight.
  /// The mask is read in this memory space and must outlive the calls.
  void set_mask(int i, const FieldT *mask) {
    m_masks.at(static_cast<std::size_t>(i)) = mask;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return static_cast<std::size_t>(m_count) * m_box.size[0] * m_box.size[1] *
           m_box.size[2];
  }

  Vector make(std::size_t n) const {
    if (n != size())
      throw std::invalid_argument("FieldOperations: vector size mismatch");
    std::vector<FieldT> f;
    f.reserve(static_cast<std::size_t>(m_count));
    for (int c = 0; c < m_count; ++c) {
      f.emplace_back(m_domain, m_box, m_halo);
      combine(f.back(), 0.0, nullptr, 0.0);
    }
    return Vector(std::move(f));
  }

  void copy(const Vector &x, Vector &y) const {
    check(x);
    if (y.count() != m_count)
      y = make(size());
    else
      check(y);
    for (int c = 0; c < m_count; ++c) {
      if (x[c].data() == y[c].data()) continue;
      if constexpr (is_host) {
        std::copy(x[c].data(), x[c].data() + x[c].size(), y[c].data());
      } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
        gpu_check(pfc::gpuMemcpyAsync(y[c].data(), x[c].data(),
                                      x[c].size() * sizeof(double),
                                      pfc::gpuMemcpyDeviceToDevice, nullptr));
        y[c].note_device_write();
#endif
      }
    }
  }

  void scale(Vector &x, double a) const {
    check(x);
    for (int c = 0; c < m_count; ++c) combine(x[c], 0.0, nullptr, a);
  }

  void axpy(Vector &y, double a, const Vector &x) const {
    check(x);
    check(y);
    for (int c = 0; c < m_count; ++c) combine(y[c], a, &x[c], 1.0);
  }

  double dot(const Vector &x, const Vector &y) {
    double out = 0;
    dots(&x, 1, y, &out);
    return out;
  }

  /// Several inner products against `y`, then one collective.
  /// Coefficients are taken before any of those columns are updated.
  void dots(const Vector *columns, int count, const Vector &y, double *out) {
    if (count < 1 || columns == nullptr || out == nullptr)
      throw std::invalid_argument("FieldOperations: empty inner-product batch");
    check(y);
    const auto start = std::chrono::steady_clock::now();
    std::vector<double> local(static_cast<std::size_t>(count), 0.0);
    for (int column = 0; column < count; ++column) {
      check(columns[column]);
      double sum = 0.0;
      if constexpr (is_host) {
        for (int c = 0; c < m_count; ++c)
          sum += host_dot(columns[column][c], y[c],
                          m_masks[static_cast<std::size_t>(c)]);
      } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
        sum = device_dot(columns[column], y);
#endif
      }
      local[static_cast<std::size_t>(column)] = sum;
    }
    if (m_comm == MPI_COMM_NULL) {
      for (int column = 0; column < count; ++column)
        out[column] = local[static_cast<std::size_t>(column)];
    } else {
      if (MPI_Allreduce(local.data(), out, count, MPI_DOUBLE, MPI_SUM, m_comm) !=
          MPI_SUCCESS)
        throw std::runtime_error("FieldOperations: MPI reduction failed");
      ++reductions.calls;
      reductions.seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count();
    }
  }

  double norm(const Vector &x) { return std::sqrt(dot(x, x)); }

  [[nodiscard]] MPI_Comm communicator() const noexcept { return m_comm; }
  [[nodiscard]] const Domain &domain() const noexcept { return m_domain; }
  [[nodiscard]] const Box3i &box() const noexcept { return m_box; }
  [[nodiscard]] int count() const noexcept { return m_count; }
  [[nodiscard]] int halo() const noexcept { return m_halo; }

private:
  static constexpr bool is_host = std::is_same_v<MemorySpace, HostSpace>;

  void check(const Vector &x) const {
    if (x.count() != m_count || x.size() != size())
      throw std::invalid_argument("FieldOperations: vector layout mismatch");
    for (int c = 0; c < m_count; ++c) {
      const auto n = x[c].local_size();
      if (n[0] != m_box.size[0] || n[1] != m_box.size[1] || n[2] != m_box.size[2] ||
          x[c].storage_halo() != m_halo)
        throw std::invalid_argument("FieldOperations: vector layout mismatch");
    }
  }

  static void combine(FieldT &y, double a, const FieldT *x, double b) {
    const std::size_t n = y.size();
    if constexpr (is_host) {
      double *p = y.data();
      const double *q = x ? x->data() : nullptr;
      for (std::size_t i = 0; i < n; ++i)
        p[i] = (a == 0.0 ? 0.0 : a * q[i]) + (b == 0.0 ? 0.0 : b * p[i]);
    } else {
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
      device_combine(y.data(), a, x ? x->data() : nullptr, b, n);
      y.note_device_write();
#endif
    }
  }

  static double host_dot(const FieldT &x, const FieldT &y, const FieldT *m) {
    const auto n = x.local_size();
    double s = 0.0;
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const std::size_t q = x.idx(i, j, k);
          const double w = m ? m->data()[q] : 1.0;
          if (w != 0.0) s += w * x.data()[q] * y.data()[q];
        }
    return s;
  }

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
  using Tag = typename FieldT::backend_tag;

  static void gpu_check(pfc::gpuError_t e) {
    if (e != pfc::gpuSuccess)
      throw std::runtime_error(std::string("FieldOperations: ") +
                               pfc::gpuGetErrorString(e));
  }

  static void device_combine(double *y, double a, const double *x, double b,
                             std::size_t n) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, CUDASpace>)
      gpu::detail::combine_cuda(y, a, x, b, n);
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, HIPSpace>)
      gpu::detail::combine_hip(y, a, x, b, n);
#endif
  }

  gpu::fops::Geometry geometry(const FieldT &f) const {
    gpu::fops::Geometry g;
    for (int a = 0; a < 3; ++a) {
      g.n[a] = m_box.size[a];
      g.pitch[a] = f.padded_extent(a);
    }
    g.hw = f.storage_halo();
    return g;
  }

  double device_dot(const Vector &x, const Vector &y) {
    const int blocks =
        gpu::fops::dot_blocks(size() / static_cast<std::size_t>(m_count));
    const std::size_t need = static_cast<std::size_t>(blocks) * m_count;
    if (m_partials.size() < need)
      m_partials = pfc::core::DataBuffer<Tag, double>(need);
    for (int c = 0; c < m_count; ++c) {
      const double *m = m_masks[static_cast<std::size_t>(c)]
                            ? m_masks[static_cast<std::size_t>(c)]->data()
                            : nullptr;
      double *partial = m_partials.data() + static_cast<std::size_t>(c) * blocks;
#if defined(OpenPFC_ENABLE_CUDA)
      if constexpr (std::is_same_v<MemorySpace, CUDASpace>)
        gpu::fops::owned_dot_cuda(geometry(x[c]), x[c].data(), y[c].data(), m,
                                  partial, blocks);
#endif
#if defined(OpenPFC_ENABLE_HIP)
      if constexpr (std::is_same_v<MemorySpace, HIPSpace>)
        gpu::fops::owned_dot_hip(geometry(x[c]), x[c].data(), y[c].data(), m,
                                 partial, blocks);
#endif
    }
    double r = 0.0;
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, CUDASpace>)
      r = gpu::fops::sum_cuda(m_partials.data(), static_cast<int>(need));
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, HIPSpace>)
      r = gpu::fops::sum_hip(m_partials.data(), static_cast<int>(need));
#endif
    scalar_transfer_bytes += sizeof(double);
    return r;
  }

  pfc::core::DataBuffer<Tag, double> m_partials;
#endif

  Domain m_domain;
  Box3i m_box;
  int m_count;
  int m_halo;
  MPI_Comm m_comm;
  std::vector<const FieldT *> m_masks;
};

/**
 * Persistent scratch plus one `HaloExchange` for stencil actions.
 * `operator()(x)` copies `x` and exchanges. The scratch outlives the
 * transient vectors created inside `gmres` / `cg` / `newton`.
 */
template <class MemorySpace> class FieldHalo {
public:
  using Ops = FieldOperations<MemorySpace>;
  using Vector = typename Ops::Vector;
  using FieldT = typename Ops::FieldT;
  using Exchange = comm::HaloExchange<MemorySpace, double>;

  FieldHalo(const Ops &ops, const decomposition::Decomposition &decomp, int rank,
            MPI_Comm comm, comm::HaloExchangeOptions options)
      : m_ops(ops), m_scratch(ops.make(ops.size())) {
    std::vector<FieldT *> fields;
    for (int c = 0; c < m_scratch.count(); ++c) fields.push_back(&m_scratch[c]);
    m_exchange =
        std::make_unique<Exchange>(std::move(fields), decomp, rank, comm, options);
  }

  const Vector &operator()(const Vector &x) {
    m_ops.copy(x, m_scratch);
    m_exchange->exchange();
    return m_scratch;
  }

  Vector &scratch() noexcept { return m_scratch; }
  void exchange() { m_exchange->exchange(); }

private:
  const Ops &m_ops;
  Vector m_scratch;
  std::unique_ptr<Exchange> m_exchange;
};

} // namespace pfc::solvers
