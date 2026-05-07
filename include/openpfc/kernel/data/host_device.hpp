// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file host_device.hpp
 * @brief Portable `__host__ __device__` annotations for code that must
 *        compile with both a host C++ compiler and `nvcc`.
 *
 * @details
 * A user-defined model that wants its `rhs(t, g)` callable from both the
 * CPU `pfc::sim::for_each_interior` driver and the GPU
 * `pfc::sim::cuda::for_each_interior_device` kernel needs the function
 * to carry the CUDA `__host__ __device__` attributes when compiled by
 * `nvcc`, and **no attribute** when compiled by GCC / Clang on a CPU-only
 * build.
 *
 * This header expands the macro to the right thing in each translation
 * unit. The header itself depends on **nothing** outside the standard
 * preprocessor — it pulls in no CUDA headers, no `<cuda_runtime.h>`,
 * nothing from OpenPFC. A model can include it without losing its
 * "framework-free" status (the only OpenPFC dependency it gains is a
 * preprocessor token).
 *
 * Example:
 * @code
 * #include <openpfc/kernel/data/host_device.hpp>
 *
 * struct MyModel {
 *   OPENPFC_HD double rhs(double t, const MyGrads &g) const noexcept {
 *     return -k * g.x;
 *   }
 * };
 * @endcode
 *
 * The same `MyModel` is then usable from
 *   - `pfc::sim::for_each_interior(model, eval, du, t)` on the CPU
 *     (the macro is a no-op), and
 *   - `pfc::sim::cuda::for_each_interior_device(model, eval_pod, du, t)`
 *     on the GPU (the kernel calls `model.rhs(t, g)` from a `__device__`
 *     context).
 *
 * The matching macro `OPENPFC_INLINE_HD` is provided for the very common
 * case of an `inline` header function that is also `__host__ __device__`.
 *
 * @note `__CUDACC__` is defined by `nvcc` while compiling either a `.cu`
 *       file or a `.cpp` / `.hpp` translation unit it has been told to
 *       treat as CUDA code. CPU compilers never define it, so the macro
 *       collapses to an empty token everywhere else and the resulting
 *       function declaration is just a normal C++ function.
 */

#if defined(__CUDACC__)
#define OPENPFC_HD __host__ __device__
#define OPENPFC_INLINE_HD inline __host__ __device__
#else
#define OPENPFC_HD
#define OPENPFC_INLINE_HD inline
#endif
