// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "test_helpers.hpp"
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
namespace mechanics_test {
#if defined(OPENPFC_TEST_MECHANICS_HIP)
using Tag = pfc::backend::HIPTag;
using Space = pfc::HIPSpace;
inline bool available() { return pfc::gpu::test::is_hip_available(); }
#else
using Tag = pfc::backend::CUDATag;
using Space = pfc::CUDASpace;
inline bool available() { return pfc::gpu::test::is_cuda_available(); }
#endif
} // namespace mechanics_test
