// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Decision-order tests for gpu_aware_mpi.hpp. These call decide_gpu_aware_mpi()
// (uncached) so setenv is safe. Do not call runtime_mpi_gpu_aware() here: that
// path caches on first use. CUDA execution of the optional device probe is not
// available on LUMI; HIP can run the probe on a GPU partition.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <optional>
#include <string>

#include <openpfc/runtime/gpu/gpu_aware_mpi.hpp>

using pfc::gpu::decide_gpu_aware_mpi;
using pfc::gpu::gpu_aware_how_cstr;
using pfc::gpu::GpuAwareMpiHow;

namespace {

struct EnvGuard {
  std::string name;
  std::optional<std::string> previous;

  EnvGuard(const char *key, const char *value) : name(key) {
    if (const char *old = std::getenv(key)) {
      previous = old;
    }
    if (value != nullptr) {
      setenv(key, value, 1);
    } else {
      unsetenv(key);
    }
  }

  ~EnvGuard() {
    if (previous) {
      setenv(name.c_str(), previous->c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }

  EnvGuard(const EnvGuard &) = delete;
  EnvGuard &operator=(const EnvGuard &) = delete;
};

} // namespace

TEST_CASE("OPENPFC_ASSUME_GPU_AWARE_MPI=0 forces GPU-aware MPI off",
          "[gpu][mpi][aware]") {
  EnvGuard assume("OPENPFC_ASSUME_GPU_AWARE_MPI", "0");
  EnvGuard cray("MPICH_GPU_SUPPORT_ENABLED", "1");
  const auto d = decide_gpu_aware_mpi();
#if !defined(OpenPFC_MPI_CUDA_AWARE) && !defined(OpenPFC_MPI_HIP_AWARE)
  REQUIRE_FALSE(d.enabled);
  REQUIRE(d.how == GpuAwareMpiHow::CompileTimeOff);
#else
  REQUIRE_FALSE(d.enabled);
  REQUIRE(d.how == GpuAwareMpiHow::AssumeOff);
  REQUIRE(std::string(gpu_aware_how_cstr(d.how)) ==
          "OPENPFC_ASSUME_GPU_AWARE_MPI=0");
#endif
}

TEST_CASE("OPENPFC_ASSUME_GPU_AWARE_MPI=1 forces GPU-aware MPI on",
          "[gpu][mpi][aware]") {
#if !defined(OpenPFC_MPI_CUDA_AWARE) && !defined(OpenPFC_MPI_HIP_AWARE)
  const auto d = decide_gpu_aware_mpi();
  REQUIRE_FALSE(d.enabled);
  REQUIRE(d.how == GpuAwareMpiHow::CompileTimeOff);
#else
  EnvGuard assume("OPENPFC_ASSUME_GPU_AWARE_MPI", "1");
  const auto d = decide_gpu_aware_mpi();
  REQUIRE(d.enabled);
  REQUIRE(d.how == GpuAwareMpiHow::AssumeOn);
#endif
}

TEST_CASE("MPICH_GPU_SUPPORT_ENABLED=1 enables when no Open MPI query",
          "[gpu][mpi][aware]") {
#if !defined(OpenPFC_MPI_CUDA_AWARE) && !defined(OpenPFC_MPI_HIP_AWARE)
  const auto d = decide_gpu_aware_mpi();
  REQUIRE_FALSE(d.enabled);
  REQUIRE(d.how == GpuAwareMpiHow::CompileTimeOff);
#elif defined(OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT) ||                              \
    defined(OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT)
  EnvGuard assume("OPENPFC_ASSUME_GPU_AWARE_MPI", nullptr);
  const auto d = decide_gpu_aware_mpi();
  REQUIRE((d.how == GpuAwareMpiHow::OpenMpiQueryOn ||
           d.how == GpuAwareMpiHow::OpenMpiQueryOff));
#else
  EnvGuard assume("OPENPFC_ASSUME_GPU_AWARE_MPI", nullptr);
  EnvGuard probe("OPENPFC_PROBE_GPU_AWARE_MPI", nullptr);
  EnvGuard cray("MPICH_GPU_SUPPORT_ENABLED", "1");
  const auto d = decide_gpu_aware_mpi();
  REQUIRE(d.enabled);
  REQUIRE(d.how == GpuAwareMpiHow::CrayMpichEnv);
  REQUIRE(std::string(gpu_aware_how_cstr(d.how)) ==
          "MPICH_GPU_SUPPORT_ENABLED=1");
#endif
}
