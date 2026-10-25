<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CMake options reference

High-level switches for OpenPFC. Defaults are chosen for a normal **MPI + HeFFTe** developer build. Full context (toolchains, CUDA/HIP, HeFFTe layout) remains in **[`INSTALL.md`](../../INSTALL.md)**.

## What to build

| Option | Default | Meaning |
|--------|---------|---------|
| **`OpenPFC_BUILD_APPS`** | ON | **`apps/`** (tungsten, aluminumNew, allen_cahn, …) |
| **`OpenPFC_BUILD_EXAMPLES`** | ON | **`examples/`** executables |
| **`OpenPFC_BUILD_TESTS`** | ON | **`tests/`** + Catch2 |
| **`OpenPFC_BUILD_BENCHMARKS`** | OFF | Extra benchmarks under **`tests/benchmarks/`** (slow) |
| **`OpenPFC_BUILD_DOCUMENTATION`** | ON | Doxygen **`docs`** target (when Doxygen available) |

Defined in **`cmake/BuildOptions.cmake`** and **`cmake/Dependencies.cmake`**.

## Features and dependencies

| Option | Default | Meaning |
|--------|---------|---------|
| **`OpenPFC_ENABLE_MPI`** | ON | MPI required for supported builds; **OFF is unsupported** — see **`INSTALL.md`**. |
| **`OpenPFC_ENABLE_HEFFTE`** | ON | Distributed FFT via HeFFTe. **OFF** is an FD-only / kernel-only build: no HeFFTe find, no `fft.cpp`, no spectral apps/examples. Catch2 TUs that include `fft_fftw.hpp` / `<heffte.h>` are skipped; FD/halo/k-space tests still build. FD apps (`heat3d_fd*`, `kobayashi_fd*`, `wave2d`, `allen_cahn`) still build. `./scripts/build.sh --cpu --no-heffte` is the script path. |
| **`OpenPFC_ENABLE_CUDA`** | OFF | CUDA toolkit/runtime support and CUDA apps that do not require spectral FFTs (for example FD/kernels-only apps). CUDA spectral targets are enabled only when CUDA HeFFTe is also found (`OpenPFC_ENABLE_CUDA_SPECTRAL=ON` in the configure summary). |
| **`OpenPFC_ENABLE_HIP`** | OFF | ROCm/HIP and **`tungsten_hip`**, **`allen_cahn_hip`**, etc. |
| **`OpenPFC_ENABLE_HDF5`** | OFF | HDF5 export for profiling dumps (see **`performance_profiling.md`**) |
| **`OpenPFC_FETCH_HEFFTE`** | (see CMake) | Fetch/build HeFFTe via CMake when not found (see **`INSTALL.md`**) |
| **`OpenPFC_ENABLE_NAN_CHECK`** | OFF | NaN checks beyond Debug (see **`debugging.md`**). When active (Debug or this option), `NAN_CHECK_ENABLED` is PUBLIC on **`OpenPFC::openpfc`**. |
| **`OpenPFC_ENABLE_CODE_COVERAGE`** | ON where supported | Coverage targets; often OFF on clusters ( **`INSTALL.md`**) |

GPU-aware MPI toggles (`OpenPFC_MPI_CUDA_AWARE` / `OpenPFC_MPI_HIP_AWARE`) and CUDA/HIP compiler discovery are described in **`INSTALL.md`** and **`INSTALL.LUMI.md`**. When CUDA or HIP is enabled, `OpenPFC_ENABLE_CUDA` / `OpenPFC_ENABLE_HIP` and the matching MPI-aware flags are **PUBLIC** compile definitions on **`OpenPFC::openpfc`** (and the vendor kernel libraries), so `find_package(OpenPFC)` consumers see the same macros as the in-tree build. `OpenPFC_ENABLE_GPU_AUTOTUNING` is exported the same way when that option is ON.

## Library and profiling

| Setting | Meaning |
|---------|---------|
| **`CMAKE_BUILD_TYPE`** | Single-config default is **RelWithDebInfo** if unset (`cmake/ProjectSetup.cmake`). |
| **`BUILD_SHARED_LIBS`** | OFF = static **`libopenpfc`** (typical); ON = shared. |
| **`OpenPFC_PROFILING_LEVEL`** | `0` / `1` / `2` — compile-time stripping of **`OPENPFC_PROFILE`** macros (`cmake/LibraryConfiguration.cmake`). |

Implementation detail: the `openpfc` library merges CMake **`OBJECT`** targets **`openpfc_kernel_obj`** and **`openpfc_frontend_obj`** (`cmake/LibraryConfiguration.cmake`). Downstream **`find_package(OpenPFC)`** usage is unchanged — link **`OpenPFC::openpfc`** only; the object targets are build internals, not installed.

## Development

| Option | Default | Meaning |
|--------|---------|---------|
| **`OpenPFC_DEVELOPMENT`** | OFF | When ON, enables **`compile_commands.json`** export and dev version suffix. |
| **`OpenPFC_ENABLE_ADDRESS_SANITIZER`** | OFF | Sanitizers (see **`CompilerSettings.cmake`**, **`INSTALL.md`** `dev-asan` preset). |

## Presets

**[`CMakePresets.json`](../../CMakePresets.json)** and **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`** pin cluster-friendly configure flags; see **`INSTALL.md`** (“VS Code / Cursor on tohtori”).

## See also

- **[`quickstart.md`](../quickstart.md)** — minimal configure line  
- **[`testing.md`](../development/testing.md)** — **`OpenPFC_BUILD_TESTS`**, CTest, Catch2 filters  
- **[`troubleshooting.md`](../troubleshooting.md)** — when configure fails  

## Package source identity

`find_package(OpenPFC CONFIG REQUIRED)` exports `OpenPFC_SOURCE_REVISION`
(the full configure-time Git SHA, or `unknown`) and `OpenPFC_SOURCE_STATE`
(`clean`, `dirty`, or `unknown`). Installed and build-tree consumers read
identical metadata. Record both when identifying numerical dependencies;
a dirty checkout is not reproduced by its HEAD alone. Source archives
without Git metadata report `unknown`. Reconfigure after changing sources
before building or installing; these variables describe the configure-time
tree and do not detect edits made after configuration.

## Installed SDK build manifest and compatibility

Every configuration writes a schema-versioned JSON manifest to
`<build>/share/openpfc/<configuration>/build-manifest.json`. Installation selects
that configuration and installs it as `<prefix>/share/openpfc/build-manifest.json`.
Use separate install prefixes for CPU, CUDA, HIP and build configurations.
`OpenPFC_BUILD_MANIFEST` points to the file after `find_package`; it needs neither
the original checkout nor its cache. The installed location remains valid when
the prefix moves. `OpenPFC_BUILD_CONFIGURATIONS` lists provider configurations;
in a multi-configuration build tree, select the corresponding manifest explicitly
when checking another configuration.

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_model LANGUAGES C CXX)
find_package(OpenPFC 0.5 CONFIG REQUIRED)
file(READ "${OpenPFC_BUILD_MANIFEST}" sdk)
string(JSON compiler_version GET "${sdk}" toolchains CXX CMAKE_CXX_COMPILER_VERSION)
openpfc_check_compatibility(RESULT status REPORT report REQUIRED BACKEND CPU)
message(STATUS "${report}")
add_executable(my_model main.cpp)
target_link_libraries(my_model PRIVATE OpenPFC::openpfc)
```

Provision a compatible installed SDK and its dependencies, then set `OpenPFC_DIR`
or `CMAKE_PREFIX_PATH`. A missing SDK or dependency is a provisioning error;
consumer configuration does not fetch or rebuild OpenPFC. Link public targets to
obtain includes, C++20, backend/ABI definitions and transitive dependencies.
The libstdc++ dual-ABI mode, and debug mode when enabled, are public requirements.
Do not copy historical optimization, architecture or compiler-path settings into
consumer flags.

Schema 1 has these top-level fields. Unknown evidence is the string `unknown`
(or a probe status beginning `unknown:`); disabled capabilities are JSON booleans.
Empty flags/definitions mean no explicit setting. CMake list values use semicolon
separators. `targets` preserves generator expressions verbatim, for their original
language/configuration context; they are not JSON expressions to evaluate.

| Field | Meaning |
|---|---|
| `version`, `source_revision`, `source_state`, `configuration` | Package/source identity and selected artifact configuration. |
| `capabilities` | Effective CPU, MPI, CUDA/HIP runtime APIs, HeFFTe, FFTW, HDF5 and vendor spectral facilities. Device targets, rather than cached toolkit discoveries, determine GPU availability. |
| `components` | Selected applications, examples, tests, profiling, NaN checks and other build components. |
| `toolchains` | Exact compiler families/versions, launchers, target/host compiler hints, implicit link libraries and GPU architecture targets. Disabled GPU toolchains do not expose stale flags. |
| `platform` | System, target triple, pointer size, toolchain hint, linker identity and compiler-resolved runtime candidates. Candidate paths/hashes are diagnostics, not proof that the dynamic loader used them. |
| `flags` | Configuration-specific C/C++/CUDA/HIP and linker flags, including explicit numerical options such as `--fmad`, `--use_fast_math` or `-ffp-contract` when supplied. Absent overrides leave compiler defaults in force. Directory options retain their CMake expressions; target records describe individual libraries. |
| `targets` | Actual SDK targets and their transitive imported targets, include/definition/link interfaces, resolved imported locations/SONAMEs where provided, target options, device-source options, standards, separable compilation and device-link policy. |
| `mpi` | C and C++ wrappers, compile/link arguments, definitions, resolved libraries, file SHA256 identities/SONAMEs and GPU-awareness settings/probe evidence. MPI_CXX_VERSION is the MPI *standard* version; vendor versions come from `host_probe`. |
| `stack` | Available package version/variant discovery hints for HeFFTe, FFTW, HDF5, CUDA and ROCm. These hints alone do not imply capability; check `capabilities` and actual `targets`. |
| `host_probe`, `probe_status` | Compiled standard-library identity/version/ABI/debug mode, MPI header vendor/version and `MPI_Get_library_version` from an executable host probe. |

Only selected build variables and dependency evidence are recorded; the manifest
does not dump the environment or cache. Credential-like assignments and URL
userinfo in diagnostic strings are redacted. Absolute paths are diagnostic and are
never used by the checker to locate consumer compilers or dependencies. As with
source provenance, reconfigure after changing the toolchain, flags or sources.

`openpfc_check_compatibility` returns `compatible`, `incompatible` or `unverified`
through `RESULT`, with concrete reasons in `REPORT`. `REQUIRED` rejects either
non-compatible outcome. It compiles, links and runs a tiny program against the
public installed target, checking standard-library ABI/debug mode, pointer size,
MPI implementation, header/library version evidence and resolved MPI library file
identities. Equal MPI version strings or SONAMEs alone never yield success.
A file SHA256 identifies bytes; it is not an invented cross-vendor MPI ABI ID.
Different library files, failed probes and unknown identities remain unverified.
GCC and Clang can pass with the same demonstrated runtime/ABI evidence; exact
compiler-version equality is not required. This bounded host probe is not a
proof of every possible C++ ABI interaction or numerical equivalence.

Optional arguments are `CONFIGURATION <name>`, `MANIFEST <path>`,
`BACKEND CPU|CUDA|HIP`, and `GPU_ARCHITECTURE <target>`. An absent requested
backend is incompatible. GPU requests remain unverified until the consumer runs
its device probes; non-exact architecture matches also need runtime validation
(for example, CUDA virtual ISA forward compatibility). The checker does not
confuse these unknowns with an incompatible GPU. Cross-compiling skips execution
and reports unverified; run the installed `openpfc_sdk_probe.cpp` with the target
compiler/MPI/public target on the target machine to collect host evidence.

For GPU consumers, extend the host probe with a translation unit using the
selected CUDA/HIP compiler and the SDK's public runtime types, link the installed
kernel target (`OpenPFC::openpfc_gpu_kernels` for CUDA or
`OpenPFC::openpfc_hip_kernels` for HIP), then call an exported fill kernel and
verify its output on each intended device architecture. This checks CUDA's host compiler and device-link/runtime compatibility, or HIP's
host standard library and ROCm dependencies. Test spectral APIs separately if
needed: a runtime/kernel SDK does not guarantee a HeFFTe GPU backend. Finally,
run a two-rank device-buffer send/receive with the site's actual MPI launcher and
transport, and validate the received values. Use the existing
`verify_gpu_aware_mpi` program where available. Neither a version label nor
`MPIX_Query_cuda_support` proves that UCX/Cray MPICH transports can move device
buffers; `mpi.gpu_aware_runtime` therefore remains explicitly unknown.

The relocatable consumer regression in
[`tests/packaging/sdk_consumer`](../../tests/packaging/sdk_consumer) exercises the
public contract, manifest queries and mismatch diagnostics. It is registered as
`SDK_InstalledCompatibility` and runs through `scripts/build.sh` with tests enabled.

## Application compiler runtime override

Applications normally use the C++ runtime selected by the configured compiler.
For a mixed compiler/vendor toolchain, set `OpenPFC_APP_CXX_RUNTIME` to an
existing runtime library in a local CMake preset or with `--cmake-arg`.
The Allen–Cahn, Wave2D and Kobayashi application links add that library and its
RPATH only when explicitly configured. No site-specific paths are searched.
