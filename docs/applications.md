<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Applications (`apps/`)

These are **full programs** (not the small `examples/` tutorials). They are built when **`OpenPFC_BUILD_APPS=ON`** (default). Binaries usually install under `<prefix>/bin` when you **`cmake --install`**.

All of them expect **MPI** for realistic runs unless noted. Match the same compiler/MPI/HeFFTe stack you used to build OpenPFC (see **[`INSTALL.md`](../INSTALL.md)**).

## Tungsten PFC (`apps/tungsten/`)

| Target | When available |
|--------|----------------|
| `tungsten` | Always (CPU) |
| `tungsten_cuda` | `OpenPFC_ENABLE_CUDA` and CUDA toolkit |
| `tungsten_hip` | `OpenPFC_ENABLE_HIP` and ROCm |
| `verify_gpu_aware_mpi` | HIP + MPI device-buffer smoke test (LUMI-style workflows) |

**Inputs:** JSON under [`apps/tungsten/inputs_json/`](../apps/tungsten/inputs_json/README.md); TOML may live alongside in `inputs_toml/` where present.

**Run (example):**

```bash
mpirun -n 4 ./apps/tungsten/tungsten /path/to/config.json
```

GPU-aware MPI and Slurm examples: **[`INSTALL.LUMI.md`](INSTALL.LUMI.md)**, **[`lumi_slurm/README.md`](lumi_slurm/README.md)**.

## Aluminum (`apps/aluminumNew/`)

| Target | Notes |
|--------|--------|
| `aluminumNew` | Sample application using OpenPFC + nlohmann_json + HeFFTe |

See [`apps/aluminumNew/README.md`](../apps/aluminumNew/README.md) (minimal; source and CMake are the reference).

## Allen–Cahn (`apps/allen_cahn/`)

| Target | When available |
|--------|----------------|
| `allen_cahn` | CPU |
| `allen_cahn_cuda` | CUDA enabled |
| `allen_cahn_hip` | HIP enabled |

## Choosing apps vs examples

- Use **`examples/`** to learn APIs and patterns in short programs.
- Use **`apps/`** when you want a **deployable binary** with model-specific parameters and inputs closer to production PFC runs.

## See also

- **[`quickstart.md`](quickstart.md)** — run an app after building
- **[`extending_openpfc/README.md`](extending_openpfc/README.md)** — how to build your own app the same way
