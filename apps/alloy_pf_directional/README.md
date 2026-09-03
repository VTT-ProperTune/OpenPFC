<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# alloy_pf_directional

Al-Cu FTA on a regular grid (FD). This is the main directional phase-field application in this tree.

Dilute Al-Cu frozen-temperature alloy solver ([Pinomaa et al., J. Crystal Growth 2020](https://doi.org/10.1016/j.jcrysgro.2019.125412); please verify volume/page before citing). Explicit Euler, Ji isotropic operators, Glasner \(\psi\), cubic anisotropy. **This is not the spectral / HeFFTe path.**

## Start here (locked bicrystal)

The app starting point is the **locked** two-grain strip [`benchmark/ly3.2_w10nm_bicrystal/`](benchmark/ly3.2_w10nm_bicrystal/): \(L=12\times 3.2\,\mu\mathrm{m}\), \(W_0=10\,\mathrm{nm}\), \(G=3\times 10^6\,\mathrm{K/m}\), \(V_p=0.4\,\mathrm{m/s}\), \(\pm 30^\circ\), \(\Delta t=0.2\,\tau_0\), **noise off**. LUMI-C gold: \(n=33457\), \(t=72.9\,\mu\mathrm{s}\), \(x_\mathrm{tip}=11.94\,\mu\mathrm{m}\), stop `wall_c`.

Do not retune FTA physics, the Zhong mapping, or the two-grain IC until a full-length rerun still matches the strip/front figures and that stop. New features (MPI/HIP, window, 3D) fork from this case; they do not replace it.

```bash
./apps/alloy_pf_directional/scripts/run_benchmark.sh          # full length
QUICK=1 ./apps/alloy_pf_directional/scripts/run_benchmark.sh # 400 steps (seeds only)
./apps/alloy_pf_directional/scripts/run_benchmark.sh --plot-only
# same locked physics:
builds/macos-cpu-release/apps/alloy_pf_directional/alloy_pf_directional_openmp
builds/macos-cpu-release/apps/alloy_pf_directional/alloy_pf_directional_openmp start
```

CLI `repro` / `ctest -R alloy-pf-directional-repro` is a **40-step last-bit** check on a \(128\times 64\) noisy slice. It is not the product figure. `ds` / `bicrystal` remain env-driven research CLIs (LUMI campaigns). The 2× noisy ensemble is a separate campaign.

## Two grains and grain–grain coupling

Each grain has its own order parameter \(\phi_\alpha\in[-1,1]\) (\(\alpha=1,2\)). The combined solid–liquid field is the Zhong mapping
\(\psi=-1+\sum_\alpha(1+\phi_\alpha)=1+\phi_1+\phi_2\) (clamped to \([-1,1]\)), used for solute interpolation. The \(\phi_\alpha\) residual keeps \(g'(\phi_\alpha)\) (not \(g'(\psi)\)) so there is no spurious fixed point at \(\phi_\alpha\approx-0.5\).

Where two grains overlap, a repulsive term
\(-\omega\,\hat\phi_\alpha\hat\phi_\beta^2\) with \(\hat\phi=(1+\phi)/2\) is added to the \(\phi_\alpha\) equation ([Zhong et al., Nat. Commun. 16, 11698 (2025)](https://doi.org/10.1038/s41467-025-66655-2), Eq. 4). \(\omega(T)\) is chosen so a one-dimensional grain boundary with \(\psi=1\) (\(\phi_2=-\phi_1\), fully solid) has equal well depths — the same method as their Eq. 5, evaluated for this app’s FTA interpolant:
\(\omega=(32/5)\,\lambda\,\mathrm{therm}/(1-k_e)\), with \(\mathrm{therm}=(T-T_l)/(m_l c_\infty)>0\) in the solid. Default: this local \(\omega(x,t)\). Override with a constant via `OPENPFC_ALCU_OMEGA`.

Two-grain IC: independent semicircles (2D) / hemispheres (3D) on the left wall \(x=0\) at \(y=0.25 L_y\) and \(0.75 L_y\). No exclusive Voronoi cut (that inserted a \(\psi\) jump along the midline). Radius is `OPENPFC_ALCU_SEED`, shrunk if needed so the \(\phi=0\) contours stay at least \(16 W_0\) apart. Orientations \(\phi_1=\pm\theta\) (default \(30^\circ\); `OPENPFC_ALCU_THETA`).

## Build

Always go through `./scripts/build.sh` at the repo root (see `AGENTS.md`).

| Binary | When | Role |
|--------|------|------|
| `alloy_pf_directional_openmp` | OpenMP found | Laptop / LUMI-C single-node baseline (2D or 3D) |
| `alloy_pf_directional_mpi` | always (MPI) | Multi-rank CPU, `Field` + 26-neighbor host halo |
| `alloy_pf_directional_hip` | `OpenPFC_ENABLE_HIP` | Multi-GCD LUMI-G; device-resident + `FullPaddedDeviceHalo` |

```bash
./scripts/build.sh                         # local / Tohtori CPU
# LUMI-G HIP (login configure, GPU compile via the script):
./scripts/build.sh --machine=lumi --with-rocm
```

Targets after configure: `alloy_pf_directional_openmp`, `alloy_pf_directional_mpi`, and `alloy_pf_directional_hip` if HIP is on.

## Run

```bash
# Locked starting point (same as `start` / `benchmark`)
./apps/alloy_pf_directional/scripts/run_benchmark.sh
# or: alloy_pf_directional_openmp results/alloy_pf_directional/benchmark/ly3.2_w10nm_bicrystal

# Research CLI (env; 1-grain `ds` or 2-grain `bicrystal`)
builds/release/apps/alloy_pf_directional/alloy_pf_directional_openmp smoke
builds/release/apps/alloy_pf_directional/alloy_pf_directional_openmp ds results/alloy_pf_directional/ds


# MPI CPU (same CLI/env)
mpirun -np 4 builds/release/apps/alloy_pf_directional/alloy_pf_directional_mpi ds results/alloy_pf_directional_mpi

# HIP (LUMI-G: 1 rank / GCD, ntasks-per-node=8)
srun -n 8 builds/.../alloy_pf_directional_hip ds results/alloy_pf_directional_hip
```

3D brick: `OPENPFC_ALCU_LZ=8e-7` or `OPENPFC_ALCU_NDIM=3` / `OPENPFC_ALCU_NZ`. Default `ds` stays **2D** (`Lz=0`).

Moving window: `OPENPFC_ALCU_WINDOW=1` and optional `OPENPFC_ALCU_WINDOW_NX`.  
Block skip: `OPENPFC_ALCU_BLOCK_SKIP=16` or `32` (OpenMP/MPI; HIP still launches the full grid).  
Timed window (I/O off): `OPENPFC_ALCU_WARMUP=10 OPENPFC_ALCU_TIMED_STEPS=50`.  
Default: persist \(e^u\) and \(u\) (not anisotropy fluxes). `OPENPFC_ALCU_STORE_EU=0` recomputes them; `OPENPFC_ALCU_STORE_AUX=1` also stores \(j_x,j_y[,j_z]\). Iso solute materializes nodal \(\alpha\) and \(\beta\) once per cell. On `Nz=1` the OpenMP engine skips z-halo copies (full-plane traffic that dominated LUMI-C wall time) and MPI uses `Full2D` instead of 26-neighbour exchange. `alloy_pf_directional_openmp` prints `ALCU_PERF`. See `scripts/profile_store_aux.sh`.

## Stored vs recomputed fields

Memory in 3D is dominated by full bricks. Default persistent auxiliaries are \(e^u\) and \(u\) (two extra bricks). Ji concentration fluxes are **not** stored.

| Persistent | Role |
|------------|------|
| \(\phi\) (and \(\psi\) if Glasner) per grain | Evolved phase |
| \(c\) | Solute |
| \(\partial_t\phi\) per grain | Antitrapping after the Euler update |
| \(dc\) (scratch) | Jacobi increment so \(c\) stays frozen while the stencil reads neighbor \(u\) |
| \(e^u\), \(u=\log(e^u)\) | Default stored; `STORE_EU=0` recomputes |
| iso nodal \(\alpha_\mathrm{diff}\), \(\alpha_\mathrm{at}\), \(\beta\) | Filled once per cell each solute step (2D: xy plane only) |

| Recomputed each use | Role |
|---------------------|------|
| Anisotropy face fluxes \(j_x,j_y,j_z\) | From \(\nabla\phi\) on the face (`STORE_AUX=1` to persist) |
| \(\nabla\phi\), \(\lvert\nabla\phi\rvert^2\), Ji stencils | Read \(\phi/\psi\) or nodal \(\alpha,\beta\) |

Halo width is **1**. Ji \(\bar S_{1,2,0}\) needs corners, so MPI uses `FullPaddedHaloExchanger` / HIP `FullPaddedDeviceHalo`. 2D CPU MPI selects `Full2D` (8 in-plane dirs); 3D keeps `Full3D`.

## Correctness before 3D production

**1. Nz=1 vs 2D (do this first)**

The 3D engine with `Nz=1` and `n_dim=2` is the 2D-equivalent path (2D Ji operators, 2D \(\mathrm{d}V\), same IC). The script runs a small 1-grain `ds` brick (periodic \(y\), noise off) on OpenMP and MPI `np=1`:

```bash
BUILD=builds/release ./apps/alloy_pf_directional/scripts/check_nz1_vs_2d.sh
# or, if your tree is elsewhere:
BUILD=builds/macos-cpu-release ./apps/alloy_pf_directional/scripts/check_nz1_vs_2d.sh
```

Documented tolerances in that script: relative mass \(\sim 10^{-9}\), tip \(\sim 10^{-10}\,\mathrm{m}\), checksums \(\sim 10^{-8}\). Do **not** set `OPENPFC_ALCU_NDIM=3` with `Nz=1` if you want a 2D match (`n_dim` changes the CFL \(\mathrm{d}t\)). The two-grain `smoke` IC (no-flux \(y\)) is stiffer and is not this check.

**2. Last-bit OpenMP (tiny noisy slice)**

```bash
BUILD=builds/macos-cpu-release ./apps/alloy_pf_directional/scripts/check_repro.sh
# or: ctest -R alloy-pf-directional-repro
```

Optional: same **box** as the gold, noise on, capped steps (still seeds at 400):

```bash
BUILD=builds/macos-cpu-release STEPS=400 NTHREADS=8 \
  ./apps/alloy_pf_directional/scripts/check_bicrystal_repro.sh
```

Do not expect last-bit identity vs LUMI (Cray vs AppleClang). Last-bit is the same binary, same thread count, twice.

**3. Light 3D smoke**

Tiny brick, coarse vs \(\mathrm{d}x/2\), a few tens of steps — mass bounded, \(\phi\) in range, no NaNs:

```bash
BUILD=builds/release ./apps/alloy_pf_directional/scripts/check_3d_smoke.sh
```

This is not a journal convergence study.

## After this 2D baseline is nailed

Treat `ly3.2_w10nm_bicrystal` as the **locked starting point**: do not “improve” FTA physics, the Zhong mapping, or the two-grain IC until a full-length rerun still looks like the strip/front figures and still stops on `wall_c` near \(x_\mathrm{tip}=11.94\,\mu\mathrm{m}\). Then add features in this order:

1. **Backends on the frozen cases, not a new box.** MPI `np=1` vs OpenMP on `repro` (fields), then a capped strip. HIP vs CPU on LUMI-G (`check_hip_vs_cpu.sh`) with looser tolerances, not last-bit. Same \(G,V_p,W_0,\theta\).
2. **Honor the 2D scaling gate.** [`docs/lumi_2d_scaling.md`](docs/lumi_2d_scaling.md) says do not start 3D production until kernel vs halo says kernel-bound. Moving window and block skip stay 2D tools until that gate.
3. **3D is already in the engine** (`Nz>1`, hemispheres, `Full3D` MPI/HIP). Do not start with a \(1200\times 320\times 320\) brick (\(\sim 10^8\) cells). First science case: same \(G,V_p,W_0,\theta\), **one grain**, thin \(L_z\) (8–16 \(W_0\)), periodic \(y,z\). Gate: \(N_z=1\) with `n_dim=2` matches the 2D gold path; then a thin slab whose mid-plane resembles the 2D cells (not identical). Then two grains. Use the moving window before a full 3D strip.
4. **Memory.** Persist \(e^u,u\); do not store anisotropy fluxes; AMR stays deferred. Window + block skip first.
5. **Noise-on full-strip / 2× ensemble** is a separate 2D campaign, not a 3D prerequisite.

## Scaling scripts (2D campaign)

Runnable LUMI 2D campaign (steps 0–2) plus a step-3 kernel-vs-halo gate.
**Do not** start 3D production bricks from these scripts.

| Script | Role |
|--------|------|
| [`scripts/check_nz1_vs_2d.sh`](scripts/check_nz1_vs_2d.sh) | Step 0 CPU: OpenMP vs MPI `np=1` |
| [`scripts/check_hip_vs_cpu.sh`](scripts/check_hip_vs_cpu.sh) | Step 0 GPU: 1 GCD HIP vs CPU checksum (LUMI-G) |
| [`scripts/lumi_scale_cpu.sh`](scripts/lumi_scale_cpu.sh) | Step 1: LUMI-C OpenMP/MPI, I/O off |
| [`scripts/lumi_scale_gpu.sh`](scripts/lumi_scale_gpu.sh) | Step 2: LUMI-G, 1 rank/GCD, `MPICH_GPU_SUPPORT_ENABLED=1` |
| [`scripts/lumi_submit_2d_scale.sh`](scripts/lumi_submit_2d_scale.sh) | Submit the sweep (`--cpu-large` for `standard`) |
| [`scripts/analyze_2d_scaling.py`](scripts/analyze_2d_scaling.py) | Step 3: table + verdict |

Exact `sbatch` lines and the halo-vs-kernel criterion:
[`docs/lumi_2d_scaling.md`](docs/lumi_2d_scaling.md).
Plan contract: [`docs/hpc/scalability_analysis_plan.md`](../../docs/hpc/scalability_analysis_plan.md).
Packed-halo env (real name in code): `OPENPFC_HIP_FORCE_PACKED_HALO=1`.

## AMR

Classic octree / hanging-node AMR is **deferred**. Use the moving window and optional block skip first. Revisit AMR only if those cannot fit the target DS brick. Comments in `defaults.hpp` and the engines match this.

## HIP / LUMI-G risks

- Halos go through `pfc::hip::FullPaddedDeviceHalo` (device buffers). GPU-aware MPI with derived types can still storm on some stacks — try `OPENPFC_HIP_FORCE_PACKED_HALO=1` (packed faces; corners may be incomplete on that fallback).
- HIP window shift uses the device kernel on `np=1`. Multi-rank x-decomp copies to host and uses the same `shift_left_one` plane exchange as MPI CPU (correct, not fast).
- Block skip is honored on OpenMP/MPI; HIP kernels still visit every cell.
- Anisotropy face fluxes are `__device__` helpers (no device lambdas).
