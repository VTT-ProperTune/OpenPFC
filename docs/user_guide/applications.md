<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Applications

The programs under `apps/` are full OpenPFC applications. They are different from the short executables under `examples/`: examples teach one API pattern at a time, while applications are meant to be run as model-specific binaries with realistic inputs. They are built when `OpenPFC_BUILD_APPS=ON`, which is the default, and they usually install under `<prefix>/bin` when you run `cmake --install`.

For realistic runs, assume MPI is involved. Use the same compiler, MPI and HeFFTe stack that you used to build OpenPFC; the install details are in [`INSTALL.md`](../../INSTALL.md). If you are still learning the library, run an example first through [`../quickstart.md`](../quickstart.md), then come back here.

## Which application should I run?

Use `./scripts/openpfc apps` to list the JSON-session apps, their presets,
and CPU/CUDA/HIP target availability. `openpfc init CASE --app=NAME --preset=PRESET`
creates a self-contained input directory; see the [CLI guide](cli.md).
Cahn–Hilliard offers both single-mode verification and seeded coarsening with
mass, composition bounds, and total-energy CSV output; see its
[diagnostics guide](../../apps/cahn_hilliard/README.md#coarsening-preset-and-diagnostics).

Start with tungsten if you want the production-style PFC path. It reads JSON or TOML, uses the `App` pipeline, writes configured fields, and has CPU, CUDA and HIP variants when the build enables them. Start with Allen–Cahn if you want a small visual sanity check with optional PNG output and fewer moving pieces. Use Heat3D when your question is about finite-difference orders, the spectral heat-equation path, timings or scaling comparisons. Use **cahn_hilliard** for conserved fourth-order spinodal decomposition (Fe–Cr-like regular solution) on the same JSON spectral-ETD path as tungsten. Use **thin_film** for lubrication dewetting / coating (\(k^4\) capillary plus disjoining pressure, including an \(A=0\) leveling case). Use **surface_diffusion** for Mullins thermal smoothing of nanoscale roughness (exact \(k^4\) decay). Use **kawahara** for odd-order capillary–gravity dispersive waves (\(ik^3\) vs \(ik^5\), not a smoother). Use **ehd_film** for a sixth-order elastohydrodynamic gap under a bending plate (\(\lambda\sim-k^6\)). Use **gradient_elasticity** for size-dependent isotropic elasticity (Helmholtz–Navier, periodic eigenstrain, one-shot spectral \(2\times 2\)). Use **higher_order_pfc** for a deliberately very high-order PFC kernel (two-mode \(k^8\) free energy, \(k^{10}\) conserved dynamics) where the second correlation peak opens a band the classical \(k^4\) kernel cannot. Use **wave2d** for a minimal **coupled first-order** wave-equation demo (displacement + velocity) with mixed periodic / physical y-boundaries. Use **kobayashi** for a **coupled phase-field + temperature** dendritic-growth-style demo (periodic torus, manual FD, PNG of \(\phi\)). Use **alloy_dendrite_elastic** when the question is *quantitative* solidification with optional eigenstrain elasticity: it is the Echebarria–Karma dilute-alloy phase field with the anti-trapping current, coupled to solute, to temperature, and to a spectral microelasticity solve on the same decomposition. Use **vlasov_maxwell** for 1D2V electromagnetic Vlasov–Maxwell kinetics (Landau damping, two-stream, Weibel). AluminumNew is mostly useful as a compact example of an `App<Model>` program wired through JSON.

If you want declarative configuration, read [`app_pipeline.md`](app_pipeline.md) before writing your own input files. If your immediate question is “what file did this run write?”, read [`io_results.md`](io_results.md).

## Tungsten PFC

Tungsten is the main 3D PFC application. The CPU binary is `tungsten`; GPU-enabled builds may also provide `tungsten_cuda` or `tungsten_hip`, and HIP builds may include `verify_gpu_aware_mpi` as a device-buffer smoke test for LUMI-style workflows. The application overview, code layout and input directories are documented in [`apps/tungsten/README.md`](../../apps/tungsten/README.md).

From your build directory, a first CPU run looks like this:

```bash
mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

The JSON inputs live under [`apps/tungsten/inputs_json/`](../../apps/tungsten/inputs_json/README.md), with TOML equivalents under `inputs_toml/`. Other sample JSON files include `tungsten_fixed_bc.json`, `tungsten_moving_bc.json` and `tungsten_performance.json`. For GPU-aware MPI and Slurm examples, use [`../hpc/INSTALL.LUMI.md`](../hpc/INSTALL.LUMI.md) and [`../lumi_slurm/README.md`](../lumi_slurm/README.md).

## AluminumNew

`aluminumNew` is a sample 3D application using OpenPFC, nlohmann_json and HeFFTe. It is useful when you want to see an `App<Model>` target without the full tungsten complexity. Its README is intentionally small; the source and CMake target are the reference. See [`apps/aluminumNew/README.md`](../../apps/aluminumNew/README.md).

## Cahn–Hilliard (Fe–Cr spinodal)

`cahn_hilliard` is a 0.2 spectral-ETD application for conserved Cahn–Hilliard
dynamics on a periodic grid. The field is Cr mole fraction `c`. The default
regular-solution free energy at 475 °C puts Fe–32Cr inside the chemical
spinodal; \(\nabla^4\) is a \(k^4\) multiplier. JSON/TOML CLI matches tungsten.
HIP builds add `cahn_hilliard_hip` with the same input. See
[`apps/cahn_hilliard/README.md`](../../apps/cahn_hilliard/README.md).

```bash
mpirun -n 1 ./apps/cahn_hilliard/cahn_hilliard \
  ../apps/cahn_hilliard/inputs_json/fe_cr_spinodal.json
```

VTK of `c` is written under `results/cahn_hilliard/` (create that directory, or
let the writer create it). This is the materials-science Cahn–Hilliard app;
`examples/12_cahn_hilliard` remains the short teaching example.

## Thin film (dewetting / coating)

`thin_film` is a 0.2 spectral-ETD lubrication model for a periodic liquid
film. The field is thickness `h`. Capillary \(\nabla^4\) damps short waves;
van der Waals plus repulsion \(\Pi(h)\) can make a finite unstable band
(dewetting) or, with `A = 0`, level roughness. HIP builds add `thin_film_hip`.
See [`apps/thin_film/README.md`](../../apps/thin_film/README.md).

```bash
mpirun -n 1 ./apps/thin_film/thin_film \
  ../apps/thin_film/inputs_json/dewetting.json
mpirun -n 1 ./apps/thin_film/thin_film \
  ../apps/thin_film/inputs_json/leveling.json
```

## Surface diffusion (Mullins)

`surface_diffusion` is a 0.2 spectral-ETD Mullins model: \(\partial_t h=-B\nabla^4 h\).
Every Fourier mode decays as \(\exp(-B|k|^4 t)\), so short-wavelength roughness
vanishes first. HIP builds add `surface_diffusion_hip`. See
[`apps/surface_diffusion/README.md`](../../apps/surface_diffusion/README.md).

```bash
mpirun -n 1 ./apps/surface_diffusion/surface_diffusion \
  ../apps/surface_diffusion/inputs_json/smoothing.json
```

## Kawahara (capillary–gravity waves)

`kawahara` is a 0.2 spectral-ETD Kawahara model:
\(\partial_t u+\alpha u\partial_x u+\beta\partial_x^3 u+\gamma\partial_x^5 u=0\).
The high-order term is **dispersive**, not dissipative: \(L(k)=-i\omega(k)\)
with \(\omega=\beta k^3+\gamma k^5\). HIP builds add `kawahara_hip`. See
[`apps/kawahara/README.md`](../../apps/kawahara/README.md).

```bash
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/pulse.json
```

## EHD film (flexible plate)

`ehd_film` is a 0.2 spectral-ETD lubrication model under a Kirchhoff plate:
\(p=B\nabla^4 h\), \(\partial_t h=\nabla\cdot[M_0\nabla p]\), so
\(\lambda=-M_0 B k^6\). HIP builds add `ehd_film_hip`. See
[`apps/ehd_film/README.md`](../../apps/ehd_film/README.md).

```bash
mpirun -n 1 ./apps/ehd_film/ehd_film \
  ../apps/ehd_film/inputs_json/relaxation.json
```

## Gradient elasticity (Helmholtz–Navier)

`gradient_elasticity` is a 0.2 one-shot spectral solve, not ETD:
\((1-\ell^2\nabla^2)L_{\mathrm{navier}}\mathbf{u}=\mathbf{f}\) on a 2-D
periodic dilatational eigenstrain. Finite \(\ell\) regularizes high-\(k\)
content; \(\ell\to 0\) recovers classical elasticity. HIP builds add
`gradient_elasticity_hip`. See
[`apps/gradient_elasticity/README.md`](../../apps/gradient_elasticity/README.md).

```bash
mpirun -n 1 ./apps/gradient_elasticity/gradient_elasticity \
  ../apps/gradient_elasticity/inputs_json/inclusion.json
```

## Heat3D

`heat3d` solves the 3D heat equation either with finite differences or with a spectral FFT step. The finite-difference path supports even orders from 2 to 20, and the app can use OpenMP over interior \((i_y,i_z)\) lines when the build enables it (the Laplacian along \(i_x\) stays serial in `finite_difference.hpp`). On **Linux**, a **single MPI rank** resets CPU affinity after `MPI_Init` (via `pfc::runtime::reset_cpu_affinity_if_single_mpi_rank` in [`runtime/common/cpu_affinity.hpp`](../../include/openpfc/runtime/common/cpu_affinity.hpp)) so OpenMP is not stuck on one core under default `mpirun` pinning (set `OPENPFC_NO_RESET_AFFINITY` to opt out). For several ranks per node, tune `mpirun` binding and `OMP_NUM_THREADS` as in [`apps/heat3d/README.md`](../../apps/heat3d/README.md).

## Allen–Cahn

Allen–Cahn is a CLI-driven 2D demo. It does not use the JSON `App` frontend, which makes it a good quick visual check. The CPU binary is `allen_cahn`; CUDA or HIP builds may provide `allen_cahn_cuda` or `allen_cahn_hip`. See [`apps/allen_cahn/README.md`](../../apps/allen_cahn/README.md) for the current arguments and example `mpirun` commands.

MPI: Use `mpirun` or Slurm `srun` with Open MPI, the same stack as at configure time — on Slurm, Open MPI **5.0.10** (PMI/PMIx) is the documented baseline so **`srun`** works (see [`INSTALL.md`](../../INSTALL.md) §1). A mismatched launcher or `libmpi` causes confusing runtime failures.

Arguments (CPU binary): `nx ny n_steps dt M epsilon [driving_force] [png_final]` or, for an initial and final snapshot, `[png_initial] [png_final]` (two paths). The optional `driving_force` is detected when the next argument is numeric; otherwise that argument is treated as a PNG path for backward compatibility. Optional PNG paths trigger a gather on rank 0 and grayscale export via `pfc::io` (see `include/openpfc/frontend/io/png_writer.hpp`).

For visible motion on the grid, use moderate ε and large M; shrinking ε alone makes interfaces sharp but slow. A positive `driving_force` favours the `φ≈+1` seed over the `φ≈-1` matrix. The app reports step-loop timing and can gather PNG output on rank zero through the frontend PNG writer.

## wave2d

`wave2d` integrates the 2D acoustic wave equation \(u_{tt} = c^2 \Delta u\) as \(\partial_t u = v\), \(\partial_t v = c^2 \Delta u\) with explicit Euler in time. **x** is periodic (MPI halos); **y** supports homogeneous **Dirichlet** or **Neumann** physical boundaries via ghost correction after the periodic exchange. CPU binaries: `wave2d_fd_manual` (fixed second-order stencil on `PaddedBrick`) and `wave2d_fd` (even orders 2–20). Optional **VTK** output (`--vtk` / `--vtk-every`) uses `pfc::VTKWriter` for ParaView time series on CPU and GPU binaries alike. CUDA/HIP builds may add `wave2d_cuda` / `wave2d_hip` (`wave2d_hip` uses `SparseExchange<HIPSpace>` on device Fields). See [`apps/wave2d/README.md`](../../apps/wave2d/README.md) for CLI, CFL guidance, and tests.

## alloy_dendrite_elastic

`alloy_dendrite_elastic` is the **thermo-solutal-elastic** solidification capstone (issue #85): the quantitative
dilute-alloy phase field of Echebarria, Folch, Karma and Plapp with Karma's anti-trapping current, coupled to a solute
field, to a temperature field with latent heat, **and** to quasi-static eigenstrain microelasticity whose energy feeds
back into the phase-field driving force. High-order central FD (`pfc::gradient::FDGradient`, orders 2–14) on a
padded `pfc::comm::HaloExchange` stack, explicit four-stage step, MPI-decomposed, 2-D (`nz = 1`) and 3-D from one
templated stepper. The elastic solve is spectral on the *same* decomposition; `--elastic=1` is the calibrated Al–Cu
coupling, `--lambda-el=0` runs the solve and discards the feedback.

Two binaries. **`alloy_dendrite_planar`** runs the Stage-1 verification — an isothermal planar front in a periodic
two-front box — and reports the steady velocity against the thin-interface prediction, the kinetic coefficient, the
solute boundary layer against \(D_l/V\), the **effective partition coefficient** against the input \(k\), and the
drift of the two conservation invariants. **`alloy_dendrite_growth`** runs a deterministic dendrite (2-D, or 3-D with
`--nz`) and writes tip position, tip velocity and tip radius to CSV. HIP builds may add a parity driver for the FD step.

What makes it worth running rather than reading: at \(dx = 0.6\,W_0\) the measured \(k_\text{eff}\) is within 0.2 %
of \(k\) and flat in velocity, while switching the anti-trapping current off moves it by 17 % and makes it climb with
velocity. Total solute and the latent-heat balance are exact discrete identities and hold to round-off. See
[`apps/alloy_dendrite_elastic/README.md`](../../apps/alloy_dendrite_elastic/README.md) for the resolution study, the
measurement definitions, and the caveats.

## vlasov_maxwell

`vlasov_run` is the **kinetic capstone** (issue #84): a 1D2V electromagnetic Vlasov–Maxwell distribution
\(f_s(x, v_x, v_y, t)\) on one OpenPFC 3-D `Domain`, coupled to \(E_x, E_y, B_z\). One stepper; `--case` selects the
validation rung (`wave`, `landau`, `twostream`, `gyro`, `weibel`, `filament`). The electrostatic reduction is a runtime
flag, not a second code path. Linear rates are checked against a dispersion relation solved at run time
(`openpfc_apps/plasma_dispersion.hpp`). HIP builds add `--device=hip` on the science driver plus `vlasov_hip_parity`
and `vlasov_hip_cost`. See [`apps/vlasov_maxwell/README.md`](../../apps/vlasov_maxwell/README.md) and
[`../hpc/vlasov_gpu.md`](../hpc/vlasov_gpu.md).

```bash
mpirun -n 1 ./apps/vlasov_maxwell/vlasov_run --case=landau --summary=results/landau.csv
```

## kobayashi

`kobayashi_fd_manual` evolves the Kobayashi phase-field / temperature pair on a periodic **x–y** slab (`nz = 1`) with explicit Euler and the same finite-difference splitting as the historical Julia `kobayashi_v1` script (anisotropic \(\epsilon(\theta)\), Biner-style cross-flux terms, latent heat coupling). It writes grayscale PNG snapshots of \(\phi\). **`kobayashi_fd_openmp`** is the **single-node, OpenMP-only** variant (periodic torus via index wrapping — no MPI halos); verification lines match the **`nproc=1`** MPI reference for the same grid and step count. **`kobayashi_fd_cuda`** is the **MPI + CUDA** variant (host-staged halos, one GPU per MPI rank via shared-memory local rank). **`kobayashi_fd_hip`** is the **MPI + HIP** variant (`HaloExchange<HIPSpace>` on device Fields). See [`apps/kobayashi/README.md`](../../apps/kobayashi/README.md).

## Building your own application

If none of these binaries matches your problem, the next step is not to copy an application wholesale. First read [`app_pipeline.md`](app_pipeline.md) so you understand how JSON and TOML become a `Simulator`, then work through [`../tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md). The extension overview is [`../extending_openpfc/README.md`](../extending_openpfc/README.md).
