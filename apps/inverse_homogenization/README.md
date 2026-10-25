<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Inverse homogenization (`apps/inverse_homogenization`)

**Issue [#161](https://github.com/VTT-ProperTune/OpenPFC/issues/161).**
PDE-constrained inverse design on the existing FFT/phase-field stack,
not another forward PDE demo. Later applications are admitted by the
criteria in [`../README.md`](../README.md#what-belongs-here), not by a
fixed catalog size.

The central problem is **microstructure inverse design**: given a target
homogenized elasticity tensor $C_{\mathrm{target}}$, find a periodic
microstructure whose effective tensor $C_H$ matches it. This is not a
minimum-compliance structural topology-optimization demo.

The inverse map is **non-unique**. Many microstructures can realize nearly
the same $C_H$.

## Build

HeFFTe must be enabled; without it this application is not built. Build
and test only through [`scripts/build.sh`](../../scripts/build.sh). On
LUMI the script configures on the login node and, by default, submits
compile and `ctest` to a GPU partition. `--cpu` builds the host
executables. The HIP executables are built only when HIP spectral
support is on (`--with-rocm` on LUMI).

```bash
./scripts/build.sh
./scripts/build.sh --machine=lumi --with-rocm
./scripts/build.sh --machine=lumi --cpu --no-submit
```

The install step places the host programs in `bin/`:

* `openpfc_homogenize`
* `openpfc_inverse_homogenize`
* `openpfc_finite_strain_forward`
* `openpfc_finite_strain_inverse`

HIP spectral builds also install `openpfc_homogenize_hip` and
`openpfc_inverse_homogenize_hip`. The commands below use build-tree
paths. An installed binary takes the same flags.

## Reuse

The elliptic solve is `EigenstrainMicroelasticity` in
[`include/openpfc/solvers/microelasticity/microelasticity.hpp`](../../include/openpfc/solvers/microelasticity/microelasticity.hpp).
Homogenization is the same Green-operator problem with **zero eigenstrain**
and an imposed macroscopic strain (`applied_strain`). There is no second
elasticity implementation, no FEM, and no unstructured mesh.

## Forward driver

```bash
mpirun -n 1 ./apps/inverse_homogenization/openpfc_homogenize \
  --shape=homogeneous --nx=16 --ny=16 --nz=16 --volume=1
mpirun -n 2 ./apps/inverse_homogenization/openpfc_homogenize \
  --shape=laminate-z --E-solid=1 --E-void=0.25
```

`--shape` is `homogeneous`, `laminate-z`, `sphere`, `rotating-cubes`,
`rotating-squares`, `reentrant-3d`, or `yang-a3`. The printed $C_H$ is
the **engineering Voigt** $6\times 6$ (order $11,22,33,23,13,12$,
$\gamma=2\varepsilon$). A homogeneous isotropic material therefore
reports $C_{44}=\mu$, not $2\mu$. The driver also prints
$S=C_H^{-1}$, compliance Poisson ratios, SPD / $\lambda_{\min}$,
cubic spreads, and periodic percolation. `--dump-dir` writes gathered
`h.bin` / `h.xdmf`. `--load-bin` reads a Fortran-order float64 brick.
`--half` and `--angle` set the rotating-square or rotating-cube size.
`--thickness` and `--inset` set the re-entrant struts.

`HOMOGENIZATION_CHECKSUM` is $\lVert C_H\rVert_F$.

When the HIP forward driver is built:

```bash
mpirun -n 1 ./apps/inverse_homogenization/openpfc_homogenize_hip \
  --shape=homogeneous --nx=8 --ny=8 --nz=8 --volume=1
```

It accepts `--nx`, `--ny`, `--nz`, the phase moduli, `--volume`, and
`--shape` of `homogeneous`, `laminate-z`, or `rotating-squares`.

## Inverse driver (Allen–Cahn)

```bash
mpirun -n 1 ./apps/inverse_homogenization/openpfc_inverse_homogenize \
  --target=isotropic --E-target=0.9 --nu-target=0.25 \
  --volume=0.5 --nx=16 --ny=16 --nz=16 --steps=10 --init=noise
```

`--target` is `isotropic`, `auxetic` (negative Poisson via
`--nu-target`), `orthotropic` (`--C11 --C22 --C12 --C66`), or `file`
(`--C-target-file` with a 6×6 Voigt text matrix). `--init` includes
`uniform`, `noise`, `spinodal`, `rotating-squares`, `reentrant`, and
`yang-a3`. `--load-bin` reads a Fortran `h` brick. The loop is
Takezawa-style Allen–Cahn, not Cahn–Hilliard and not MMA.
`INVERSE_CHECKSUM` is the last $J$. Optional `--csv=PATH` writes the
per-step history. `--ch-steps` (with `--ch-kappa`, `--ch-dt`, and
`--ch-aniso-y`) substitutes a Cahn–Hilliard process map for the
free-topology step.

The final report prints the full $6\times 6$ $C_H$ of the
**initial**, **final physical**, and **$h>0.5$ thresholded** fields,
plus compliance Poisson ratios $\nu_{xy}=-S_{12}/S_{11}$ (and cyclic),
SPD / $\lambda_{\min}$, cubic spreads, elasticity residuals, and
periodic percolation on any rank count. `--dump-dir` writes gathered
Fortran `h_init.bin`, `h_final.bin`, `h_thresh.bin`, and optional
`h_%04d.bin` snapshots. The shortcut $C_{12}/(C_{11}+C_{12})$ is still
printed as `nu_shortcut`. Qualification uses the compliance tensor.

`--steps` / `--max-steps` is a **ceiling**, not success. SIMP and
`lambda-reg` interpolate over `--continuation-steps` (default 300) and
then freeze. Convergence is impossible during continuation. After
freeze, three metrics must hold together for `--conv-window` (20)
consecutive iterates, then a `--verify-convergence-steps` (100) hold
with the same frozen problem. A broken hold rejects the candidate
immediately. Morphology-change fraction is a diagnostic, not a stop.

Logged row $s$ compares consecutive **accepted** post-projection
states, not a post-update design RMS paired with a pre-update objective:

* design RMS $\lVert h_s-h_{s-1}\rVert_2/\sqrt{N}<10^{-4}$ is measured
  **before** the Allen–Cahn update (not the pre-projection `step_rms`);
* $\lvert J(h_s)-J(h_{s-1})\rvert/\max(1,\lvert J(h_{s-1})\rvert)<10^{-6}$;
* $\lVert C_H(h_s)-C_H(h_{s-1})\rVert_F /
  \max(\lVert C_H(h_{s-1})\rVert_F,\varepsilon)<10^{-4}$.

$J$ and $C_H$ are the homogenization of that same accepted $h_s$.
The in-place Allen–Cahn update then produces a trailing candidate
$h_{s+1}$. On `CONVERGED`, `MAX_STEPS`, or `ELASTICITY_FAILURE` the
driver restores $h_s$ before writing `h_final.bin` / `h_thresh.bin`
and before the unpenalized `FINAL_RECOMPUTE`. The declared material is
exactly the certified accepted field. The first iterate has no
predecessor and is never quiet. CSV volume is the accepted-state mean,
not the post-update candidate.

CPU and HIP write the same 26-column iterate schema
(`kInverseCsvHeader`). The unpenalized final $C_H$ is a
`# FINAL_RECOMPUTE` comment, not a data row. A `# CERTIFIED_STEP`
comment records the iterate index. Comment lines are not iterate
records. The final accepted field is always snapshotted, even between
`--dump-every` points. `MAX_STEPS` is not labelled `CONVERGED`.

HIP SIMP uses the same `simp_density` / `simp_chain` helpers as CPU:
elasticity on $h^p$, sensitivity chain rule $p h^{p-1}$. $p=1$ is
identity and does not call $\mathrm{pow}(h,0)$.

### Checkpoint / restart

`--checkpoint-dir` publishes a complete generation (`gen_<next_step>/`
with `h.bin`, `h_prev.bin`, `state.txt`, and HIP `dump_steps.txt`) and
then atomically retargets `CURRENT`. A walltime kill during the write
leaves the previous published generation loadable. `--restart=DIR`
reads `CURRENT` or an explicit generation directory. Schema 3
fingerprints the target $C$, phase moduli, SIMP / regularization
endpoints, step/projection, window, and tolerances, and records whether
the bundle is a running or terminal state. `--max-steps` is a run
budget and may change; any other mismatch is rejected. `CONVERGED`,
`MAX_STEPS`, and `ELASTICITY_FAILURE` bundles are terminal snapshots
and are not continuation restarts. Schema 1 and schema 2 files do not
load.

A restart requires `--max-steps` strictly greater than the saved
`next_step`. An exhausted requested budget is rejected before field
restoration, so an unevaluated next field cannot be exported as the
previous accepted material. Checkpoint metadata writes are checked
through close before publishing `CURRENT`. A staging or publication
failure exits nonzero on every rank and leaves the preceding published
generation available. GPU restart rejects a missing, malformed, or
inconsistent snapshot-index ledger instead of resetting its frame
identities. Field sizes are checked before publication. This is not a
guarantee against storage-hardware corruption or power loss, and it
does not claim different-rank bitwise equivalence. Keep each
allocation's CSV under a distinct name. The checkpoint's next-step
index defines the retained prefix of an interrupted allocation. The
`CURRENT` generation is the restart authority, not the newest CSV row.

### HIP inverse driver

When the HIP inverse driver is built:

```bash
mpirun -n 1 ./apps/inverse_homogenization/openpfc_inverse_homogenize_hip \
  --nx=8 --ny=8 --nz=1 --steps=1 \
  --init=rotating-squares --lambda-reg=0 --dt=0.05
```

The accepted flags follow the CPU driver (`--target`, `--init`,
`--load-bin`, `--csv`, `--dump-dir`, `--checkpoint-dir`, `--restart`,
continuation, and the convergence tolerances). With `--dump-dir` it
also writes `h_final_material.json` and `h_thresh_material.json` beside
the accepted `h_final.bin` and its strict `h>0.5` thresholded field.
Each record identifies the accepted step, termination reason, grid, and
spacing. `MAX_STEPS` remains nonconvergence, even when the endpoint
elasticity solves succeed.

The records preserve all 36 raw unpenalized stiffness entries.
Diagnostics use a separate symmetric tensor, engineering-Voigt order
`[xx, yy, zz, yz, xz, xy]`, and engineering shear strain.
Compliance-based `nu_xy` means transverse $y$ response under uniaxial
$x$ stress; all six ordered axis pairs are reported. The compliance
and Poisson fields are null unless all six solves converge, the tensor
is finite, positive definite, and invertible, and compliance
normalization is defined.

```bash
python3 apps/inverse_homogenization/scripts/render_certified_trajectory.py \
  /path/to/fields/run_manifest.json results/movie
```

Rendering requires NumPy, PyVista/VTK with offscreen OpenGL, and ffmpeg.
`--validate-only` checks input identity without graphics dependencies.
The output `frames.json` records field, frame, and movie checksums.
The script draws the manifest snapshots; it does not synthesize
intermediate designs.

## Finite-strain drivers

`openpfc_finite_strain_forward` prints `FINITE_STRAIN_CHECKSUM` from
the shipped 2-D plane-strain ladder:

1. compressible neo-Hookean and St.\ Venant--Kirchhoff
   $P=\partial W/\partial F$;
2. homogeneous uniaxial $F_{11}$ with transverse relaxation $P_{22}=0$;
3. two-material $y$-laminate;
4. tangent $\nu_t=-\mathrm{d}\ln F_{22}/\mathrm{d}\ln F_{11}$ by finite
   difference of the same shipped $F_{22}(F_{11})$;
5. Newton residual and $J>0$ stability.

`openpfc_finite_strain_inverse` prints `FINITE_STRAIN_INVERSE_CHECKSUM`
from the shipped $16^2$ neo-Hookean rotating-square comparison
(single-material plus void, and a hinge-painted stiff/compliant/void
cell) along an $F_{11}$ path.

```bash
./apps/inverse_homogenization/openpfc_finite_strain_forward
./apps/inverse_homogenization/openpfc_finite_strain_inverse
```

Optional `--evidence PATH` writes the same numbers as JSON. Neither
binary reads an input archive.

## Workflow

```text
h(x) → C(x) → six periodic elasticity solves → C_H[h] → J → dJ/dh
     → Allen–Cahn step or CH process map → new h(x)
```

`PeriodicHomogenizer::compute` and `objective_sensitivity` are the
first two arrows after $C(x)$. `openpfc_inverse_homogenize` runs the
rest as Takezawa-style Allen–Cahn. The HIP twins
(`openpfc_homogenize_hip`, `openpfc_inverse_homogenize_hip`) use a
device Green operator (FFT + Eyre–Milton) on the same loop.

## Tests

From the build directory, `ctest -R inverse-homogenization` runs this
application's checks. Host names:

* `inverse-homogenization-homogeneous-smoke` — the forward CLI prints
  `HOMOGENIZATION_CHECKSUM` for a homogeneous cell.
* `inverse-homogenization-rotating-cubes-smoke`,
  `inverse-homogenization-rotating-squares-extrusion-smoke`,
  `inverse-homogenization-reentrant-3d-smoke`, and
  `inverse-homogenization-yang-a3-smoke` — the same checksum line for
  those shapes.
* `inverse-homogenization-ac-smoke` — the inverse CLI prints
  `INVERSE_CHECKSUM`.
* `inverse-homogenization-finite-strain-smoke` and
  `inverse-homogenization-finite-strain-inverse-smoke` — the finite-strain
  CLIs print `FINITE_STRAIN_CHECKSUM` and
  `FINITE_STRAIN_INVERSE_CHECKSUM`.
* `inverse-homogenization-forward` — engineering Voigt algebra and the
  binary-laminate closed form; a homogeneous cell recovers the
  interpolated stiffness; a smooth $z$-laminate matches the Backus
  average; a sharp $z$-laminate matches the two-phase Postma
  stiffness; a centred sphere is cubic; homogeneous sensitivity matches
  the closed-form derivative; adjoint sensitivity matches central finite
  differences.
* `inverse-homogenization-forward-mpi` — that same binary on two ranks,
  when multi-rank MPI suites are enabled.
* `inverse-homogenization-ac-tests` — spectral Laplacian of a cosine;
  volume drive and volume projection; a short descent of $J$; the
  double well on a uniform grey field; rotating-square, re-entrant,
  rotating-cube, and Yang A3 seeds; spinodal mean conservation;
  percolation and opening loss; Voigt-file and Fortran-brick I/O;
  restoration of the certified design; endpoint material records.
* `inverse-homogenization-finite-strain-tests` — Lamé conversion,
  homogeneous neo-Hookean $P_{22}=0$, the small-strain tangent
  Poisson, laminate stretches, a stable St.\ Venant--Kirchhoff uniaxial
  path, and convergence of the declared forward rungs.
* `inverse-homogenization-finite-strain-grid-tests` — the periodic
  displacement gauge, odd and even homogeneous solves, recovery of the
  homogeneous $F_{22}$ on a uniform grid, and the rotating-square
  periodic homogenizer.
* `inverse-homogenization-convergence-tests` — continuation freeze, the
  three-metric window and verification hold, CSV header width, SIMP
  $p=1$, and checkpoint schema, restart budget, staging, and
  publication failures.

HIP spectral builds add `inverse-homogenization-hip-homogeneous-smoke`
and `inverse-homogenization-hip-ac-smoke`, which require the same
checksum lines from the HIP executables.
