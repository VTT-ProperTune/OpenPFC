<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Inverse homogenization (`apps/inverse_homogenization`)

**Issue [#161](https://github.com/VTT-ProperTune/OpenPFC/issues/161).** Explicit
catalog exception: this is the sixteenth application because it is
PDE-constrained inverse design on the existing FFT/phase-field stack, not
another PDE demo. Do not treat it as a licence to add a seventeenth app.

The central problem is **microstructure inverse design**: given a target
homogenized elasticity tensor \(C_{\mathrm{target}}\), find a periodic
microstructure whose effective tensor \(C_H\) matches it. This is not a
minimum-compliance structural topology-optimization demo.

The inverse map is **non-unique**. Many microstructures can realize nearly
the same \(C_H\).

## Status

Stages 1–4 of #161 are in tree:

| Stage | What | Where |
|-------|------|--------|
| 1 | Forward periodic \(C_H\) | `openpfc_apps/homogenization.hpp`, `openpfc_homogenize` |
| 2–3 | Allen–Cahn descent on the tensor-mismatch objective, volume penalty, perimeter | `phase_field_inverse.hpp`, `openpfc_inverse_homogenize` |
| 4 | Discrete mutual-energy \(\delta J/\delta h\) + finite-difference check | homogenization.hpp; Catch2 `apps-common-homogenization` |

Stages 5–8 (free-topology target campaign, spinodal constraint,
manufacturability, LUMI 3-D) are **not** implemented yet.

## Reuse

The elliptic solve is `EigenstrainMicroelasticity` in
[`apps/common/include/openpfc_apps/microelasticity.hpp`](../common/include/openpfc_apps/microelasticity.hpp).
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

`--shape` is `homogeneous`, `laminate-z`, or `sphere`. The printed \(C_H\) is
the **engineering Voigt** \(6\times 6\) (order \(11,22,33,23,13,12\),
\(\gamma=2\varepsilon\)). A homogeneous isotropic material therefore reports
\(C_{44}=\mu\), not \(2\mu\).

`HOMOGENIZATION_CHECKSUM` is \(\lVert C_H\rVert_F\); the smoke test greps it.

## Inverse driver (Allen–Cahn)

```bash
mpirun -n 1 ./apps/inverse_homogenization/openpfc_inverse_homogenize \
  --target=isotropic --E-target=0.9 --nu-target=0.25 \
  --volume=0.5 --nx=16 --ny=16 --nz=16 --steps=10 --init=noise
```

`--target` is `isotropic`, `auxetic` (negative Poisson via `--nu-target`), or
`orthotropic` (`--C11 --C22 --C12 --C66`). The loop is Takezawa-style
Allen–Cahn, not Cahn–Hilliard and not MMA. `INVERSE_CHECKSUM` is the last
\(J\).

## Tests

`ctest -R homogenization` (HeFFTe builds):

* `apps-common-homogenization` — homogeneous oracle, laminate/Postma,
  cubic symmetry, closed-form homogeneous sensitivity, finite-difference
  gradient check on random \(h\).
* `apps-common-homogenization-mpi` — same binary on two ranks.
* `inverse-homogenization-homogeneous-smoke` — the CLI entry point.

Build through `./scripts/build.sh` only. On LUMI, **configure** may run on
the login node (FetchContent needs the network); **compile, test, debug and
run** go to `standard` (CPU) or `standard-g` (HIP). Never compile on the
login node. CPU suite:

```bash
sbatch apps/inverse_homogenization/slurm/build_and_test_cpu.sbatch
```

## Workflow (target, not yet the inverse binary)

```
h(x) → C(x) → six periodic elasticity solves → C_H[h] → J → dJ/dh
     → optimizer / phase-field evolution → new h(x)
```

`PeriodicHomogenizer::compute` and `objective_sensitivity` are the first two
arrows after \(C(x)\). The rest waits on later PRs of #161.
