<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Evidence-like content: classification ledger (#311)

Status at `master` `450ef179`, 2026-10-28. This page classifies files only.
It moves and deletes nothing. Each move needs a destination study and its
provenance checked first (#311 migration requirements). Every "study owner
(to decide)" entry is a decision for the maintainer.

## Classes

1. **Package regression fixture**: input or expected output that a package
   test reads.
2. **Implementation verification**: a check that supported behaviour is
   correct.
3. **Numerical oracle**: an independent reference the software is
   compared against.
4. **Reusable example**: input/output or recipe that a user can rerun.
5. **Campaign protocol**: frozen sweep or ladder definition for a study.
6. **Campaign output**: measured tables, run archives, job records.
7. **Interpretation**: scientific or performance conclusions drawn from (6).

Classes 1–4 may stay in OpenPFC. Classes 5–7 belong to the owning study.

"Test import" means a package test (`tests/`, `scripts/tests/`,
`apps/*/tests/`, CTest) reads the file. Consumers were found with
`git grep` on the file name.

## Already resolved

| Path | Resolution |
| --- | --- |
| `apps/inverse_homogenization/evidence/` | Removed by `e7f6cc52` ("docs(apps): drop inverse-homogenization campaign records"). Not on `master`. |
| `docs/hpc/heat3d_n3_cost.md` | Turned into a reusable recipe by #319 (`c4d44e8e`). Class 4, stays. |

## `docs/hpc/` campaign records

| Path | Class | Rationale | Consumers | Test import | Owner / destination |
| --- | --- | --- | --- | --- | --- |
| `docs/hpc/heffte-protocol-evidence/` (`all-runs.csv`, `campaign-runs.csv`, `scaling.csv`, `scaling.md`, `records.json.gz`, `compilation.txt.gz`, `provenance.json`, `README.md`) | 6 | Retained run archive and provenance of one HeFFTe protocol campaign | `heffte_protocol_tournament.md`, `heffte_topology_crossover.md`, `REUSE.toml`, `docs/public_pages.txt`, `docs/public_redirects.json` | no | study owner (to decide) |
| `docs/hpc/heffte_protocol_tournament.md` | 7 (with 5) | Tournament design and its measured winner | `lumi_gpu_scaling.md`, `docs/lumi_slurm/README.md` | no | study owner (to decide) |
| `docs/hpc/heffte_bandwidth_calibrate.md` | 7 | Calibration conclusions from measured runs | `docs/lumi_slurm/README.md` | no | study owner (to decide) |
| `docs/hpc/heffte_bandwidth_calibrate.csv` | 6 | Measured bandwidth table | `heffte_bandwidth_calibrate.md`, `REUSE.toml` | no | study owner (to decide) |
| `docs/hpc/heffte_pencils_ab_32n.csv` | 6 | Measured 32-node pencil A/B table | `heffte_bandwidth_calibrate.md`, `REUSE.toml` | no | study owner (to decide) |
| `docs/hpc/heffte_topology_preanalysis.md` | 5 | Pre-registered analysis plan of the topology campaign | `heffte_topology_crossover.md` | no | study owner (to decide) |
| `docs/hpc/heffte_topology_crossover.md` | 7 | Crossover conclusion from the campaign | `docs/lumi_slurm/README.md` | no | study owner (to decide) |
| `docs/hpc/heffte_topology_observability.md` | 7 | Campaign observability analysis | `heffte_topology_crossover.md` | no | study owner (to decide) |
| `docs/hpc/heffte_topology_campaign_scaling.csv` | 6 | Measured scaling table | `heffte_topology_crossover.md`, `REUSE.toml` | no | study owner (to decide) |
| `docs/hpc/heffte_topology_crossover_descriptors.csv` | 5 | Campaign case descriptors read by the launcher | `heffte_topology_crossover.md`, `REUSE.toml`, `docs/lumi_slurm/submit_heffte_topology_crossover.sh` | no | study owner (to decide); move with its launcher |
| `docs/hpc/fd_order_scaling.md` | 7 | FD-order scaling conclusions | `docs/lumi_slurm/README.md` | no | study owner (to decide) |
| `docs/hpc/fd_order_campaign_scaling.csv` | 6 | Measured scaling table | `fd_order_scaling.md`, `REUSE.toml` | no | study owner (to decide) |
| `docs/hpc/fd_order_geometry.csv` | 5 | Case geometry read by the launcher | `fd_order_scaling.md`, `REUSE.toml`, `docs/lumi_slurm/submit_fd_order_scaling.sh` | no | study owner (to decide); move with its launcher |
| `docs/hpc/fd_order_overlap_diag.csv` | 6 | Measured overlap diagnostics | `fd_order_scaling.md`, `REUSE.toml` | no | study owner (to decide) |
| `docs/hpc/lumi_gpu_scaling.md` | 7 (with 6) | Measured LUMI GPU scaling and conclusions | `CHANGELOG.md`, `heffte_protocol_tournament.md`, `lumi_gpu_heat3d_fd_weak.md`, `scalability_analysis_plan.md`, `docs/lumi_slurm/README.md`, `docs/lumi_slurm/submit_*_scaling.sh`, `tests/baselines/BASELINES.md` | no (`BASELINES.md` cites it as prose) | study owner (to decide). Retarget `tests/baselines/BASELINES.md` and `docs/lumi_slurm/README.md` when it moves; performance JSON pins stay. |
| `docs/hpc/lumi_gpu_heat3d_fd_weak.md` | 7 (with 6) | Weak-scaling measurements and conclusions | `docs/lumi_slurm/README.md` | no | study owner (to decide) |

Kept as developer documentation (not evidence): `docs/hpc/README.md`,
`operator_guide.md`, `gpu_path_decision.md`, `build_cpu_gpu.md`,
`mpi_io_layout_checklist.md`, `performance_profiling.md`,
`profiling_export_schema.md`, `INSTALL.LUMI.md`, `INSTALL.tohtori.md`,
`scalability_analysis_plan.md`, `heffte_selector_regret.md`, `vlasov_gpu.md`.
A later pass may still find campaign tables inside them.

## Launchers and scripts

| Path | Class | Rationale | Consumers | Test import | Owner / destination |
| --- | --- | --- | --- | --- | --- |
| `apps/alloy_dendrite_elastic/slurm/showcase_dendrite{2d,3d}.sbatch` | 5 | Showcase campaign launchers, added after the "legacy, do not grow" note | app README | no | study owner (to decide) |
| `apps/vlasov_maxwell/slurm/showcase_weibel.sbatch` | 5 | Showcase campaign launcher | app README | no | study owner (to decide) |
| `docs/lumi_slurm/*.sbatch`, `submit_*.sh` | 4 or 5 per file | Legacy launchers. Those that only build or rerun a documented recipe (for example `heat3d_n3_cost.sbatch`) are class 4. Those bound to a campaign's case table are class 5. | `docs/lumi_slurm/README.md` | no | Class 4 stays. Class 5 moves with its campaign CSV, study owner (to decide). |
| `apps/heat3d/scripts/fd_weak_ladder.py` | 5 | Weak-ladder driver, no test import | `lumi_gpu_heat3d_fd_weak.md`, `docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh` | no | study owner (to decide); a classified copy is needed before any deletion |
| `apps/higher_order_pfc/scripts/run_seed_ensemble.sh` | 5 | Frozen eight-seed ensemble with a scratch path | app README | no | study owner (to decide) |
| `apps/inverse_homogenization/scripts/render_certified_trajectory.py` | 2 | Renders and checks a certified trajectory; tested on synthetic input | app README, `scripts/tests/test_certified_trajectory.py` | yes | package (stays) |
| `apps/inverse_homogenization/scripts/audit_converged_resolution.py` | 2 | Converged-resolution admission rule; test builds a synthetic archive ("Campaign archives are not package fixtures") | `scripts/tests/test_converged_resolution_reducer.py` | yes | package (stays) |

## Application documentation

| Location | Class | Rationale | Owner / destination |
| --- | --- | --- | --- |
| `apps/incompressible_flow/README.md`, "Resolution ladder" (from line 149) and "Stationary ladder" (from line 309) | 5 and 6 (with 7) | Frozen ladders, measured tables, job results and `/scratch/...` result paths (lines 178, 329) | Study owner (to decide). Keep a short verified-capability summary in the README. Not edited here. |
| `apps/alloy_dendrite_elastic/README.md`, job records such as job 22045158 (line 963) | 6 | Job IDs and measured showcase results inside user docs | study owner (to decide) |
| `apps/higher_order_pfc/README.md`, frozen seed ensemble (line 307) | 5 and 6 | Ensemble protocol and outcome | study owner (to decide) |

Package tests and oracles in `apps/incompressible_flow/tests/` (forced and
decaying HIT, device parity) are classes 1 to 3 and stay.

## Counts

| Class | Items |
| --- | --- |
| 1 Package regression fixture | 0 (stand-alone; the app tests stay) |
| 2 Implementation verification | 2 |
| 3 Numerical oracle | 0 (stand-alone) |
| 4 Reusable example | 1 resolved (`heat3d_n3_cost.md`), plus recipe launchers |
| 5 Campaign protocol | 8 (2 case tables, 1 preanalysis, 3 app launchers, 2 drivers), plus campaign launchers |
| 6 Campaign output | 6 (1 archive directory, 5 tables), plus 3 README sections |
| 7 Interpretation | 7 pages |

## Decisions needed

1. A destination study for the HeFFTe protocol/bandwidth/topology campaigns,
   the FD-order campaign and the LUMI GPU scaling records.
2. A destination for the incompressible-flow ladders and the
   alloy-dendrite and higher-order-PFC showcase records.
3. For each `docs/lumi_slurm` launcher: class 4 (stays) or class 5 (moves
   with its campaign).
4. Whether `docs/hpc/heffte-protocol-evidence/` keeps its public redirect
   after it moves (`docs/public_redirects.json`, `docs/public_pages.txt`).
