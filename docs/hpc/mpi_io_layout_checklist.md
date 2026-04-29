<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Checklist: MPI layout, paths, and binary I/O

Use this before debugging “missing file” or **hang** issues on clusters.

## MPI and launcher

- [ ] **Same MPI** at build and run: `mpirun` / `srun` comes from the same prefix as `mpicc` used for OpenPFC and HeFFTe ([`INSTALL.md`](../../INSTALL.md), [`troubleshooting.md`](../troubleshooting.md)).  
- [ ] **Rank count** matches what you expect for decomposition tests (`mpirun -n` vs `#SBATCH` tasks).

## Working directory and config paths

- [ ] Job script **`cd`**s to the directory where relative paths in JSON are valid.  
- [ ] **`argv[1]`** config path is correct from that directory (or use an absolute path).  
- [ ] **`fields[].data`** directories exist or are creatable; rank 0 creates parents in the default wiring ([`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).

## Binary writers (`BinaryWriter`)

- [ ] **`saveat` > 0** and `fields` present, or you intentionally rely on custom writers only ([`io_results.md`](../user_guide/io_results.md)).  
- [ ] Every rank in the writer’s **communicator** participates in each collective `write()` — skipping ranks causes **deadlock** ([`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).  
- [ ] Filename template **`%`** placeholders match the simulator increment you expect.

## GPU / device MPI (if applicable)

- [ ] GPU-aware MPI env vars set per site (e.g. LUMI: [`INSTALL.LUMI.md`](INSTALL.LUMI.md)).  
- [ ] `plan_options.backend` matches how the binary was built ([`build_cpu_gpu.md`](build_cpu_gpu.md), [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md)).

## See also

- [`tutorials/hpc_slurm_day_one.md`](../tutorials/hpc_slurm_day_one.md) — minimal Slurm script  
- [`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md) — JSON keys  
- [`learning_paths.md`](../learning_paths.md) — run track  
