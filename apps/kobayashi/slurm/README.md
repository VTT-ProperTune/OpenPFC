<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Kobayashi Slurm scaling (gen05_epyc)

## Script

[`kobayashi_scaling_gen05_epyc.sbatch`](kobayashi_scaling_gen05_epyc.sbatch) runs **`kobayashi_fd_manual`** with **512×512 cells**, **5000 steps**, **Δt = 1e-4**, **Δx = 0.03** (override with env vars below) for MPI rank counts **1, 2, 4, 8, 16, 32, 64, 128, 192** on partition **`gen05_epyc`**.

PNG output is disabled (`OPENPFC_KOBAYASHI_SKIP_PNG=1`) so timing reflects the integration loop. Progress printing is off (`OPENPFC_KOBAYASHI_QUIET=1`).

### OpenMP thread scaling (`kobayashi_fd_openmp`)

[`kobayashi_openmp_scaling_gen05_epyc.sbatch`](kobayashi_openmp_scaling_gen05_epyc.sbatch) runs **`kobayashi_fd_openmp`** as **one Slurm task** on **`gen05_epyc`**, sweeping **`OMP_NUM_THREADS`** (default **1 … 192** in steps matching the MPI script). It requests **`--cpus-per-task=192`** so high thread counts have cores to bind to; override the list with **`KOBAYASHI_OPENMP_THREADS`**.

The executable **parallel-first-touches** auxiliary buffers before the timed integration loop so heap pages fault under the OpenMP team rather than during timing (see `kobayashi_fd_openmp_engine.cpp`).

**Slurm / OpenMP launcher:** By default the scaling script runs **`kobayashi_fd_openmp` directly** (no nested **`srun`**). Optional **`KOBAYASHI_OPENMP_USE_SRUN=1`** wraps each sweep with **`srun --ntasks=1 --cpus-per-task=<threads> --cpu-bind=cores`** so the CPU mask matches **`OMP_NUM_THREADS`** for that step (using **`--cpus-per-task=$SLURM_CPUS_PER_TASK`** for every sweep while varying threads has caused **severe regressions** with **`OMP_PROC_BIND=spread`** on **gen05_epyc**). Do **not** default **`OMP_WAIT_POLICY=passive`** for this kernel: there are several **`parallel for`** regions per time step, so passive barriers inflate wall time at high thread counts. Set **`OMP_WAIT_POLICY` yourself** only when experimenting.

Optional **`KOBAYASHI_NUMA_INTERLEAVE=1`** prepends **`numactl --interleave=all`**. The script still exports **`OMP_DYNAMIC=false`** by default (override as needed).

```bash
export OPENPFC_BUILD_DIR=/path/to/build   # must contain kobayashi_fd_openmp
sbatch apps/kobayashi/slurm/kobayashi_openmp_scaling_gen05_epyc.sbatch
python3 apps/kobayashi/slurm/summarize_openmp_scaling.py kobayashi_openmp_scaling_<jobid>/
python3 apps/kobayashi/slurm/plot_strong_scaling.py kobayashi_openmp_scaling_<jobid>/
```

The generated **`summary.tsv`** uses the header **`nthreads`** so **`plot_strong_scaling.py`** labels the horizontal axis as **OpenMP threads**.

### Open MPI / Slurm PMI

Scaling uses **`srun`**. Your Open MPI must be built **with Slurm PMI/PMIx** — on **gen05_epyc** load **`openmpi/5.0.10`** (default in the sbatch). Binaries linked against an **older Open MPI without Slurm PMI** fail at `MPI_Init` with *“OMPI was not built with SLURM's PMI support”*.

**Rebuild OpenPFC** (and ensure **HeFFTe** was built with the **same** MPI) using:

```bash
export OPENPFC_REPO=$PWD   # repository root
sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_gen05_epyc.sbatch
```

Then point **`OPENPFC_BUILD_DIR`** at the build directory printed by that job (default **`builds/kobayashi-ompi510-slurm`**) when submitting the scaling script.

Override modules with **`GCC_MODULE`** / **`OPENMPI_MODULE`** if your site names differ.

**GCC note:** On some sites, **`module load openmpi/5.0.10`** switches **`gcc/11.2.0` → a newer GCC** (e.g. **15.2.0**). That is expected for that package; what matters for **`srun`** is linking **`libmpi`** from the **5.0.10** prefix. The rebuild script **removes the build directory by default** so CMake does not keep **`MPI_CXX_COMPILER`** pinned to a **stale cached MPI** from an older configure (set **`KOBAYASHI_REBUILD_KEEP_BUILD_DIR=1`** only if you know what you are doing).

The **`tohtori-gcc11-openmpi.cmake`** toolchain defaults to **`/share/apps/OpenMPI/5.0.10`** when **`OPENMPI_ROOT`** is unset. The rebuild sbatch **`module purge`s**, resolves **`mpicc`** with **`readlink -f`**, exports **`OPENMPI_ROOT`**, and passes **`-DMPI_C_COMPILER` / `-DMPI_CXX_COMPILER`** explicitly so the linked **`libmpi`** matches **`openmpi/5.0.10`**.

### Libfabric OFI (`fi_domain` / `mtl_ofi_component`)

Those messages come from Open MPI’s **OFI MTL** talking to **Libfabric**: on some nodes no usable **`fi_domain`** exists (misconfigured providers, missing hardware, or image mismatch), so initialization fails or spams stderr. That is **not** your application bug; it is the MPI transport stack.

The scaling script sets **`OMPI_MCA_mtl=^ofi`** by default so Open MPI avoids that layer and uses the usual **`ob1` + BTL path** (fine for **single-node** `srun`). To keep OFI (e.g. site requirement), submit with **`KOBAYASHI_OMPI_KEEP_OFI=1`**.

### CUDA module

Both **`kobayashi_rebuild_openpfc_gen05_epyc.sbatch`** and **`kobayashi_scaling_gen05_epyc.sbatch`** run **`module load cuda`** (override name with **`CUDA_MODULE`**, default **`cuda`**). If that module is missing, the job prints a warning and continues CPU-only so CPU batches still run.

On **CPU-only** nodes, the NVIDIA **driver** (`libcuda.so.1`) is often absent even after **`module load cuda`**. Open MPI may still try to open CUDA MCA plugins and print **`mca_accelerator_cuda` / `libcuda.so.1`** messages. The scaling script prepends **`$CUDA_HOME/lib64/stubs`** (when that path exists) to **`LD_LIBRARY_PATH`** so those plugins load harmlessly; real GPU runs still need a driver and appropriate **`#SBATCH`** GPU flags.

### Singleton MPI (`KOBAYASHI_MPI_COMM_WORLD_SIZE=1` with `srun -n > 1`)

If every rank prints **`nproc=1`**, Slurm did not expose a shared PMI namespace to Open MPI. The scaling script sets **`SLURM_MPI_TYPE=pmix`** and **`srun --mpi=pmix`** by default (your site may only ship **`pmix_v2`** — generic **`pmix`** selects it). If **`srun`** rejects the type, run **`srun --mpi=list`** and set **`KOBAYASHI_SRUN_MPI`** accordingly (e.g. **`pmix_v2`** or **`pmi2`**). As a fallback, **`export KOBAYASHI_MPI_LAUNCHER=mpirun`** uses **`mpirun -np …`** inside the allocation.

### `DependencyNeverSatisfied` on the scaling job

Slurm sets this when an **`afterok:`** prerequisite **did not finish successfully** (failed build, cancelled job, timeout). It is **not** a bug in the scaling script logic — check the **rebuild** log (`kobayashi_rebuild_openpfc_<jobid>.out` / `.err`). Example: resolving **`mpicc`** with **`readlink -f`** followed a symlink into **`opal_wrapper`** under a different prefix than the real install and broke **`mpicc --version`**; the rebuild sbatch avoids that.

Use **`sacct -j <rebuild_jobid> -o JobID,State,ExitCode`** to confirm the rebuild state before resubmitting scaling.

### AMDGPU / HIP (`kobayashi_fd_hip`)

Inspect queue health (partition often has **one node**; another user’s job can hold **`allocated`** while yours stays **`PD (Resources)`**):

```bash
apps/kobayashi/slurm/inspect_amdgpu_resources.sh amdgpu
```

Rebuild on an AMDGPU partition (ROCm + MPI; CPU HeFFTe is usually enough for this FD-only app — see root **`INSTALL.md`** §8):

```bash
export OPENPFC_REPO=$PWD
# optional: export CMAKE_PREFIX_PATH=/path/to/rocm:/path/to/heffte
sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_amdgpu.sbatch
```

Strong scaling **1 vs 2 MPI ranks** (expects **two GPUs** on the node; default rank list **`1 2`**):

```bash
export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-hip-amdgpu   # or your HIP build tree
sbatch apps/kobayashi/slurm/kobayashi_hip_scaling_amdgpu.sbatch
python3 apps/kobayashi/slurm/summarize_scaling.py kobayashi_hip_scaling_<jobid>/
```

Edit **`#SBATCH --partition`** / module names in the AMDGPU sbatch files if your site uses different queue or module identifiers; **`ROCM_MODULE`**, **`GCC_MODULE`**, and **`OPENMPI_MODULE`** environment variables are honored by the AMDGPU scripts where noted.

### NVIDIA H100 / CUDA (`kobayashi_fd_cuda`)

Rebuild on the **`nvidia_h100`** partition (CUDA + MPI; CPU HeFFTe is enough for this FD-only app). The rebuild script configures with **`OpenPFC_MPI_CUDA_AWARE=ON`** by default so **`kobayashi_fd_cuda`** can use **GPU-aware MPI** halos when Open MPI reports CUDA support (`MPIX_Query_cuda_support`). To force the old packed-only compile path, submit with **`KOBAYASHI_REBUILD_CUDA_MPI_AWARE=0`**.

```bash
export OPENPFC_REPO=$PWD
sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_cuda_h100.sbatch
```

Strong scaling **1 vs 2 MPI ranks** (expects **two GPUs** on the node; default rank list **`1 2`** via **`KOBAYASHI_CUDA_SCALING_NPROC`**, not **`KOBAYASHI_SCALING_NPROC`**, so CPU scaling exports do not leak in):

```bash
export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-cuda-h100   # or rely on default under submit dir
# optional: export CUDA_MODULE=cuda/13.0   # default in the sbatch on VTT H100 (GPU nodes ship CUDA 13 runtime)
sbatch apps/kobayashi/slurm/kobayashi_cuda_scaling_h100.sbatch
python3 apps/kobayashi/slurm/summarize_scaling.py kobayashi_cuda_scaling_<jobid>/
```

CPU vs GPU on **512×512**, **5000** steps (**`nvidia_h100`**, **non-exclusive**): one allocation (**32** Slurm tasks × **1** CPU/task = **32 CPUs** for **`kobayashi_fd_manual`** at **32 MPI ranks / 1 core per rank**), plus **`--gres=gpu:4`**, then **sequential** **`kobayashi_fd_cuda`** runs with **1**, **2**, and **4** MPI ranks. Edit **`#SBATCH --gres`** if your site needs a different GPU count.

```bash
export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-cuda-h100
sbatch apps/kobayashi/slurm/kobayashi_compare_cpu_gpu_h100.sbatch
# logs: kobayashi_compare_cpu_gpu_h100_<jobid>.out and kobayashi_compare_cpu_gpu_h100_<jobid>/summary.tsv
```

GPU-only **`kobayashi_fd_cuda`** on **`nvidia_h100`** with **one Slurm CPU per MPI rank** (separate jobs so each reserves only **`ntasks × 1`** CPUs plus matching **`gres/gpu`**):

```bash
export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-cuda-h100
sbatch apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np1_1cpu.sbatch
sbatch apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np2_1cpu.sbatch
sbatch apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np4_1cpu.sbatch
```

Larger **grid** (not stencil order — Kobayashi FD uses fixed **2nd-order** stencils): override **`KOBAYASHI_CMP_NX`**, **`KOBAYASHI_CMP_NY`**, **`KOBAYASHI_CMP_STEPS`** (same vars as the compare script). Example **4096×4096**, **5000** steps: `export KOBAYASHI_CMP_NX=4096 KOBAYASHI_CMP_NY=4096 KOBAYASHI_CMP_STEPS=5000` then `sbatch …/kobayashi_fd_cuda_h100_np1_1cpu.sbatch`. Runtime scales ~with **cell count per step**; reserve enough **`--time`**.

Single GPU on node **g0005** (same partition **`nvidia_h100`**; **`--nodelist=g0005`** pins the H100 node):

```bash
export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-cuda-h100
# optional problem size (else defaults 256×256, 2000 steps; overrides stray KOBAYASHI_SCALING_*):
#   export KOBAYASHI_G0005_STEPS=2000
sbatch apps/kobayashi/slurm/kobayashi_fd_cuda_g0005.sbatch
```

Use **`KOBAYASHI_CUDA_OPENPFC_BUILD_DIR`** as an alias for **`OPENPFC_BUILD_DIR`** when chaining jobs. The CUDA **rebuild** script ignores generic **`OPENPFC_BUILD_DIR`** so it is not accidentally set to an AMDGPU HIP build path.

If the CUDA link step fails with **`undefined reference to std::__cxx11::basic_string<...>::_M_replace_cold`** (or **`__cxa_call_terminate`**), the linker is using an older **`libstdc++.so`** than the **g++** that compiled OpenPFC (common when **`module load openmpi`** switches GCC **11 → 15**). CMake now resolves **`libstdc++.so`** via **`${CMAKE_CXX_COMPILER} -print-file-name`**. Override manually with **`export OPENPFC_LIBSTDCXX_SO=/path/to/libstdc++.so`** before configuring if needed.

Override **`CMAKE_CUDA_ARCHITECTURES`** at rebuild time if your GPUs are not **sm_90** (H100). Edit **`#SBATCH --partition`** if your centre names the queue differently.

### Submit scaling

```bash
export OPENPFC_BUILD_DIR=/path/to/your/openpfc/build   # must match MPI loaded in the sbatch (5.0.10)
# optional: export SBATCH_ACCOUNT=...
sbatch apps/kobayashi/slurm/kobayashi_scaling_gen05_epyc.sbatch
```

Adjust `#SBATCH` account, walltime, or **`--ntasks`** if your node has fewer than 192 cores.

### Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `GCC_MODULE` | `gcc/11.2.0` | Compiler module |
| `OPENMPI_MODULE` | `openmpi/5.0.10` | PMI-capable Open MPI for **`srun`** |
| `CUDA_MODULE` | `cuda` | Optional CUDA stack for future GPU binaries (warns if absent) |
| `SLURM_MPI_TYPE` / `KOBAYASHI_SRUN_MPI` | `pmix` | Passed as **`srun --mpi=…`**; run **`srun --mpi=list`** on the node — many VTT images expose **`pmix`** / **`pmix_v2`**, not **`pmix_v3`** |
| `KOBAYASHI_MPI_LAUNCHER` | `srun` | Set to **`mpirun`** if **`srun`** PMIx is broken on your partition (still inside the Slurm allocation) |
| `KOBAYASHI_OMPI_KEEP_OFI` | unset (`0`) | Set to **`1`** to skip **`OMPI_MCA_mtl=^ofi`** (keep Libfabric OFI MTL) |
| `KOBAYASHI_SCALING_NX` / `NY` | 512 | Grid |
| `KOBAYASHI_SCALING_STEPS` | 5000 | Time steps |
| `KOBAYASHI_SCALING_DT` | 1e-4 | Δt |
| `KOBAYASHI_SCALING_DX` | 0.03 | Δx (=Δy) |
| `KOBAYASHI_SCALING_NPROC` | `1 2 4 8 16 32 64 128 192` | Rank counts (space-separated) |
| `KOBAYASHI_OPENMP_THREADS` | `1 2 4 8 16 32 64 128 192` | OpenMP thread counts (space-separated) for **`kobayashi_openmp_scaling_*.sbatch`** |
| `KOBAYASHI_OPENMP_USE_SRUN` | `0` | Set **`1`** for **`srun --cpu-bind=cores`** per sweep (**`--cpus-per-task`** = thread count for that step) |
| `KOBAYASHI_OPENMP_SRUN_CPUS` | unset | Override **`srun --cpus-per-task`** (default = sweep **`OMP_NUM_THREADS`**) |
| `KOBAYASHI_NUMA_INTERLEAVE` | `0` | Set **`1`** to wrap runs with **`numactl --interleave=all`** (requires **`numactl`**) |
| `OMP_PROC_BIND` / `OMP_PLACES` | `spread` / `cores` | OpenMP affinity (override as needed) |
| `OMP_DYNAMIC` | `false` | Set in the OpenMP scaling sbatch unless overridden in the environment |
| `OMP_WAIT_POLICY` | unset | Not forced by the sbatch; avoid defaulting to **`passive`** for this benchmark |

### Outputs

Under `kobayashi_scaling_<SLURM_JOB_ID>/`:

- `run_np_<n>.log` — full stdout (includes `KOBAYASHI_VERIFY` lines)
- `summary.tsv` — quick wall-time table

### Verification + scaling table

After the job finishes:

```bash
python3 apps/kobayashi/slurm/summarize_scaling.py kobayashi_scaling_<jobid>/
python3 apps/kobayashi/slurm/plot_strong_scaling.py kobayashi_scaling_<jobid>/   # writes strong_scaling.svg next to summary.tsv
```

This prints **speedup** (vs first log, usually **nproc=1**) and **parallel efficiency** \(\frac{T_1}{N T_N}\times 100\%\). It also compares **`KOBAYASHI_VERIFY_HEX`** (`sum_phi`, `sumsq_phi`, `sum_T`, `sumsq_T`) across runs: **matching hex strings mean bitwise-identical reductions** on the globally assembled fields (MPI decomposition is deterministic for this discretisation).

**Note:** We intentionally accumulate global sums in **fixed \((g_x,g_y)\) order on rank 0** after an `MPI_Gatherv`, so the reported scalars are **reproducible** and comparable across rank counts. Using `MPI_Allreduce` on partial sums would **not** guarantee bitwise-identical floating-point sums when `nproc` changes.

## Interpreting scaling

For a **fixed 512² problem**, you are measuring **strong scaling**: ideal speedup is linear in `nproc`; efficiency drops when communication and subdomain size dominate. This FD driver is **not** threaded — each rank runs its owned patch serially — so you expect good scaling only while local work per rank stays large enough relative to halo exchange.

### Flat wall times and multiple `KOBAYASHI_VERIFY` lines

If **`summary.tsv`** shows almost the same seconds for every rank count, open **`run_np_<n>.log`** and check:

1. **`KOBAYASHI_MPI_COMM_WORLD_SIZE`** (printed once on rank 0 at startup) must equal **`n`** for that run. If it prints **`1`** while **`srun -n N`** launched **`N` tasks**, those tasks are **not** sharing one `MPI_COMM_WORLD` (often **singleton MPI**): each process solves the **full** grid in parallel with the others, so wall time does not drop and you may see **`N` copies** of **`KOBAYASHI_VERIFY`** all reporting **`nproc=1`**.
2. **Binary vs runtime MPI:** run **`ldd "${BIN}" | grep libmpi`** on the compute node (as in the rebuild sbatch). The resolved **`libmpi.so`** must live under the **same** Open MPI prefix as **`module show openmpi/...`**. A binary linked against **one** Open MPI tree while the job **`PATH`** / **`LD_LIBRARY_PATH`** / **`srun`** environment implies **another** is a common way to get broken or singleton behaviour.
3. **Transport noise:** repeated **`mtl_ofi` / `fi_domain`** errors suggest Libfabric/OFI problems on that host; they can distort or break multi-rank startup. Site-specific workarounds (e.g. forcing **`ob1`/`vader`** on shared memory, or UCX if installed) may be needed — ask your centre’s MPI notes.

After fixing MPI wiring, you should see **exactly one** **`KOBAYASHI_VERIFY`** line per log and **`nproc=<n>`** matching the filename **`run_np_<n>.log`**.
