<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: end-to-end run and artifacts

This tutorial connects **configure → run → files you can inspect**. Prerequisites: a working OpenPFC build per [`INSTALL.md`](../../INSTALL.md) (MPI, HeFFTe, modules as needed).

## 1. Build the project

From the repository root:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

Ensure examples and apps are enabled (defaults): `OpenPFC_BUILD_EXAMPLES=ON`, `OpenPFC_BUILD_APPS=ON`. If you disabled them earlier, reconfigure with `ON` and rebuild.

---

## Track A — PNG snapshots (2D Allen–Cahn, fastest visual)

Goal: two PNG files (initial and final field) without writing JSON.

1. From your **build directory**:

   ```bash
   cd build
   mkdir -p png_out
   mpirun -n 4 ./apps/allen_cahn/allen_cahn 128 128 500 0.0015 2.0 0.35 \
     png_out/initial.png png_out/final.png
   ```

2. **Success**: `mpirun` exits with status 0. On rank 0 you should find `png_out/initial.png` and `png_out/final.png` (grayscale, fixed \[-1,1\] scale — see [`io_results.md`](../io_results.md)).

3. **Details**: full CLI reference — [`apps/allen_cahn/README.md`](../../apps/allen_cahn/README.md).

---

## Track B — Config-driven Tungsten + binary field dumps

Goal: run the **Tungsten** `App` with JSON and produce **MPI-IO binary** time slices under a directory you control.

The stock file [`apps/tungsten/inputs_json/tungsten_single_seed.json`](../../apps/tungsten/inputs_json/tungsten_single_seed.json) uses a **small** \(32^3\) domain but ships with **absolute** `fields[].data` paths. For a self-contained tutorial, copy it and point output at `./data/`.

1. From `build/`, create a writable config (example edits):

   ```bash
   cd build
   mkdir -p data
   cp ../apps/tungsten/inputs_json/tungsten_single_seed.json ./tutorial_tungsten.json
   ```

2. Edit `tutorial_tungsten.json`:

   - Set each `fields[]` entry’s `"data"` to a path under your build tree, e.g. `"./data/psi_%d.bin"` and `"./data/psimf_%d.bin"` (same pattern the stock file uses for the two fields).
   - Keep `"saveat"` > 0 in `timestepping` so writers run (the sample uses `"saveat": 1.0`).

3. Run (CPU binary):

   ```bash
   mpirun -n 4 ./apps/tungsten/tungsten ./tutorial_tungsten.json
   ```

4. **Success**: exit code 0; rank 0 logs progress. **Artifacts**: binary files under `./data/` whose names match your `data` templates (frame index in place of `%d`). **On-disk layout:** [`binary_field_io_spec.md`](../binary_field_io_spec.md); overview: [`io_results.md`](../io_results.md).

5. **Visualization**: VTK/ParaView from the default JSON path is not automatic — the stock wiring registers binary writers. For VTK, attach `VTKWriter` in code (see [`io_results.md`](../io_results.md) and `examples/11_write_results.cpp`). For large campaigns, binary + postprocessing is typical.

---

## What to read next

| Goal | Document |
|------|----------|
| Full app comparison (JSON vs CLI, GPU) | [`applications.md`](../applications.md) |
| JSON → `Simulator` order | [`app_pipeline.md`](../app_pipeline.md) |
| Example executables by topic | [`examples_catalog.md`](../examples_catalog.md) |
| Figures and which runnable produced them | [`showcase.md`](../showcase.md) |
