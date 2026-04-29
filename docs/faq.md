<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Frequently asked questions

Short answers; deeper detail lives in **[`INSTALL.md`](../INSTALL.md)**, **[`architecture.md`](architecture.md)**, and **[`quickstart.md`](quickstart.md)**.

## Getting started

**Where do I begin after cloning the repo?**  
Follow **[`quickstart.md`](quickstart.md)** (configure → run an example or an app → or link OpenPFC from your own CMake project).

**Is OpenPFC header-only?**  
No. You **link** the compiled **`openpfc`** library and include headers from **`include/openpfc/`**. See [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md).

**Do I need MPI?**  
Yes for the documented workflows: the distributed FFT stack and examples assume an **MPI**-enabled build. There is no supported serial-only configuration.

## Build and CMake

**`find_package(OpenPFC)` cannot find OpenPFC**  
Install OpenPFC (or point at the build tree if your workflow exports the package), then set **`CMAKE_PREFIX_PATH`** to the installation prefix, or **`-DOpenPFC_DIR=/path/to/lib/cmake/OpenPFC`**. See the CMake error walkthrough in [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md).

**Examples or apps are missing from my build directory**  
Ensure **`OpenPFC_BUILD_EXAMPLES=ON`** and **`OpenPFC_BUILD_APPS=ON`** (both default **ON**). If you previously configured with **OFF**, clear the option or delete the build directory and reconfigure.

**CUDA vs CPU build**  
Use **separate build trees** when toggling GPU options so CMake does not mix flags; see [`build_cpu_gpu.md`](build_cpu_gpu.md).

## Running

**Where are the example executables?**  
Under **`<build>/examples/`** when examples are enabled. Names match the source basename (e.g. `05_simulator`). Full list: [`examples_catalog.md`](examples_catalog.md).

**Tungsten / app cannot find my JSON**  
Pass an **absolute path**, or a path relative to your **current working directory** (often the `build/` folder). Stock samples live under **`apps/tungsten/inputs_json/`** in the source tree.

## Extending the framework

**How do I add a custom model or IC?**  
See **[`extending_openpfc/README.md`](extending_openpfc/README.md)** and the numbered examples **`14_custom_field_initializer.cpp`**, **`17_custom_coordinate_system.cpp`**.

## Documentation map

| Need | Document |
|------|-----------|
| Index of all guides | [`README.md`](README.md) |
| Onboarding | [`quickstart.md`](quickstart.md) |
| Published HTML API | [GitHub Pages dev docs](https://vtt-propertune.github.io/OpenPFC/dev/) |

---

## Future documentation improvements (ideas)

These are **not** implemented as full guides yet; they would further lower the barrier for new users and operators.

1. **Troubleshooting appendix** — Collate common configure/runtime errors (`mpi.h` not found, HeFFTe mismatch, stale `CMakeCache`) with one fix each; could live as a new section in **`INSTALL.md`** or a dedicated **`docs/troubleshooting.md`**.
2. **Configuration reference** — One place describing shared TOML/JSON sections (`domain`, `timestepping`, `plan_options`, profiling blocks) with links to validators and example files under **`apps/`** and **`examples/`**.
3. **IDE / cluster presets** — Short pointer from **`quickstart.md`** to **`CMakePresets.json`** and **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`** for reproducible configure without interactive `module load`.
4. **Glossary** — PFC-specific terms (order parameter, inbox/outbox, halo, spectral vs FD) with links to **`architecture.md`** and **`halo_exchange.md`**.
5. **“First failure” screenshots or session transcripts** — Optional: log snippets for a successful `05_simulator` and a minimal tungsten run for visual reassurance.
6. **Doxygen main page** — Ensure the published site’s landing page highlights **`quickstart.md`** (if Breathe/Sphinx ever wraps markdown) or duplicate a one-paragraph onboarding blurb in **`README.md`** used as Doxygen main page.
7. **CHANGELOG user notes** — For releases, a short “upgrade from X” subsection when CMake options or config keys change.
