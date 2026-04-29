<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Frequently asked questions

Short answers; deeper detail lives in **[`INSTALL.md`](../INSTALL.md)**, **[`architecture.md`](architecture.md)**, **[`quickstart.md`](quickstart.md)**, and **[`troubleshooting.md`](troubleshooting.md)**.

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

**How do I know an example or tungsten run succeeded?**  
Expect **`mpirun`** exit code **0** and rank-0 **INFO** logs. Examples do not all print the same banner; apps may write result files when configured. Short checklist: **[`quickstart.md`](quickstart.md)** (sections **2A** / **2B**).

## Extending the framework

**How do I add a custom model or IC?**  
See **[`extending_openpfc/README.md`](extending_openpfc/README.md)**, **[`class_tour.md`](class_tour.md)** (where types live), **[`app_pipeline.md`](app_pipeline.md)** (JSON sections), and examples **`14_custom_field_initializer.cpp`**, **`17_custom_coordinate_system.cpp`**, **`10_ui_register_ic.cpp`**. For an out-of-tree binary with **`App<Model>`** and a config file, follow **[`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md)**. Optional startup validation of **`model.params`**: **[`parameter_validation.md`](parameter_validation.md)**.

## Documentation map

| Need | Document |
|------|-----------|
| Index of all guides | [`README.md`](README.md) |
| Onboarding | [`quickstart.md`](quickstart.md) |
| Troubleshooting | [`troubleshooting.md`](troubleshooting.md) |
| Config files | [`configuration.md`](configuration.md) |
| Terminology | [`glossary.md`](glossary.md) |
| `App` + JSON pipeline | [`app_pipeline.md`](app_pipeline.md) |
| Main types / headers map | [`class_tour.md`](class_tour.md) |
| Minimal custom `App` + CMake | [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md) |
| Parameter validation | [`parameter_validation.md`](parameter_validation.md) |
| CMake options | [`build_options.md`](build_options.md) |
| Editing documentation | [`contributing-docs.md`](contributing-docs.md) |
| Contributing (code, tests, changelog) | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Release history / upgrades | [`../CHANGELOG.md`](../CHANGELOG.md) |
| Examples folder | [`../examples/README.md`](../examples/README.md) |
| Published HTML API | [GitHub Pages dev docs](https://vtt-propertune.github.io/OpenPFC/dev/) |

---

## Future documentation improvements (ideas)

Smaller polish items that may still help:

1. **Session transcripts** — Optional verbatim log excerpts for **`05_simulator`** / minimal tungsten (beyond the success hints in **[`quickstart.md`](quickstart.md)** **§2A** / **§2B**).
2. **CHANGELOG user notes** — Keep meaningful entries under **`[Unreleased]`** in **[`CHANGELOG.md`](../CHANGELOG.md)**; add “upgrading from X” blurbs when CMake or config keys change.
3. **Published site** — Keep GitHub Pages in sync with **[`quickstart.md`](quickstart.md)** and **[`README.md`](README.md)** so API-only readers find the prose guides.

**Link checks:** run **`python3 scripts/check_doc_links.py`** before merging doc changes (see **[`contributing-docs.md`](contributing-docs.md)**).
