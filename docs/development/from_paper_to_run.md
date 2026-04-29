<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# From paper to run

The canonical publication for the framework is linked from the root [`README.md`](../../README.md) (Modelling Simul. Mater. Sci. Eng., 2024). This page maps **literature → repository entry points** without promising bit-for-bit reproduction of every figure (hardware, random seeds, and undocumented details always matter).

| Paper concept | Start in-repo |
|-----------------|----------------|
| Large-scale PFC / tungsten-style solidification | Shipped app: [`applications.md`](../user_guide/applications.md) → **Tungsten**; inputs [`apps/tungsten/inputs_json/README.md`](../../apps/tungsten/inputs_json/README.md); science notes [`science_tungsten_quicklook.md`](../science/tungsten_quicklook.md) |
| Framework FFT / MPI scaling story | [`INSTALL.md`](../../INSTALL.md), [`spectral_stack.md`](../concepts/spectral_stack.md), [`performance_profiling.md`](../hpc/performance_profiling.md) |
| Learning the API incrementally | [`examples_catalog.md`](../reference/examples_catalog.md) tiers, [`tutorials/spectral_examples_sequence.md`](../tutorials/spectral_examples_sequence.md) |

**Reproducibility checklist**

1. Record **git commit** or **release tag**, **HeFFTe** build, **MPI** modules.  
2. Store the **exact JSON/TOML** and job script.  
3. Note **rank count** and **domain dimensions**; binary outputs need sidecar metadata ([`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).

## See also

- [`documentation_versioning.md`](documentation_versioning.md) — prose vs release tags  
