<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Polycrystal shear

Finite-strain simple shear of a periodic FCC polycrystal. Orientations are
piecewise constant on a cubic grain grid. Each increment calls the existing
Newton–CG equilibrium solver and the DAMASK aluminum slip law, then writes
the mid-plane von Mises scalar of the symmetric Piola stress.

```bash
mpirun -n 1 ./polycrystal_shear --size 4 --steps 2 --shear 0.002 \
  --outdir results/polycrystal-smoke --commit test
```

`--render 1` encodes `microstructure.mp4` from those slices at `--fps`
(60 by default). The launch script and the bulky frames stay outside this
repository.
