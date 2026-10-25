<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Polycrystal shear

Finite-strain simple shear of a periodic FCC polycrystal. Orientations are
piecewise constant on a cubic grain grid. Each increment calls the existing
Newton–GMRES equilibrium solver and the DAMASK aluminum slip law, then
writes one mid-plane scalar of the symmetric Piola stress. The scalar is
the Frobenius norm of the deviator of that symmetric tensor. It is not
`von_mises`, which multiplies the full-tensor deviator by sqrt(3/2).

```bash
mpirun -n 1 ./polycrystal_shear --size 4 --steps 2 --shear 0.002 \
  --outdir results/polycrystal-smoke --commit test
```

`openpfc_render_polycrystal --size N --steps K --outdir DIR --fps 60`
encodes `microstructure.mp4` from those slices at the chosen frame rate.
Run it after the solver; the solver writes data only. The encoder is `ffmpeg` on `PATH`, or the executable
named by `FFMPEG`. That path is one argument, not a shell command. A
missing encoder fails rendering while leaving solver output available. An output directory
that cannot be created stops every rank before the solve. The launch
script and the bulky frames stay outside this repository.
