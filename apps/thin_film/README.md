<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Thin film (`apps/thin_film`)

Lubrication **dewetting / coating** of a periodic liquid film, integrated
with spectral ETD. This is the 0.2 application for GitHub issue `#78`.

Binaries: `thin_film` (CPU); `thin_film_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL`
is on.

## Physics

One field `h` (film thickness) with constant mobility \(M_0\):

\[
\partial_t h=\nabla\cdot\bigl[M_0\nabla p\bigr],\qquad
p=-\gamma\nabla^2 h-\Pi(h).
\]

Flow is down the pressure gradient, so capillary \(\nabla^4\) *damps* short
waves. (A leading minus on the divergence with this \(p\) would reverse that
and is not used.) Cubic mobility \(M\propto h^3\) is a flux nonlinearity; this
slice uses \(M_0=M(h_0)\), which is exact for the linear band.

Disjoining pressure (mean thickness \(h_0\)):

\[
\Pi(h)=A\bigl((h_0/h)^3-(h_0/h)^9\bigr)
\]

(van der Waals attraction plus short-range repulsion). \(A=0\) is a leveling
coating.

In Fourier space, with OpenPFC \(k_{\mathrm{lap}}=-|k|^2\),

\[
L(k)=-M_0\gamma\,k_{\mathrm{lap}}^2-M_0\Pi'(h_0)\,k_{\mathrm{lap}}.
\]

Linear growth:

\[
\lambda(k)=M_0 k^2\bigl(\Pi'(h_0)-\gamma k^2\bigr).
\]

The fastest-growing mode is \(k_{\mathrm{peak}}^2=\Pi'(h_0)/(2\gamma)\) when
\(\Pi'(h_0)>0\). Capillary \(k^4\) is a multiply.

## Run

```bash
mkdir -p results/thin_film
mpirun -n 1 ./apps/thin_film/thin_film \
  ../apps/thin_film/inputs_json/dewetting.json
mpirun -n 1 ./apps/thin_film/thin_film \
  ../apps/thin_film/inputs_json/leveling.json
```

Dewetting: 128², \(A=0.05\), cosine near \(k_{\mathrm{peak}}\). Leveling:
\(A=0\), roughness decays. VTK of `h` under `results/thin_film/`.

JSON `model.params`: `h0`, `gamma`, `M0`, `A`. Initial condition
`"type": "cosine_mode"` with `h0` / `amplitude` / `nx` / `ny` / `nz`.

## Tests

`ctest -R thin-film` checks \(\Pi'(h_0)\), the \(L(k)\) formula, volume
conservation, an unstable mode vs \(\exp(\lambda t)\), a stable high-\(k\)
mode decaying, and \(A=0\) leveling. HIP builds add `HIP_ThinFilmETD` and
`thin-film-hip-smoke`. LUMI-G smoke: job 21781449 (`small-g`, 16², mean
\(h=1\)).

## Layout

| Path | Role |
|------|------|
| `include/thin_film/thin_film_physics.hpp` | `SpectralETDPhysics` + schema |
| `include/thin_film/thin_film_pointwise.hpp` | Device-capable \(\Pi\) remainder |
| `include/thin_film/thin_film_session.hpp` | JSON session, field `h`, 2/3 dealias |
| `src/thin_film.cpp` / `src/hip/thin_film.cpp` | CPU / HIP `main` |
| `inputs_json/dewetting.json` | Unstable coating |
| `inputs_json/leveling.json` | \(A=0\) capillary smoothing |
