<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Kawahara (`apps/kawahara`)

Capillary–gravity **dispersive waves** with competing third- and fifth-order
dispersion. This is the 0.2 application for GitHub issue `#80`.

Binaries: `kawahara` (CPU); `kawahara_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL`
is on.

## Physics

Long weakly nonlinear free-surface waves can need both \(\partial_x^3\) and
\(\partial_x^5\) when the leading dispersive term is small or changes sign.
The Kawahara equation (1972) is

\[
\partial_t u + \alpha u\partial_x u + \beta\partial_x^3 u + \gamma\partial_x^5 u = 0.
\]

Defaults \(\alpha=1\), \(\beta=1\), \(\gamma=-1\) are the classic
capillary–gravity regime. This is a 1D equation on a periodic line
(\(N_y=N_z=1\)).

Linear Fourier modes rotate rather than decay:

\[
\omega(k)=\beta k^3+\gamma k^5,\qquad
L(k)=-i\omega(k),\qquad
u_k(t)=u_k(0)\,e^{-i\omega(k)t}.
\]

Phase velocity \(c_p=\omega/k=\beta k^2+\gamma k^4\). With the default
signs, \(c_p>0\) for \(|k|<1\) (third-order wins) and \(c_p<0\) for
\(|k|>1\) (fifth-order wins). That is the opposite of even-order
dissipative operators such as Cahn–Hilliard or Mullins surface diffusion,
where \(k^4\) is a real negative multiplier and short waves simply damp.

The quadratic term is evaluated pseudospectrally as \(N=u^2\) with
\(M(k)=-i(\alpha/2)k_x\), under the Orszag 2/3-rule.

## Run

```bash
mkdir -p results/kawahara
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/pulse.json
```

The shipped input is a 256-point line (\(L_x=64\pi\)) with a Gaussian pulse.
VTK of `u` goes to `results/kawahara/`. The packet travels and radiates
dispersive ripples; it does not flatten the way a \(k^4\) smoother would.

JSON `model.params`: `alpha`, `beta`, `gamma`. Initial conditions:
`"type": "cosine_mode"` (`u0`/`amplitude`/`nx`) or `"gaussian_pulse"`
(`amplitude`/`sigma`/`x0`).

## Tests

`ctest -R kawahara` checks \(\omega(k)=\beta k^3+\gamma k^5\), a linear
cosine against that dispersion with no amplitude loss, opposite phase
velocities on either side of \(|k|=1\), and mean-\(u\) conservation with
the quadratic term on. HIP builds add `HIP_KawaharaETD` and
`kawahara-hip-smoke`. LUMI-G smoke: job 21792406 (`small-g`, 32-point
line, mean \(u=0\)).

## Layout

| Path | Role |
|------|------|
| `include/kawahara/kawahara_physics.hpp` | Complex \(L(k)=-i\omega(k)\) |
| `include/kawahara/kawahara_pointwise.hpp` | \(N=u^2\) |
| `include/kawahara/kawahara_session.hpp` | JSON session, field `u`, 2/3 dealias |
| `src/kawahara.cpp` / `src/hip/` | CPU / HIP `main` |
| `inputs_json/pulse.json` | Localized long-wave pulse |
