<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Solved heat and a substrate PFC line

This is the one-way coupling between a solved temperature line and a
conserved density line. It is not an alloy model. The aluminum application
still prescribes a clamped temperature and still uses a density reservoir
inside a periodic box. Those reservoirs are not this substrate, and this
header does not read that preset.

## Ownership, units, and time

The caller owns both arrays. Each node of one non-periodic `DiffusionAxis`
holds one temperature sample and one density sample. The views must be
nonoverlapping, including shifted subspans of the same backing allocation.
A partially shared buffer is rejected before either field or the flux ledger
is changed. There is no second grid and no interpolation. The time step is the same value for heat and density.
There is no subcycling. `advance_one_way_substrate` advances heat first and
then reads the resulting temperature. The density step does not write
temperature.

Both fields are code units. `temperature_scale`, `melting_level`, `cubic`,
`wave_number`, `mobility`, and `diffusivity` are caller coefficients. Nothing
here converts them to kelvin, metres, or a named material.

Latent heat, interface heat release, and any other two-way source are not
part of this contract.

## Heat

The heat line is

∂t T = ∂x(α ∂x T)

on `DiffusionAxis::apply`, with the material coefficient equal to
`diffusivity`. The integrator is RK4. The first node is the substrate end.
Its outward heat flux is the constitutive value `substrate_flux`. The last
node carries `far_flux`. Both are `FaceCondition` data, not an
application-local ghost. Positive outward flux removes thermal inventory.
`FluxLedger` records that flux. The weighted temperature change of an
accepted step matches the accepted integral.

A periodic axis has no face condition. Passing one is rejected by
`DiffusionAxis`. A failed heat trial rejects the ledger and does not write
the temperature array. Coefficient checks throw before `ledger.begin()`.

## Density

`SolidificationBoundary` keeps three meanings apart.

- `Periodic` is a periodic direction. It is not a wall and not a reservoir.
- `Reservoir` imposes a density value. It is not a wall and not this heat
  line. This header does not advance a reservoir.
- `Substrate` is a physical wall. It is not a density reservoir and not
  periodic. The density condition is a zero outward derivative.

`reject_free_surface` throws. A free surface is not an enumerator, and it is
not another name for the reservoir or the substrate.

`advance_substrate_pfc` runs only for `Substrate`. The chemical potential is

μ = r(T) n + (q0² + ∂xx)² n + cubic n³,

r = temperature_scale (T − melting_level),

with q0 = `wave_number`. The square is two public Laplacians, each closed by
the zero outward derivative. The rate is `mobility` times one more
`DiffusionAxis::apply` of μ, again with that wall condition. The density step
is forward Euler, so the difference of two temperature readings divided by
the step is the difference of those rates. The weighted density change of
that step is the wall balance: zero outward derivative, so no density flux.

## What this does not claim

This is not a grain boundary, not a free surface, and not a casting
prediction. The periodic spectral solidification applications are unchanged.
Material calibration stays with the application that chooses the
coefficients.
