<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# The GPU path of `vlasov_maxwell` (LUMI-G, HIP)

What was ported to the device, what was deliberately left on the host, the
measurement that decided each, the CPU/GPU parity tolerances and where they
come from, and the limitations. Every number is from a compute-node run and
carries its Slurm job id; nothing here was measured on a login node.

This page covers issue #84's "CPU / MPI / GPU path" requirement and its
acceptance criterion *"CPU/GPU parity on a small case, to a stated
tolerance"*. The physics and the numerics are documented by the application's
own headers; start at
`apps/vlasov_maxwell/include/vlasov_maxwell/parameters.hpp`.

## The files

| File | What it is |
|------|------------|
| `apps/vlasov_maxwell/include/vlasov_maxwell/device_step_hip.hpp` | kernel declarations, the device-resident brick, and `DeviceStepper` — an alternative driver of the *same* `vlasov::Stepper` state |
| `apps/vlasov_maxwell/src/hip/vlasov_device_kernels.hip` | every kernel; the only translation unit hipcc compiles |
| `apps/vlasov_maxwell/src/hip/vlasov_hip_parity.cpp` | `vlasov_hip_parity`, the CPU-vs-GPU comparison |
| `apps/vlasov_maxwell/src/hip/vlasov_hip_cost.cpp` | `vlasov_hip_cost`, the per-phase measurement |

The CPU application is untouched. `advect.hpp`, `moments.hpp`, `maxwell.hpp`
and `step.hpp` remain the definition of what the application computes.
`vlasov_run` reaches the same kernels behind `--device=hip` (optional
`--device-x=0` leaves the spectral \(x\)-shift on the host). The two extra
binaries answer different questions: `vlasov_hip_parity` whether the device
computes the same thing as the host, and `vlasov_hip_cost` where the time
goes.

## Build and run

```bash
# HIP tree (the flag is --with-rocm; there is no --hip)
./scripts/build.sh --machine=lumi --with-rocm --no-submit --no-test --jobs=32 \
    --cmake-arg=-DOpenPFC_BUILD_TESTS=ON \
    --build-dir=/flash/project_462001519/$USER/build/openpfc-lumi-rocm

# one GCD
srun -n 1 --gpus-per-node=8 vlasov_hip_parity --nx=32 --nvx=32 --nvy=32 --steps=20
srun -n 1 --gpus-per-node=8 vlasov_hip_cost  --sizes=64,128,192,256 --reps=5
```

`vlasov_hip_parity` exits non-zero if any check fails, so it can be wired to
`ctest` as it stands. Both binaries need hipFFT (`find_package(hipfft)` in
the application's `CMakeLists.txt`); the CPU build never looks for it.

On LUMI-G use whole nodes (`--ntasks-per-node=8 --gpus-per-node=8`) and the
CPU mask from `docs/lumi_slurm/tungsten_hip_scaling.sbatch`. Do **not** set
`ROCR_VISIBLE_DEVICES`: the binaries call `bind_local_device(local_rank)` and
need every GCD visible.

## What runs where

The Strang step of `step.hpp` has five phases:

| phase | operator | work | where |
|-------|----------|------|-------|
| A | `advect_x`, exact spectral shift along `x` | `N log N` per line, `N_vx N_vy` lines | **device** (hipFFT) |
| B | `advect_vx`, semi-Lagrangian gather | `p` FMA per cell | **device** |
| C | `advect_vy`, semi-Lagrangian gather along the distributed axis | `p` FMA per cell | **device** |
| D | `rho`, `J_x`, `J_y` and the kinetic Casimirs | one pass over the brick | **device** |
| E | Gauss, Ampère and the ETD2 transverse pair | `N_x` | **host** |

Two further pieces are split rather than placed:

- **The velocity-space halo exchange** stays an `MPI_Sendrecv` pair on
  host-staged slabs. A `v_y` ghost slab is `hw N_x N_vx` doubles — kilobytes
  against a gigabyte brick — and staging it through host memory costs less
  than depending on the site's MPI being GPU-aware. Measured below as
  `halo`.
- **The Lagrange coefficients.** The host computes each advected line's
  departure cell and fractional offset, because that needs the fields and,
  on `v_y`, the collective halo guard that has to be able to throw; the
  device expands each fraction into `p` weights, because that is
  `p(p-1)` *divisions* per line and divisions were what the host was
  actually spending its time on. Measured below as `coeffs`.

### What was not ported, and why

**Phase E, the field solve.** `E_x`, `E_y` and `B_z` are 1-D arrays of `N_x`
doubles — 8 kB at the resolution this application targets, against a 2 GB
distribution. The measured cost is 0.13 ms per step at `256³` — 0.012% of
the host step and 1.9% of the fully device-resident one — and the phase is
roughly a dozen short 1-D transforms, so the kernel launches alone would be
the same order as the whole phase. This is not a deferral; there is nothing
there to win.

**The `v_y` halo exchange, as a GPU-aware MPI transfer.** On one GCD the
host-staged exchange is 0.02 ms per step at `256³`, 0.3% of the device step,
and a device-pointer send would save a fraction of that while making the
application depend on a property of the site's MPI build. On **eight** GCDs
the same exchange is 1.05 ms per step and half the step at `128³`, so the
number that says "do not bother" on one GCD says something else on eight —
and the honest reading is that most of the eight-GCD cost is skew and slab
staging together, which a GPU-aware send would only partly remove. Not done;
measured; written down rather than assumed either way.

**`vlasov_run` itself.** The science driver takes `--device=hip` (and
`--device-x=0|1`). `DeviceStepper` is a drop-in for the `advance` /
`deposit_all` pair; a production Landau or Weibel run can stay on the device
for the whole step.

### The scope decision, in the order it was actually made

The instruction was to measure before porting, and the measurement was made
twice, because the first answer stopped being the relevant one as soon as it
was acted on.

**Round one — where does the host step go?** At `256³`, phases B, C and D are
**82%** of the host Strang step and phase A is 9%. So B, C and D are the
right things to port, which is what the issue's minimum asked for, and the
measurement agreed with it rather than merely permitting it.

**Round two — where does the *ported* step go?** With B, C and D on the
device and A left on the host, the step at `256³` fell from 1.04 s to 0.127 s
(8.2×) and its composition inverted: phase A on the host was now **77%** of
what remained and the four whole-brick bus transfers it forces were another
**19%**. The three kernels the port existed for were 2.6% of it.

That second number is what justified porting phase A as well — not "an FFT
would be faster on a GPU", which was never in doubt, but "leaving one `O(N³)`
phase on the host pins the distribution to the bus and caps the whole
exercise at 8×". With phase A on the device the brick never leaves it, the
transfers disappear entirely, and the speedup at `128³` goes from 6.7× to
94×. Both configurations are still buildable and both are measured, because
the comparison is the argument.

The same logic applied once more, at smaller scale, to the coefficient setup.
With the weights computed on the host and uploaded, `coeffs` was 5.67 ms per
step at `256³` (job `21932623`) — `p(p-1)` divisions per advected line, and
more than every device kernel in the step put together. Moving the *weight
expansion* to the device while leaving the departure offsets on the host took
it to 0.64 ms (job `21933310`). The offsets did not move, because they carry
the collective halo guard, and because they are cheap: what was expensive was
the divisions.

## The measurement

### Provenance

| job id | partition | what |
|--------|-----------|------|
| `21932623` | `standard-g`, 1 node, 8 GCDs | first cost sweep, phase A on host only; and the first parity run, whose two failing columns were the tolerance bug described below |
| `21933090` | `dev-g`, 1 node | bring-up of the device phase A |
| `21933340` | `dev-g`, 1 node, 8 GCDs | full sweep with the final binaries, used as a cross-check |
| `21933310` | `standard-g`, 1 node, 8 GCDs | **every number in this section**: six parity configurations and the full cost sweep with the final binaries |

MI250X (`gfx90a`), ROCm through `LUMI/25.09 partition/G cpeGNU`, double
precision throughout, `Release`, `interp_order = 5` unless stated. The
binaries were copied to a private path before submission so that a
concurrent rebuild could not swap them under a running job. Every run used
the LUMI-G 8-GCD CPU mask from `docs/lumi_slurm/tungsten_hip_scaling.sbatch`.

The `standard-g` job `21933310` and the `dev-g` job `21933340` were the same
binaries on the same shapes: the full Strang step at `256³` came out at
1.0363 s vs 1.0297 s on the host and 6.616 ms vs 6.718 ms on the device, and
every parity number agreed to the digit. The partition is not a variable.

### Where the host step goes

One rank, one core for the host phases, a Weibel-like state, `measure_mass`
off. Seconds per call; A, B and D are called twice per Strang step, C once.

| phase | `64³` | `128³` | `192³` | `256³` | `128×256×256` |
|-------|-------|--------|--------|--------|----------------|
| A `advect_x` (dt/2) | 0.000828 | 0.005463 | 0.02112 | 0.04721 | 0.02748 |
| B `advect_vx` (dt/2) | 0.002986 | 0.022891 | 0.07866 | 0.18933 | 0.09243 |
| C `advect_vy` (dt) | 0.002156 | 0.018002 | 0.06644 | 0.15503 | 0.07859 |
| D moments | 0.002266 | 0.018479 | 0.06577 | 0.15703 | 0.07776 |
| D moments, memory order | 0.002202 | 0.017984 | 0.06194 | 0.14812 | 0.07449 |
| E fields | 0.000024 | 0.000056 | **0.005494** | 0.000127 | 0.000057 |
| **full Strang step** | **0.01585** | **0.12418** | **0.44751** | **1.03634** | **0.52452** |

At `256³` the composition of the measured step is B 37%, D 30%, C 15%,
A 9%, E 0.01% — so **B, C and D are 82%** and that is the first measurement.
It says to port B, C and D, which is what the issue's minimum asked for; the
measurement agreed with it rather than merely permitting it.

Two host-side observations that are not about the GPU at all:

- **`N_x = 192` costs 5.5 ms in phase E.** `maxwell.hpp`'s `dft` is a
  radix-2 FFT for power-of-two lengths and an honest `O(N²)` direct sum
  otherwise, and the 1-D field solve runs several transforms per step. At
  `N_x = 192` that turns a free phase into 64% of the fully device-resident
  step. It is a property of the CPU application, not of the port, and it is
  why the `192³` speedup below is an outlier. Anyone running this
  application at a non-power-of-two `N_x` should know.
- **The host reduction runs at 0.9 GB/s, and the traversal order is not
  why.** `moments.hpp` iterates its view with `v_y` innermost, which in the
  padded x-fastest brick is a stride of `npx npy` doubles — half a megabyte
  at `256³` — so it looked like the obvious suspect. The cost driver
  therefore times a second, memory-order traversal of exactly the same cells
  (`OrderedView` in `vlasov_hip_cost.cpp`) and it is only 2–6% faster:
  0.1481 s against 0.1570 s at `256³`. The reduction is bound by the `log`
  in the entropy column, not by its memory order. Worth knowing before
  anyone "fixes" the loop.

### Where the ported step goes

Same runs, one GCD, seconds per call.

| phase | `64³` | `128³` | `192³` | `256³` | `128×256×256` |
|-------|-------|--------|--------|--------|----------------|
| A `advect_x` (dt/2) | 0.000040 | 0.000191 | 0.000555 | 0.001338 | 0.000667 |
| B `advect_vx` (dt/2) | 0.000076 | 0.000170 | 0.000384 | 0.000741 | 0.000424 |
| C `advect_vy` (dt) | 0.000105 | 0.000273 | 0.000678 | 0.001374 | 0.000755 |
| D moments | 0.000077 | 0.000135 | 0.000265 | 0.000511 | 0.000310 |
| H2D whole brick | 0.000156 | 0.000871 | 0.002708 | 0.006064 | 0.003170 |
| D2H whole brick | 0.000161 | 0.000887 | 0.002753 | 0.006157 | 0.003221 |

Effective bandwidths at `256³`, counting the payload each phase must move —
`(p+1)` brick passes for a gather, one for the reduction, ten for phase A's
pack / transform / multiply / transform / unpack chain — are **1003, 1087,
586 and 263 GB/s** for A, B, C and D, against an MI250X GCD's ~1.3 TB/s
peak. The bus runs at 22 GB/s each way, which is the number the next section
turns on.

### The two configurations, end to end

| shape | host step | A on host (hybrid) | A on device | hybrid | device |
|-------|-----------|--------------------|-------------|--------|--------|
| `64³` | 0.01585 s | 0.003345 s | 0.000521 s | 4.7× | **30.4×** |
| `128³` | 0.12418 s | 0.018548 s | 0.001324 s | 6.7× | **93.8×** |
| `192³` | 0.44751 s | 0.062115 s | 0.008615 s | 7.2× | 52.0× (see `N_x = 192`) |
| `256³` | 1.03634 s | 0.127003 s | 0.006616 s | 8.2× | **156.7×** |
| `128×256×256` | 0.52452 s | 0.070697 s | 0.003570 s | 7.4× | **146.9×** |

The middle column is what the issue's minimum scope produces: gathers and
reduction on the device, spectral shift on the host. It is worth 8× and no
more, and the split says exactly why. Seconds per Strang step at `256³`:

| item | A on host | A on device |
|------|-----------|-------------|
| A | 0.09828 (77%) | 0.00265 (40%) |
| B + C | 0.00217 | 0.00216 |
| D | 0.00109 | 0.00102 |
| coefficients (host offsets, device weights) | 0.00081 | 0.00064 |
| fields (host) | 0.00013 | 0.00013 |
| `v_y` halo | 0.00002 | 0.00002 |
| **H2D + D2H** | **0.02451 (19%)** | **0** |

Leaving one `O(N³)` phase on the host does not cost only its own time; it
costs its own time *plus* four whole-brick bus crossings per step, and
together those are 96% of the hybrid step. That is the measurement that
justified porting phase A, and it is a different measurement from the one
that justified porting B, C and D.

### Eight GCDs

Same binary, `--ntasks-per-node=8 --gpus-per-node=8`, global grid split on
`v_y`, so the host column is also eight-way parallel and the comparison is
at equal rank count.

| shape | per rank | host step | A on host | A on device |
|-------|----------|-----------|-----------|-------------|
| `128³` | 16 `v_y` cells | 0.02050 s | 0.004690 s (4.4×) | 0.001493 s (**13.7×**) |
| `256³` | 32 `v_y` cells | 0.14177 s | 0.022354 s (6.3×) | 0.002934 s (**48.3×**) |

The kernels keep their single-GCD speed, but the step does not scale with
them, and the split says where it goes. At `256³`, going from one GCD to
eight takes phase D from 1.02 ms to 0.50 ms per step — a factor of two, not
of eight — while the `v_y` halo goes the *wrong* way, from 0.02 ms to
1.05 ms. Those are the two places the step synchronises: an `MPI_Allreduce`
over the packed moment buffer and a `Sendrecv` pair on staged slabs. At
`128³` on eight GCDs each rank owns sixteen `v_y` cells and the halo alone
(0.73 ms) is half the step.

This is a property of the `v_y`-only decomposition, not a defect of the
port. One GCD does `256³` in 6.6 ms, so eight of them have under a
millisecond of arithmetic each and no amount of kernel tuning hides a
collective. A production run should give each GCD a *large* slab — which is
what the `N_vy / halo` rank cap pushes towards anyway.

## Parity

`vlasov_hip_parity` asks two separate questions.

**1. Operator parity.** One call of each device operator against one call of
the host operator on bitwise-identical input. A statement about the kernels.

**2. Integrated parity.** `n` Strang steps from the same initial condition,
comparing `f`, `rho`, `J`, all three field components and every ledger
column. A statement about error *growth*, which can fail while (1) passes,
because a Vlasov system amplifies round-off through its instabilities.

The case is Weibel-like — a bi-Maxwellian with `T_y/T_x = 9`, a seeded `B_z`
and a density perturbation — deliberately, because with zero fields every
velocity shift is exactly zero, the Lagrange weights collapse to a Kronecker
delta, and every gather in the test becomes a `memcpy` that would agree
bitwise no matter how wrong the kernel was.

### The tolerances and where they come from

Let `u = 2^-53 = 1.11e-16`.

| quantity | tolerance | derivation |
|----------|-----------|------------|
| gathers B, C | `p u max\|f\|` | `p` products accumulated in the same order with the same weights; the only arithmetic freedom left is whether each `w*f + acc` is fused, which is one rounding per stencil point |
| spectral shift A | `eps_fft max\|f\|`, `eps_fft = 8 log2(N_x) u sqrt(N_x)` | Higham's `c log2(N) u ||f||_2` for a radix-2 FFT, twice over for the forward/inverse pair, twice again for two independent implementations, with `||f||_2 <= sqrt(N) ||f||_inf` and `c = 4` |
| moment profiles, scalars | `eta = 2 N_v u kappa`, `N_v = N_vx N_vy`, `kappa = sum\|f\|/\|sum f\|` measured | worst-case error of a sequential sum is `(N-1) u sum\|f_i\|`; the device's partial-then-combine order has its own error of the same form, and the difference is bounded by the sum of the two |
| `f_min`, `f_max`, `f_face_max` | **0** | extrema do not depend on the reduction order, so these are required to be bitwise equal |
| `f` after `n` steps | `n (8 eta max(1, alpha_max) + 2 eps_fft)` relative | per step: the moments' error reaches `f` through the field and hence the shift (`8` for three shifts and two depositions), plus the two spectral shifts; worst case they add linearly |
| `rho`, `J`, fields, ledger | `n (4 eta + 2 eps_fft)` relative | the fields are a handful of linear operations on the deposited moments |

Two of these deserve their own paragraph.

**`kappa` is measured, not assumed.** It is `1.000000` for the test case,
because `f` is a distribution function and is non-negative to within the
interpolation's undershoot. A run where it is not is a run whose moments are
ill-conditioned, and the driver prints the number so a reader can see which
kind of run they are looking at.

**A signed sum must be judged against the magnitudes of its terms, not
against its own value.** `sum v_x f` over a distribution symmetric in `v_x`
cancels to zero to fifteen digits, so `J_x` is `~1e-18` while the terms
summed are `O(1)`. Dividing the difference by `1e-18` and calling the result
a relative error asks the two paths to agree to `1e-30` and tests nothing —
it was, in fact, the first version of this driver, and it failed on exactly
those two columns while everything it could actually measure passed. The
honest scale is `sum |v_x f| <= v_max sum f`, i.e. `v_max` times the density
profile, and that is what the driver uses; the same correction applies to the
ledger's total momentum, which is zero by construction in every benchmark
here.

### The gathers are *not* bitwise, and the reason is on the host side

The design aims at bitwise agreement for B and C: identical weights,
identical accumulation order, `#pragma clang fp contract(off)` in the
kernels. The measured bitwise fraction is 0.83 for B and 0.75 for C, with a
worst difference of exactly one ulp of `max|f|`.

The remaining ulp is the *host* compiler contracting `row[i] += w[m] * f(...)`
into an FMA — the LUMI `cpeGNU` build targets a machine that has one, and
GCC's default is `-ffp-contract=fast`. The host rounds once where the device
rounds twice. The driver therefore reports the bitwise fraction alongside the
tolerance rather than asserting bitwise and quietly relaxing the assertion
later, and the `p u` bound above is exactly one rounding per stencil point,
which is what that difference is.

Phase A is 0.90 bitwise against FFTW, also at one ulp, which is a stronger
agreement than the `eps_fft` bound requires; the bound stands because it is
the one that follows from the arithmetic rather than from this pair of
libraries on this machine.

### Results

Six configurations, all from job `21933310` (`standard-g`), all **PASS**,
and reproduced to the digit by job `21933340` on `dev-g`. `u = 1.11e-16`
throughout; `kappa` measured as `1.000000` in every case.

| configuration | ranks | `eta` | `eps_fft` | A bitwise | B bitwise | C bitwise | `f` after 20 steps | tolerance | used |
|---|---|---|---|---|---|---|---|---|---|
| `32³`, p=5, A on device | 1 | 2.27e-13 | 2.51e-14 | 0.899 | 0.831 | 0.753 | 2.25e-15 | 3.74e-11 | 6.0e-5 |
| `32³`, p=5, A on host | 1 | 2.27e-13 | 0 | — | 0.831 | 0.753 | 2.36e-15 | 3.64e-11 | 6.5e-5 |
| `32×32×64`, p=5 | 4 | 4.55e-13 | 2.51e-14 | 0.918 | 0.827 | 0.754 | 3.19e-15 | 7.38e-11 | 4.3e-5 |
| `32×32×128`, p=5 | 8 | 9.10e-13 | 2.51e-14 | 0.916 | 0.831 | 0.750 | 2.63e-15 | 1.47e-10 | 1.8e-5 |
| `64³`, p=7 | 1 | 9.10e-13 | 4.26e-14 | 0.715 | 0.821 | 0.752 | 4.64e-15 | 1.47e-10 | 3.2e-5 |
| `48×64×64`, p=3 | 1 | 9.10e-13 | 3.44e-14 | 0.430 | 0.812 | 0.752 | 5.37e-15 | 1.47e-10 | 3.7e-5 |

"used" is the measured divergence as a fraction of its tolerance: the worst
case uses 0.006% of the budget, so the bound is loose by four orders of
magnitude — which is what a worst-case round-off bound is supposed to be, and
which is why the table also prints the measured number rather than only the
verdict.

Every operator-level check passes with the same kind of margin. At `32³`,
one rank, phase A on the device:

| check | measured | tolerance |
|-------|----------|-----------|
| A `advect_x`, max abs | 1.78e-15 | 3.90e-13 |
| B `advect_vx`, max abs | 1.78e-15 | 8.63e-15 |
| C `advect_vy`, max abs | 1.78e-15 | 8.63e-15 |
| D `rho` profile | 2.00e-15 | 2.28e-13 |
| D `J_x`, `J_y` profiles | 4.0e-18, 6.9e-18 | 2.73e-13 |
| D particle number | 1.53e-13 | 2.86e-12 |
| D entropy | 6.61e-13 | 5.86e-12 |
| D `f_min`, `f_max`, `f_face_max` | **0** | **0** (required bitwise) |

`1.78e-15` is one ulp of `max|f| = 15.5` in all three transport rows, so the
gathers and the spectral shift each differ from the host by a single
rounding and nothing more.

Integrated over 20 steps, the fields track the distribution: `rho` agrees to
`4.3e-12` relative, `E_x` to `1.2e-14`, `B_z` to `2.5e-15`, and the Gauss
residual — the sharpest diagnostic the application has — reads `2.8718e-05`
on both paths, agreeing to `6.1e-13` absolute. The per-step trace shows the
divergence growing like `sqrt(n)` rather than exponentially over this window,
which is what a stable transport problem seeded at round-off should do; a
longer run through Weibel saturation would not, and this driver does not
claim otherwise.

**Non-power-of-two `N_x` weakens the spectral shift's agreement, as
expected.** At `N_x = 48` rocFFT and FFTW take different factorisations and
the bitwise fraction of phase A falls from 0.90 to 0.43 — while the worst
difference stays at one ulp and the tolerance is untouched. Worth knowing if
someone reads the bitwise column as a quality metric: it is a diagnostic of
*how* the two libraries differ, not of how much.

**One column of an earlier version of this driver was wrong, and it is worth
recording why.** `J_x` and `J_y` were compared against their own magnitude,
which for a `v_x`-symmetric distribution is `1e-18` of pure cancellation; the
test demanded agreement to `1e-30` and failed on job `21932623` while every
column it could actually measure passed. Judging a signed sum against the
magnitudes of its terms instead — `v_max` times the density profile — is the
fix, and it is a derivation, not a loosened constant.

## Limitations

- **`--device=hip` is the science path.** `vlasov_run` wires `DeviceStepper`
  behind that flag. The remaining device limitations are the ones below, not
  a missing switch.
- **One species is exercised.** `DeviceStepper` allocates a brick per species
  and loops over them, but every measurement and every parity run here is the
  default single electron species. A mobile-ion run is untested on the
  device.
- **`std::log` in the entropy column.** The device and host libm may differ
  by one ulp per cell. That is inside the factor of 2 in `eta`, but it means
  the entropy column can never be bitwise and is not asked to be.
- **Memory.** The device path holds two padded bricks per species (the
  gathers cannot be done in place), plus phase A's packed real buffer and its
  complex spectrum — together about `4.1 x N_x N_vx N_vy x 8` bytes per
  species: `550 MB` at `256³`, `4.4 GB` at `512³`, and `8.8 GB` for the
  issue's `1024 x 512 x 512` target, whose distribution alone is 2.1 GB.
  That fits one MI250X GCD's 64 GB several times over, so on this machine
  the `N_vy / halo` rank cap and not memory is what limits the application.
- **`N_x` must factor well.** hipFFT is happy with powers of two and with
  small-prime sizes; a large prime `N_x` will plan, but slowly. The host
  fallback (`--device-x=0`) exists and is measured.
- **No device path for the field solve, and no reason to add one.** Stated
  above with its number, so that "it is on the host" is a measurement rather
  than an omission.
- **Single-node only.** Parity was run at 1, 4 and 8 GCDs on one node; no
  multi-node device run was made. The `v_y` decomposition is one-dimensional
  and the exchange is a single-neighbour `Sendrecv`, so nothing about it
  changes off-node, but that is an argument and not a measurement.
