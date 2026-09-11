<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

## [Unreleased]

### Added

- **Allen–Cahn inverse homogenization** (`openpfc_inverse_homogenize`,
  issue #161 Stages 2–3). Explicit gradient flow on
  \(\tfrac12\lVert W\odot(C_H-C_{\mathrm{target}})\rVert_F^2\) with a
  volume-fraction penalty and a phase-field perimeter. Catch2 checks a
  spectral Laplacian oracle, that a uniform grey cell is driven toward
  the homogeneous target volume, and that \(J\) decreases from noisy
  initialization. Not Cahn–Hilliard (reserved for Stage 6) and not an
  external optimizer.
- **Periodic FFT homogenization and discrete \(C_H\)-sensitivity**
  (`apps/common/include/openpfc_apps/homogenization.hpp`, issue #161
  Stages 1 and 4). Six imposed-macroscopic-strain solves on the existing
  `EigenstrainMicroelasticity` Green operator assemble the engineering
  Voigt \(C_H\); the mutual-energy formula then gives
  \(\partial J/\partial h\) of
  \(\tfrac12\lVert W\odot(C_H-C_{\mathrm{target}})\rVert_F^2\). Catch2
  oracles: homogeneous cell, Postma/Backus laminate, cubic symmetry,
  closed-form homogeneous gradient, and a finite-difference check on
  random \(h\) (relative error \(<10^{-4}\)). CLI `openpfc_homogenize`.
  This is the **sixteenth application**, an explicit catalog exception
  for PDE-constrained inverse design — not a licence to add a
  seventeenth. Inverse loop, spinodal constraint, and LUMI 3-D design
  are not in this change.
- **FTA directional solidification and two-seed bicrystal** on
  `alloy_dendrite_growth` (leftover Stage 4 of issue #85; geometry from
  unmerged PR #103). Frozen-temperature field
  `theta = (G/ΔT_h)(x − x0 − V_p t)` with `evolve_theta` off and `M_c theta`
  live; crystal-frame cubic anisotropy `n' = R(θ_c)^T n`; second tanh seed
  with its own `crystal_angle`. Downstream tip and GB-groove diagnostics.
  Catalog stays at fifteen applications. Not a paper-scale melt-pool result.
- **Thermo-solutal-elastic dendritic solidification**
  (`apps/alloy_dendrite_elastic`, equations (5)-(7) of the issue #85 spec).
  The application's phase field, solute and temperature are now coupled to
  quasi-static elasticity through a composition- and temperature-dependent
  eigenstrain whose energy feeds back into the phase-field driving force:
  `elasticity.hpp` attaches `openpfc_apps/microelasticity.hpp` to the
  finite-difference stack and `material.hpp` carries Al-4.5 wt% Cu in SI with
  provenance and the two unit conversions (`eps_c` is `d eps*/dU`, not
  `d eps*/dc`) that are easiest to get wrong by orders of magnitude. The
  local physics runs on high-order FD with a halo exchange and the elliptic
  mechanical equilibrium runs spectrally on the *same* decomposition inside
  the same time step; the constructor refuses to run if the padded FD owned
  box and the HeFFTe real-space inbox disagree, because the two are copied
  index for index. The coupling is real, signed and monotone: at the
  conventional liquid-shear regularisation `mu_l/mu_s = 0.05` the tip slows
  16.7% and fattens 11.0% while `sigma*` moves 2.5%. A 200-fold scan of
  `mu_l/mu_s` (0.001 to 0.200) moves that slowdown from 3.26% to 28.6% and
  is still falling, so the inviscid limit is a few per cent, not a
  prediction about Al–Cu. With the temperature field on, the solutal and
  thermal eigenstrains oppose each other and together store a third of what
  the solutal misfit stores alone. Verified by a zero-coupling run
  reproducing the reference to ten digits, by doubling the stiffness
  reproducing `lambda_el = 2 lambda` exactly with `2.0000x` the energy, and
  by 1-vs-4-rank agreement at `3e-14` relative.
- **Field snapshots from the alloy dendrite** (`--fields-dir`,
  `--fields-every`): `phi`, `U`, `theta` and, with the coupling on, `f_el`,
  `df_el/dphi` and the hydrostatic and von Mises stress invariants, as raw
  Fortran-ordered bricks through `pfc::BinaryWriter` plus a JSON manifest.
  Correct at any rank count, unlike the single-piece `.vti` the other report
  figures use. `scripts/check_decomposition.py` compares two such directories
  and reports the relative max-norm and *where* it sits, which is what
  separates round-off from a subdomain-seam bug.
- **Tip radius measured in units of the tip radius**
  (`measure_tip_scan_relative`). The existing scan takes its fit half-widths
  in cells, which answers whether the grid resolves the fit and not whether
  the shape is a parabola. At `eps4 = 0.04` the cell-window spread is 49% at
  both `dx = 0.5` and `dx = 0.4` while the tip velocity converges to 0.7%: a
  number that does not move under refinement is not a discretisation error.
- **1D2V electromagnetic Vlasov–Maxwell** (`apps/vlasov_maxwell`, issue #84).
  A kinetic distribution on a 3-D phase-space grid coupled self-consistently
  to Maxwell, with a real Lorentz cross-product. One stepper; `--case`
  selects the validation rung. Linear rates are checked against a dispersion
  relation solved at run time. HIP path measured at 157× on one MI250X GCD
  at `256³`; science driver takes `--device=hip`.
- **HIP twin of the alloy-dendrite finite-difference step**, with CPU/GPU
  parity and a coupled-cost driver that times the host elastic solve against
  the device phase field. The 2-D device halo exchange now accepts
  `halo_width > 1` on an `nz == 1` slab.

### Fixed

- **`explicit_dt_limit` ignored the finite-difference order**
  (`apps/alloy_dendrite_elastic`). It returned the second-order von Neumann
  bound `dx^2/(2 d D)` for every stencil, which corresponds to a Nyquist
  eigenvalue of 4; the true eigenvalue of the order-`p` central second
  derivative rises with `p` (4, 5.33, 6.04, 6.42, 6.68, 6.87 for orders 2 to
  12, tending to `pi^2`), so at order 12 the guard authorised a step 1.7x
  larger than the scheme can take. No published result is affected -- every
  run used a safety factor of 0.2 -- but a user at 0.8 and order 12 would
  have diverged while the guard said the step was legal. The eigenvalue is
  now read out of the same coefficient table the stepper differentiates with.
- **A diverged dendrite run reported a plausible velocity.** Samples whose
  tip fit fails were skipped rather than counted, so the trailing-window fit
  fell back on the last healthy samples: `dx = 1.0 W0` blew up at `t = 1065`
  of a `t_end = 2000` run and reported `v_tip = 0.065`, `sigma* = 0.026`.
  Validity is now a statement about the final state as well as the fit, the
  summary CSV carries `n_samples`, `n_samples_failed`, `state_finite` and
  `valid`, and the driver says loudly that the numbers above it are not
  results.
- **Canosa 1973 was the wrong journal.** The tabulated Langmuir-root sanity
  check cited *J. Plasma Phys.* **8**, 187; the paper is *J. Comput. Phys.*
  **13**, 158–160, DOI 10.1016/0021-9991(73)90131-9. Header, test, and
  bibliography now match the primary source.

- **3-D eigenstrain microelasticity on the FFT stack**
  (`apps/common/include/openpfc_apps/microelasticity.hpp`). A reusable
  quasi-static elastic solver for phase-field applications whose transforming
  phase carries a lattice misfit: `div sigma = 0` with
  `sigma = C(phi):(eps - eps*)`, inverted by the Fourier Green operator of a
  homogeneous reference `C0` (Khachaturyan 1983), and iterated over the
  polarisation `tau = sigma - C0:eps` (Hu & Chen 2001) because the liquid is
  soft and the solid is stiff, which makes the problem inhomogeneous and the
  solve not one-shot. Cubic and isotropic `C`; the eigenstrain is one scalar
  amplitude field times a constant symmetric pattern, which covers the
  dilatational `h(phi) eps0 delta_ij` the alloy capstone needs. Returns the
  strain, the stress, `f_el`, and `d f_el/d phi` with **both** terms of the
  capstone's eq. (7) -- the transformation work `-sigma : d eps*/d phi` and
  the modulus contrast `(1/2)(eps-eps*):dC/dphi:(eps-eps*)`, the second being
  the one that is small, easy to drop, and wrong to drop, since it lives
  exactly where the stiffness varies. Host only, periodic only, small strain,
  no plasticity; a device path is separate work because the pointwise
  six-component contraction with a spatially varying stiffness has no
  `SpectralETDOps` kernel behind it the way `spectral_flux.hpp`'s elementwise
  work does. Two fixed points are provided and the accelerated one is the
  default, because the plain one cannot carry the application: a liquid
  supports no shear, so an honest solid/liquid stiffness ratio is 10 to 100,
  and the Hu & Chen Neumann series contracts at only `(r-1)/(r+1)`. The
  **Eyre-Milton scheme** (Eyre & Milton, *Eur. Phys. J. AP* **6**, 41 (1999))
  rewrites both the constitutive law and the equilibrium/compatibility
  conditions as reflections on `sigma +- C0:eps` and alternates them; the
  denominator of the local gain becomes `lambda + lambda_0` rather than
  `lambda_0`, the optimal reference becomes the geometric rather than the
  arithmetic mean, and the contraction becomes `(sqrt(r)-1)/(sqrt(r)+1)`. Its
  global half turns out to be exactly the Green application the basic scheme
  already performs (`y = z + 2 C0:W(z)`), so it costs one extra local 6x6
  solve per cell and not one extra transform -- which is why it was preferred
  to Moulinec & Suquet's augmented Lagrangian, which reaches the same rate but
  needs another field of state and a penalty parameter. Measured on a 32^3
  tanh sphere, cold start, `tol_el = 1e-6`, at ratio 1 / 2 / 4 / 10 / 100:
  basic **1 / 11 / 22 / 53 / 466** iterations at contraction — / 0.265 / 0.524
  / 0.767 / 0.970 against the predicted 0 / 0.333 / 0.600 / 0.818 / 0.980;
  Eyre-Milton **1 / 7 / 12 / 20 / 66** at — / 0.118 / 0.262 / 0.453 / 0.793
  against the predicted 0 / 0.172 / 0.333 / 0.519 / 0.818. Seven times fewer
  iterations at ratio 100, and both predictions bound their measurement from
  above, which is what makes the square root a fact about this code rather
  than a citation. The two schemes agree to 7e-13 in the strain, asserted, so
  the accelerated one is checked against the reference every CI run rather
  than trusted. `soft_liquid()` builds the liquid stiffness from the solid's
  by softening only the two shear channels and leaving the bulk modulus alone
  -- liquids really are nearly as incompressible as solids -- with a
  documented default `mu_l/mu_s = 0.05` costing 16 accelerated iterations
  against 44 basic, and 32 at the soft end of the literature range (0.01).
  That range is what sets `n_el_iter = 50` rather than the capstone spec's 20,
  raised deliberately and priced in the header rather than tuned quietly.
  A homogeneous modulus costs exactly one Green-operator application in both
  schemes, for two different reasons, and both are asserted.
  `apps/common/tests/test_microelasticity.cpp` (new ctest
  `apps-common-microelasticity`, ~5 s on one rank) is thirteen cases against
  closed forms, not baselines: the single-mode Green operator to 1e-14 (isotropic,
  against the analytic `3K/(lambda+2mu) k_m k_n/k^2`) and to 1e-13 (cubic,
  against an independently coded Gaussian-elimination solve of the acoustic
  tensor); the exact dilatation identity `tr(eps) = 3 alpha (a - <a>)` with
  `alpha = (1+nu)/(3(1-nu))` derived from the Eshelby tensor; Eshelby's
  spherical inclusion, whose interior strain and stress match the closed form
  to 3.3e-16 and 6.7e-16 relative because the periodic `<eps> = 0` condition
  turns them into an exact statement, with the genuinely shape-sensitive
  checks (interior uniformity, 3.0% at R = 8 dx and 0.74% at R = 16 dx, and
  the `r^-3` far-field decay, fitted exponent -2.913) reported separately;
  `div sigma = 0` at 1e-12 of the terms that must cancel; the elastic energy
  against `-(1/2) int sigma:eps*` in closed form to 4e-14 and against the
  Eshelby energy `V_i E eps*^2/(1-nu)` with its exact finite-cell factor
  `1 + alpha f/(1-alpha)`; and `d f_el/d phi` against a central difference of
  the *fully re-converged* energy, which also tests the Hellmann-Feynman claim
  that the partial derivative at frozen strain is the total one. One bug worth
  recording: `k_component` maps index `N/2` to `+k_Nyquist` on every axis, so
  conjugate-partner modes in a Nyquist plane were handed different `Gamma`
  directions, the resulting `eps_hat` was not Hermitian, and the inverse
  transform silently projected the difference away -- `|div sigma|` sat at
  3.4e-4 and the energy closed form at 5.5e-9 until the Nyquist component was
  zeroed per axis (`k_component_odd`'s rule), after which both dropped to
  round-off. Self-conjugate corner modes, where zeroing would make `k` vanish
  and the acoustic tensor singular, keep the raw direction; their coefficient
  is real, so it is safe. The trap is not specific to elasticity -- it catches
  *any* operator that is even under `k -> -k` but not under flipping one
  component alone -- so it is now written up at `k_component_odd` in
  `kernel/fft/kspace.hpp`, symptom first: a small, resolution-insensitive
  accuracy floor on a quantity that should be at round-off, with no error
  anywhere.
- **The solidification core of the multiphysics capstone** (`apps/alloy_dendrite_elastic`,
  issue #85): the quantitative dilute-alloy phase field of Echebarria, Folch,
  Karma and Plapp with Karma's anti-trapping current, coupled to solute and to
  temperature with latent heat -- equations (1)-(4) of the capstone model spec.
  High-order central FD (`pfc::gradient::FDGradient`, orders 2-14) on padded
  `pfc::comm::HaloExchange` fields, explicit four-stage step, MPI-decomposed on
  `pfc::Domain`/`Box3i`, 2-D and 3-D from one templated stepper. Elasticity
  (equations (5)-(7)) is deliberately absent; what is present is the field-based
  attachment point for it, with a ctest that proves it is wired.

  The point of the application is that its numbers are checkable, so the
  Stage-1 planar-interface verification is the product rather than a
  by-product. At `dx = 0.6 W0`, `k = 0.15`, `D_l = 2`, `lambda = 1`: the steady
  front velocity is within 0.19% of the thin-interface prediction
  `(Omega-1)/(k beta)`, the kinetic coefficient recovered as `-U_i/V` is within
  0.15% of `a1 (tau0/(lambda W0))(1 - a2 lambda W0^2/(tau0 D_l))`, the solute
  boundary layer is within 0.16% of `D_l/V`, the effective partition
  coefficient is `0.15018` against an input `k = 0.15`, and the steady-state
  mass balance `U_inf = k U_s - 1` closes to `-9.6e-5` absolute. Those five
  are independent relations, not restatements of one another, so an error in
  any single term of the model breaks at least one of them. A resolution study from
  `dx/W0 = 1.6` down to `0.4` walks the velocity error from -58.5% to -0.02%
  and `k_eff` from -27.8% to +0.16%; the collapse is faster than fourth order
  because what is being resolved is the `tanh` profile, not the stencil.

  `k_eff` is the measurement that says whether the anti-trapping current is
  right, and it does: with the current on it is flat in velocity to under 2%
  over an eight-fold range (`+0.02%` at `V = 0.05`, `+1.7%` at `V = 0.39`, the
  residual tracking the interface Peclet number as the asymptotics says it
  should); with the current switched off it climbs from `+7.8%` to `+65%`, and
  with its sign flipped from `+16%` to `+288%`. It is measured by extrapolating
  the outer solute profile back to the interface rather than by sampling the
  first liquid cell, because the pointwise version carries a resolution
  artefact the same size as the effect.

  Total solute `sum[P(phi)/(1-k) + P(phi) U]` and the latent-heat balance
  `sum theta - sum phi / 2` are **exact discrete identities** rather than
  approximations -- collocated central differencing of a flux telescopes on a
  periodic grid, `P` is affine in `phi`, and the same `d_t phi` array drives
  the phase field and the solute source -- so the measured drift is round-off
  (`1e-15` to `7e-13` over up to 500 000 steps, growing with cell count as a
  floating-point sum should) and a drift above that is a broken term rather
  than a loose tolerance. Getting that required evolving `P(phi) U` as the
  primary variable and recovering `U` by division, which is why the code does
  not evolve `U` directly.

  Also shipped: a deterministic 2-D dendrite (three minutes on a login-node
  core) with tip position, tip velocity and tip radius written to append-only
  CSV, all three defined operationally in `diagnostics.hpp` next to the code
  that computes them; a 3-D path exercised by a dimensional-consistency test
  that requires a z-invariant `Stepper<3>` run to reproduce `Stepper<2>` to
  round-off; and CPU-only backends, because a HIP twin of a four-kernel,
  three-exchange, fourteen-field step that cannot be run from a login node
  would look verified without being it.

  Two errors in the capstone model spec were found by implementing it and are
  documented in `step.hpp`. Equation (3) pairs the conservative left-hand side
  `d_t[P U]` with the source `(1/2)(1 + (1-k) U) d_t phi` that belongs to the
  non-conservative form `P d_t U`; the mixture is neither, and running it
  (`--spec-source=1`) breaks solute conservation -- 10% drift at `V = 0.1`,
  and the `V = 0.4` run diverged -- and biases `k_eff` by 15-34%. Equation
  (1)'s un-normalised `a_s = 1 + eps4 sum n_i^4` gives an effective anisotropy `(eps4/4)/(1 + 0.75 eps4)` in 2-D, about four times
  smaller than the input number, so the Karma-Rappel `eps4` values do not
  transfer. The anti-trapping sign in the spec is correct, and
  `parameters.hpp` records the cancellation that fixes it.

- **Where spectral beats finite difference, and where it does not**
  (`heat3d_spectral_content_study`, `apps/heat3d`). The scalability chapter
  had cost per step and parallel scaling measured for both spatial operators
  but accuracy measured only for a single Fourier mode -- the most favourable
  possible case for a high-order stencil -- so it declined to give a
  recommendation. That is now closed. Because the heat equation is linear on
  a periodic box, the L2 error of *any* initial field is a closed-form sum
  over that field's own spectrum, so accuracy is mapped against one
  parameter: `f`, the fraction of Nyquist at which the initial amplitude
  spectrum is down to 1e-3 of its peak. Under an N^3 cost model at a fixed
  step count, FD order p is the cheaper route to a given accuracy iff the
  coarsest grid it can use keeps the content above `cbrt(c_fd/c_spectral)` of
  Nyquist -- pure cost arithmetic, 0.32 (FD-2) to 0.49 (FD-12) with the
  measured costs, because a 32x cheaper step buys only 32^(1/3) = 3.2x in
  grid spacing. The crossover in accuracy is at L2 ~ 6e-7, owned by FD-12:
  looser than that, finite differences win outright (FD-2 costs a twentieth
  of the spectral path at 1e-2); tighter, no shipped stencil order is the
  cheaper way there. Broadband fields are past every threshold regardless of
  tolerance -- at the dealias-safe tungsten spacing the crystal's third
  harmonic sits exactly on Nyquist. Ten points of the map were re-measured by
  running the shipped FD stack (padded `Field` + `HaloExchange` +
  `FDGradient<HeatGrads>`) under RK4 against the exact solution: prediction
  and measurement agree to between eight and ten significant figures, worst
  case 3.7e-9 relative. Implementation note: the stencil's dispersion defect
  cannot be computed by subtraction (order 12 at theta=0.1 gives 1.2e-19 from
  two terms of size 1e-2), so it is computed as the tail of the identity that
  the order-2M central second difference is exactly
  `(2 asin(delta/2))^2 = sum a_m delta^(2m)` truncated at m = M -- an
  all-positive series with no cancellation. `test_heat3d_spectral_content.cpp`
  pins that identity against the shipped `EvenCentralD2` tables, the map's
  grid-independence, and the crossover predicate.

- Scaling to 16 nodes, and a spectral-versus-finite-difference comparison, in
  the scalability chapter. Weak scaling of `tungsten_hip` at exactly 67.1 M
  cells per GCD from 1 to 16 nodes, ending on 8.59 billion cells at 49.6%
  efficiency -- the curve is flat inside a node (95.6% at 2), pays ~30 points
  crossing onto the fabric, then loses only 15 more over a further 4x in
  ranks. Strong scaling at 1280^3 reaches 82% at 16 nodes, against a 768^3
  curve that fell below 50% by four; the earlier curves were grid-limited, not
  code-limited. A 12-node point sits at 38% because 1280/96 = 13.33 and
  OpenPFC decomposes into z-slabs -- a 10-node run (1280/80 = 16 exactly) was
  added to test that and lands back on the curve at 84.1%, so the rule is to
  pick a rank count that divides the slab axis. 2048^3 is recorded as needing
  at least 16 nodes: it was OOM-killed host-side at 32, 64 and 96 GCDs.
  For heat3d, the only app with both solvers on one PDE: at equal grid the
  spectral step costs 32x a second-order FD step and 8.7x a twelfth-order one,
  yet all three strong-scale to 80-85% at 16 nodes, so parallel efficiency is
  not a reason to prefer either in this range. At the time this landed the
  chapter deliberately declined to turn cost, scaling and accuracy into a
  single verdict, because the convergence study evolved one smooth Fourier
  mode and that ranking does not transfer to fields with content near the
  grid scale; the accuracy axis has since been measured properly (see the
  spectral-content entry above) and the chapter now gives the
  recommendation.

- Field figures for the two application chapters that had none, rendered from
  real single-rank runs of the shipped science presets: `07_surface_diffusion`
  gets the isotropic-against-anisotropic nanosurface anneal (the same crossed
  corrugation after the same anneal, one keeping its egg-crate and the other
  reduced to stripes — `energy_kx_frac`/`energy_ky_frac` = 0.249/0.751 drawn as
  a surface), and `09_ehd_film` gets the compliant-against-stiff plate at the
  instant the load lifts (the deepest either dent gets: `h_centre` 0.659 with
  radius 12.3 against 0.747 with radius 14.2, on one shared scale). Both runs
  are in `docs/report/figures/run_field_demos.sh` and neither extends the
  shipped preset's `t1`.

  Getting them needed a real change to the two applications, not just to the
  report: `surface_diffusion_anisotropic` and `ehd_film_nonlinear` are the two
  science drivers that own their `main()` rather than running on
  `SpectralETDSession`, and neither had ever had a field writer — their science
  presets could only be read through the diagnostics CSV. Both now honour the
  session's own `fields[]` spelling through the new
  `apps/common/include/openpfc_apps/field_snapshots.hpp`, and the four presets
  `nanosurface_isotropic.json`, `nanosurface_anisotropic.json`,
  `load_relaxation_compliant.json` and `load_relaxation_stiff.json` set it.
  Omitting the key keeps the previous CSV-only behaviour, which is what the
  test presets rely on; a malformed one is rejected rather than silently
  ignored, since a preset that believes it is writing output and is not stays
  invisible until a figure is missing. Covered by `ctest -R field_snapshots`.
- Field figures for the four application chapters that had none:
  `08_kawahara`, `13_wave2d`, `14_allen_cahn`, `15_kobayashi`, each rendered
  from a real single-rank run recorded in
  `docs/report/figures/run_field_demos.sh`. Kawahara is one-dimensional, so
  `field_plots.py` grows a line-plot pair (`render_line_panel`,
  `render_line_comparison`) alongside the existing image renderers: a
  `512 x 1 x 1` snapshot drawn as an image is a one-pixel stripe, and the
  thing that figure shows -- a wave train shed at a few percent of the pulse
  height -- is an amplitude, which a linear axis reads and a colour bar does
  not. Lines use the Okabe-Ito palette paired with distinct dash patterns, so
  they survive colour-blind vision and a greyscale print.
  `field_io.read_gray_png` inverts `pfc::io::write_mpi_scalar_field_png_xy`'s
  fixed affine map, which is the only field output `apps/allen_cahn` and
  `apps/kobayashi` produce -- checkably so: the superlevel-set areas recovered
  from the Allen-Cahn PNGs are the same integers the program prints for its
  own exit-code criterion.

- Three discrepancies between what an application chapter claimed and what
  its binary actually does, found while rendering those figures and now
  recorded in the chapters and in `docs/report/README.md` rather than left
  implicit. `wave2d`'s advertised observable `global_rms_u_interior` is
  always exactly zero, because the interior reduction skips `half_width`
  cells on every axis including `z` and the application is an `nz == 1`
  slab, so the loop covers no cells; `wave2d_fd`'s advertised even FD orders
  2..20 are in practice order 2 only, since a wider halo cannot fit in that
  same `nz == 1` slab and orders 4 and up abort before the first step; and
  every `.vti` the JSON session pipeline writes carries `Spacing="1 1 1"`
  regardless of the run's `domain.dx`, because `pfc::apply_writer_domain`
  never calls `VTKWriter::set_spacing`. None of these is fixed here -- the
  fields themselves are correct in all three cases -- but a reader of the
  report should not have to rediscover them.
- Field figures for the three crystal-selection chapters of the applications
  report, each rendered from a real single-rank run added to
  `docs/report/figures/run_field_demos.sh`. `10_higher_order_pfc`: the two
  noise-seeded presets at `t=400`, side by side and cropped to the central
  `64^2` cells, where the six-neighbour triangular packing of the single-mode
  `k^4` kernel and the four-neighbour square packing of the two-mode `k^8`
  one can be counted directly -- the chapter's central claim as a real-space
  picture rather than the ring-power inference it warns about.
  `11_gradient_elasticity`: hydrostatic stress around the `circular_inclusion`
  disk at `ell=8` against the same problem at `ell=0`, showing the
  oscillatory boundary layer of width `~ell` that raises the peak
  `|sigma_h|` from 0.0147 to 0.0213. `04_aluminum`: an FCC seed in an
  isothermal melt at two radii, one above and one below the critical size.

- `apps/aluminumNew/inputs_json/fcc_seed_nucleus.json`: a `192^3` single-FCC-seed
  aluminium preset that runs on one core in a few minutes, added because the
  only other shipped inputs are a half-billion-cell demonstration case and a
  `16^3` *constant* field with no seed in it -- neither of which can be shown.
  Sweeping the seed radius over 30/40/50/60 to `t=1000` measures a **critical
  radius**: every seed shrinks at first, because `SeedFCC` writes a diffuse
  profile (peak `psi` 3.16) that must relax onto the model's own solid
  amplitude (peak `psi` near 5), and the radius spent paying for that
  relaxation decides the outcome. Radius 60 bottoms out at 52 reduced units
  around `t=500` and then grows; radius 50 dissolves by `t~700`; radius 30 is
  gone by `t=200`. At `T_const = 980`, `n0 = -0.006` the critical radius is
  therefore between 50 and 60 reduced units, about five FCC lattice constants.
  The isothermal aluminium case is a nucleation experiment, *not* the steady
  "seeded growth" run the chapter and the app README described; both now say
  so.

- `apps/aluminumNew/tests/test_aluminum_inputs.cpp` (ctest
  `aluminum-shipped-inputs`), the aluminium twin of the tungsten shipped-input
  guard: every file in `inputs_json/` is parsed through the aluminium schema
  and every initial condition constructed through the catalog the binaries
  use, so a renamed or dropped key fails a test instead of an unlucky reader's
  run.

- `tungsten_dealias_study` and `tungsten/resolution.hpp`: what running the
  cubic PFC nonlinearity undealiased actually costs. The crystal sits at
  `k0 = 1` and carries real content at `2k0` and `3k0`, so a grid must both
  fit the third harmonic under Nyquist and keep the 2/3 cut above the second.
  Both conditions reduce to `dx <= pi/3` (six points per lattice period), and
  every shipped preset uses `dx = 1.1107`, 6.1% coarser -- failing both at
  once, so the mask cannot simply be switched on. Measured over the same
  seeded solidification run twice at each spacing: the mask changes total
  spectral power by 1.70% at the shipped resolution and by 0.009% at eight
  points per period, a factor of 190, which is the signature of a
  discretisation artefact rather than a modelling choice. The selected
  wavenumber -- the lattice constant, and the observable these runs are
  normally read for -- moves by at most 5e-4 at any resolution. Reported in
  the tungsten chapter with a figure; `[resolution]` pins the grid criteria in
  a test that runs instantly, since the study itself takes minutes and CI does
  not call it.

- `heat3d_fd_convergence_study` (`apps/heat3d`): measures the *observed*
  order of accuracy of `pfc::gradient::FDGradient<HeatGrads>`'s central FD
  stencils (orders 2–20) against their *design* order, on a periodic
  single-Fourier-mode problem swept over `N` in {16, 24, 32, 48, 64}. Every
  design order 2–12 was reproduced to within a few tenths of a unit
  (measured table in `apps/heat3d/README.md`); fd_order 12 hits the
  double-precision round-off floor at N=64, as expected for a 12th-order
  stencil on a smooth solution. Isolates spatial from temporal error by
  evolving the semi-discrete ODE with its own *exact* eigenvalue rather
  than a time-stepping loop (zero time-discretization error by
  construction) — two dt-based explicit-Euler designs were tried first and
  both reproduced the "all curves flatten at one floor" failure mode this
  kind of study is warned about; see `apps/heat3d/include/heat3d/convergence_study.hpp`.
  Also documents a boundary-shell gap in the `FDCPUStack` +
  `pfc::sim::steppers::create` pattern (`pfc::sim::for_each_interior` skips
  a `fd_order/2`-wide shell that only `FDCPUStack::du<G>()` covers), found
  while building this study. Data: `docs/report/data/heat3d_fd_order_convergence.csv`.
  Figure: `docs/report/figures/heat3d_fd_order_convergence.svg`. Regression
  guard: `test_heat3d_fd_convergence.cpp` / ctest `heat3d-fd-convergence`.

### Fixed

- The applications report renders. `docs/report/14_allen_cahn.qmd` contained
  `$R^\*$`; a backslash-escaped asterisk is invalid in math mode and LuaTeX
  stops on it with `Missing { inserted`. MathJax accepts it silently, so
  neither the HTML render nor the markdown checkers noticed -- only a PDF
  render finds this, which is why the same mistake had already reached the
  repository once before in the thin-film chapter. Both formats now build:
  19 HTML pages and a 111-page PDF, with no unresolved cross-references in
  either.

- The shipped `allen_cahn` preset demonstrated Allen-Cahn arithmetic rather
  than Allen-Cahn physics, and the two halves of that could not be fixed
  separately. Its interface was `eps*sqrt(2M) = 0.76` cells wide -- sub-grid,
  so the front was pinned by the lattice -- and its driving force sat 6%
  under the bistability ceiling `F eps^2 < 2/(3 sqrt 3)`, close enough that
  `driving_force = 20` or `epsilon = 0.3` flips the whole box. It measured
  `+9%` against the sharp-interface law, which looked like agreement and was
  two errors of opposite sign cancelling: the lattice slowing the front by
  ~12% and the finite tilt (the law is the `F eps^2 -> 0` limit) speeding the
  continuum front by ~26%. The obvious repair -- back the driving force off
  and leave `eps` alone -- makes it worse, not better: at 0.76 cells the
  residual walks from `+17%` at the ceiling to `-65%` at a quarter of it, so
  there is no safe driving force at that resolution. Measured across
  interface width, margin and grid in
  `docs/report/data/allen_cahn_resolution_margin.csv`.

  The preset is now `epsilon = 0.75`, `driving_force = 0.25`, `M = 8.0`
  unchanged, `dt = 0.005`, `256^2`: a 3.0-cell interface and `F eps^2 = 0.141`
  a factor 2.7 under the ceiling. `M` is kept and the other two are round
  quarters, so the preset stays recognisable. The grid had to move with them
  -- a resolved interface at a safe driving force has a critical nucleus
  `R* = M/v` of 7.1 cells against 0.70 before, and the seed
  `sigma = 0.055*min(nx,ny)` is only 4.1 cells at `64^2`, so the old default
  grid now dissolves its own seed rather than growing it.

  The pass criterion moved with the preset, because a slower front makes the
  curvature term matter: a disc obeys `dR/dt = (3/2) F eps sqrt(2M) - M/R`,
  and `M/R` is 20% of the answer here and does *not* shrink with the grid
  (the seed scales with the box, so `R/R*` is nearly grid-independent).
  Comparing a disc against a flat-front law left a 20% bias that looked like
  a grid dependence; with the curvature term the residual is 1.6% at the
  shipped preset and at most 2.3% from `128^2` to `512^2`, so the accepted
  band tightens from a factor of two to +/-25%. The app also prints `R*`, the
  distance to the bistability ceiling as a percentage, and warns below a
  2-cell interface, above half the ceiling, and on a subcritical seed.

  No golden moved: `allen-cahn-cpu-golden` and the CPU-vs-CUDA/HIP parity
  tests all set every parameter explicitly rather than reading the defaults,
  and the seed width at their `32^2` is on the `max(2, ...)` floor either
  way. What did move is `allen-cahn-interface-kinetics`, which now runs
  `256^2`/`384^2` instead of `64^2`/`128^2` and compares each grid with its
  own curvature-corrected prediction rather than comparing the two raw
  speeds -- the old pair agreed to 0.3% by luck, and extending the same
  comparison to `512^2` spreads them by 10%. Two new cases pin the preset
  itself (interface width, margin, supercritical seed, dt headroom) and the
  coupling that forced the joint fix.

- `AluminumPhysics::from_json` skipped its own schema. A guard
  `if (!params_json.is_null() && !params_json.empty())` meant `"params": {}`
  -- which is exactly what `apps/aluminumNew/inputs_json/smoke.json` shipped
  -- silently took the C++ member initialisers while all 25 schema fields
  were declared `required`. For a calibrated material model that is wrong in
  both directions: a run's parameters were not recoverable from its input
  file, and the struct defaults are an uncited second copy of coefficients
  this repository cites no source for. The parse is unconditional now; an
  empty or absent `model.params` reports every missing field at once, and
  `smoke.json` spells all of them out.

  Nine of those 25 fields -- not the four the README named -- were read
  nowhere. `T_min` and `T_max` are now implemented: they clamp
  `T_const + T_var`, which the moving-frame profile
  `T_const + G(x - x0 - Vt)` otherwise leaves unbounded along a 1393-unit
  domain, and an inverted window or a `T_const` outside it is rejected at
  load time instead of silently pinning the domain at one end. Every shipped
  preset is isothermal (`G_grid = V_grid = 0`), so the clamp is a no-op on
  all of them and the pinned `aluminum-etd-cpu-golden` checksum is unchanged.
  The other seven named nothing this model computes and were removed from the
  schema: `alpha_farTol` and `alpha_highOrd` parametrise *tungsten's* `C2`,
  not the FCC dual-Gaussian peak; `shift_u` and `shift_s` are the vapour
  shift that produced the `p*_bar`/`q*_bar` coefficients, already applied
  offline; and `n0`, `n_sol`, `n_vap` are duplicates of values that act in
  the `initial_conditions` and `boundary_conditions` blocks, where a second
  copy that nothing reads can disagree with the one that does. Unknown keys
  are ignored, so inputs carrying the removed seven still load. Guards:
  `[schema]` and `[temperature]` in `aluminumTest`.
- The report's abstract, introduction and conclusions describe the report that
  now exists. They still claimed scaling "out to 32 GCDs across four nodes",
  listed "no weak-scaling study" as a gap *and* as the top suggested next
  measurement, and generalised a one-node result -- halo exchange scaling
  better than a distributed transpose, 78% against 59% -- that the 16-node
  data contradicts: at 1536^3 across 4 to 16 nodes the spectral path, FD-2 and
  FD-8 land within five points of each other. Front and back matter now match
  the body, the closed gaps are removed, and the gaps that remain are the real
  ones: no CPU control, accuracy measured only for a single smooth mode, and a
  memory boundary characterised but not diagnosed.

- `wave2d`'s advertised observable, `global_rms_u_interior`, was identically
  zero for every configuration on both CPU drivers. The interior visitor
  trimmed the stencil half-width off each axis of each rank's owned box, and
  the app is an `nz == 1` slab: `[hw, 1 - hw)` in z is empty, so the reduction
  summed no cells and printed the `0` that an empty sum and a genuinely zero
  field share. Nothing caught it because nothing distinguished the two. The
  observable now trims only axes that can spare the shell -- the slab's single
  z layer is kept, because z is not a dimension this model has -- and trims in
  *global* index space rather than per rank, so the number no longer shrinks
  as ranks are added and eat a shell at every subdomain seam. `report()` also
  carries the visited count: the run line prints `interior_cells=`, an empty
  interior prints `global_rms_u_interior=undefined interior_cells=0` and exits
  non-zero, and the reduction can no longer report nothing as if it were zero.
  `wave2d_fd` at order 2 and `wave2d_fd_manual` now agree to the last digit
  (0.219101 for the 64x64x50 preset), which is the cross-check the observable
  was supposed to provide all along. Guards: `[reporting]` in `test_wave2d`
  pins the degenerate-axis rule and sums a 2- and a 4-rank decomposition to
  the 1-rank answer.

- `wave2d_fd` advertised even FD orders 2 to 20 and only order 2 ran. Order 4
  aborted before its first step with `create_padded_face_types_6: owned
  extents 64x64x1 cannot host halo_width=2 owned send slabs`: a 1-thick owned
  z extent cannot provide a 2-thick send slab. It was never asked to -- the
  Laplacian reads x and y only -- but the face-type builder validated all
  three axes whatever direction set the exchanger had been given, contradicting
  its own documented contract that "orthogonal thin axes (e.g. `nz == 1` for
  Axes2D) remain valid". `create_padded_face_types_6` now takes the active-slot
  mask, builds and validates only the slots that will carry a message, and
  `HostFacesHalo` resolves that mask before asking for types. `wave2d_fd` asks
  for `halo::presets::Axes2D()`, which is what the preset was written for, and
  all ten advertised orders run. Rejecting orders 4-20 at parse time was the
  alternative; making them work was preferable because the y-ghost fill was
  already general in the halo width, so the stencils were the only thing
  standing between the app and the accuracy its usage line promised. Guard:
  `[halo]` in `test_wave2d` constructs the exchanger at every advertised order
  and steps a 32x32 slab at order 4, and still requires that asking for the
  z faces on a 1-thick z is an error.

- Every `.vti` written through a JSON session claimed `Spacing="1 1 1"`.
  `pfc::apply_writer_domain` set the index geometry -- global size, local size,
  offset -- and dropped the physical geometry the `Domain` was carrying right
  next to it, even though `VTKWriter::set_spacing` / `set_origin` existed and
  every hand-written call site already used them. Harmless for the `dx = 1`
  presets and wrong by 4x for a `dx = 0.25` run, in a way nothing in the file
  reveals: ParaView simply renders at the wrong scale and every length measured
  off it inherits the error. `ResultsWriter` gains a `set_geometry(origin,
  spacing)` hook, defaulting to a no-op so index-only sinks such as
  `BinaryWriter` need not implement it, `VTKWriter` overrides it, and
  `apply_writer_domain` forwards what the `Domain` already knows. No committed
  test or golden file depended on the old value -- the two existing spacing
  assertions in `test_vtk_writer` set the spacing themselves. Guard: two
  `[geometry]` cases write a real `.vti` through `apply_writer_domain` at
  `dx = 0.25` and `dx = 0.5` and read the header back.

- `allen_cahn`'s pass criterion measured the grid rather than the physics, and
  doubled as the process exit status. It required the superlevel area to reach
  5x its initial value, but the seed radius scales with the grid
  (`sigma = 0.055*min(nx,ny)`) while the interface speed does not, so
  `((R0 + v t)/R0)^2` shrinks as the box grows: the same physics scored 6.08x
  at 64^2, 3.50x at 128^2 and 2.55x at 256^2 -- pass, fail, fail. The criterion
  is now the interface velocity `dR/dt`, with `R = sqrt(A/pi)` the equivalent
  radius of the same superlevel area, measured over the *second half* of the
  run and compared with the sharp-interface solvability result
  `v = (3/2) F eps sqrt(2M)`. The second half because the Gaussian initial
  condition is far from the equilibrium `tanh` and its collapse moves the
  contour by a distance that scales with the seed -- a whole-run average
  inherits exactly the grid dependence the area ratio had. Measured that way
  the three grids agree to 5% (12.4 / 12.8 / 13.1 against a prediction of
  11.4). The accepted band is a factor of two either way rather than a
  percentage, because the shipped preset's interface is `eps*sqrt(2M) = 0.76`
  cells wide -- under one grid spacing, so the front is lattice-limited and
  sits tens of percent off the continuum law (`+11%` at the defaults, `-35%`
  at `driving_force = 5`); the app now prints that width and warns. It also
  prints the bistability limit `F eps^2 < 2/(3 sqrt 3)`, which the shipped
  parameters clear by 6% and which `driving_force = 20` or `epsilon = 0.3`
  crosses, flipping the whole domain instead of growing a grain -- previously
  reported as an area ratio of 75.85 and a pass. The verdict prints as
  `physics_check=PASS|FAIL|SKIPPED` and no longer sets the exit status: a run
  too short to have outrun the transient, or one whose predicted advance is
  under a cell, is skipped rather than failed, and the exit status answers
  "did the run finish?". `--strict` opts back in to gating on a `FAIL`. Guard:
  the new `allen-cahn-interface-kinetics` ctest -- the app had none -- runs the
  shipped update at 64^2 and 128^2 and requires the same verdict and speeds
  within 10%.

- CI no longer fails on a third-party apt repository it does not use. The
  GitHub runner images carry apt sources for Google Chrome and Microsoft
  prod; on 2026-09-09 Chrome served a `Packages.gz` whose hash disagreed with
  its own `Release` file, `apt-get update` exited 100, and every workflow that
  installs a package died on its first line. Because the build matrix is gated
  on the Code Quality job, healthy branches reported a red pipeline. All
  runner-image installs now go through `scripts/ci/apt_update.sh`, which drops
  the unused sources before refreshing the index, so only an outage of the
  Ubuntu archives we actually install from can fail the step.
- A spectral application on a one-dimensional grid computed its nonlinear term
  from a destroyed field, on FFT backends that use their input as scratch.
  `SpectralETDSystem::attempt()` transforms `psi` and then evaluates the
  pointwise nonlinearity from that same `psi`; `FFT_Impl::forward` takes its
  input by `const&` and every caller relies on that. HeFFTe 2.4.1 declares the
  input `input_type const input[]` and then writes to it anyway on some FFTW
  builds. Measured on LUMI with two builds differing *only* in the FFTW
  library -- same source, same compiler, same HeFFTe version, same buffer
  address -- a forward of a 512-point line left the input untouched against
  Cray FFTW 3.3.10.10 and destroyed it against vanilla FFTW 3.3.10
  (`input[0]` 0.04 to 0.32). The transform *output* is correct to 4e-15 in
  both cases, which is why this went unnoticed: nothing looks wrong until a
  second stage reads the input back, and the platform this project develops on
  never did. Probing HeFFTe directly, the damage is confined to 1-D grids
  (512x1x1, 256x1x1 and 64x1x1 destroyed; 32x32x1, 16^3, 32^3 and 64^3
  preserved) -- which is why only `kawahara` showed it and the 3-D golden
  checksums stayed green on the very platform that breaks the 1-D ones. The host `forward` now hands the backend a copy. Only `forward`
  is guarded -- `backward` was measured on both builds and leaves its input
  alone, and the device path has not been shown to have the problem;
  `tests/unit/kernel/fft/test_fft_input_preserved.cpp` asserts the contract in
  both directions so a backend that starts breaking it fails there. No
  measurable cost: 40000 spectral steps time the same either side of the
  change, within run-to-run noise.
- `tungsten_moving_bc_options.json` can be run at all. It declared
  `"initial_position": "end"`, a key nothing in the code reads, where
  `MovingBC` requires a numeric `xpos`; the shipped input aborted on the first
  line of boundary-condition wiring with `missing or invalid 'xpos' field`.
  It now carries `xpos = 139.17225402106772`, the end of the domain less the
  boundary width, the same convention `tungsten_moving_bc.json` uses. Nothing
  had caught it because no test ever loaded a shipped input -- they all build
  their JSON inline -- so `tungsten-shipped-inputs` now walks every file in
  `apps/tungsten/inputs_json/` and constructs what it declares through the
  same schema and catalog the application uses. The other apps' inputs are
  still unguarded; the pattern is worth copying.
- The finite-difference convergence study no longer breaks the REUSE lint. It
  wrote an SPDX header into the CSV it generates, and the linter read those
  string literals as a second, malformed license declaration for the source
  file itself. Everything under `docs/report/` is already covered by the
  `REUSE.toml` annotation and the sibling data CSVs carry a plain provenance
  comment, so the header was redundant; the generator and the committed CSV
  now carry that comment instead. Regenerating the CSV reproduces the
  committed data byte for byte.

- `ctest` can be registered in a `--no-heffte` CPU build again (`#105`). The
  Catch2 `-isystem` loop in `tests/CMakeLists.txt` guarded on the unevaluated
  string, so `$<INSTALL_INTERFACE:include>` passed the check and expanded to
  nothing, emitting a bare `-isystem` that swallowed `-MD`. Install-interface
  entries are now skipped. `heat3d`'s spectral Catch2 suite is gated on HeFFTe;
  its FD operator-evaluation and rhs-pattern suites stay available.

### Changed

- `docs/report/12_heat3d.qmd` reports the measured orders of accuracy from
  `heat3d_fd_convergence_study` (2.00, 3.98, 5.96, 7.95, 9.94, 11.84 against
  design 2 to 12) in place of the monotonicity check it used to describe, and
  embeds `heat3d_fd_order_convergence.svg` -- which was committed with the
  study but never referenced from any chapter, so nothing displayed it.

- Every application chapter in `docs/report/` and the app READMEs that lacked
  one now carry the physical-experiment contract (`#112`). The report was
  strong on operators and weak on experiments: a reader could see `L(k)`
  without being told what object was being simulated, on what domain, from
  what initial state, or how mature the model was. Each chapter now opens with
  a physical question and a *named* observable, then a `Problem setup` table
  (use case, domain, grid, boundary conditions, initial condition, key
  physical parameters, observable, maturity), then the equations. Model
  maturity is reported on three independent axes — numerical verification,
  physical completeness, calibration — because collapsing them into one word
  hides the common case of an exactly verified model with invented parameters.
  Where an application ships both, the verification preset is named and
  explicitly not presented as a production science case. Periodicity is
  explained per chapter rather than boilerplate: what it means for a spinodal
  alloy is not what it means for a dendrite in a box with no heat sink.
  Existing provenance notes (Andersson–Sundman Fe–Cr, Wu–Adland–Karma
  two-mode PFC, the surface-stiffness form, the capillary–gravity mapping) are
  folded into the calibration axis rather than restated or softened.
- The same pass corrected report claims that were untrue of `master`. The most
  consequential: `heat3d_spectral` and `heat3d_spectral_pointwise` had their
  descriptions transposed (the former is the implicit-Euler driver, the latter
  the explicit one); `tungsten` and `aluminumNew` were described as dealiased
  when `SpectralETDOptions::dealias` defaults to `false` and neither session
  enables it; `aluminumNew`'s temperature profile was given with a spurious
  `T0` offset and a `T_min`/`T_max` clamp that no code performs; `thin_film`
  listed a two-mode `k^4` ratio test that does not exist for that application;
  `tungsten-golden-4rank` was called a pinned checksum when it only asserts
  finiteness; and heat3d's "observed order matches design order" was not
  supported by any test. Also documented, rather than fixed here: `aluminumNew`
  requires `T_min`/`T_max` and never reads them, and
  `tungsten_moving_bc_options.json` cannot load because `MovingBC` requires an
  `xpos` the file does not have.
- Chapters left stale by the per-application science upgrades (`#113`–`#118`)
  were brought current, since the setup tables would otherwise contradict the
  prose beside them: Fe–Cr Redlich–Kister thermodynamics and the coarsening
  measurements, the `h^3` lubrication rupture case, the anisotropic
  surface-stiffness driver, the nonlinear compliant-lubrication load case, the
  real-space \(\psi_4\)/\(\psi_6\) crystal-selection benchmark, and the
  inclusion size-effect sweep.
- LUMI HIP work uses a clean `origin/master` clone at
  `/flash/project_462001519/juaho/dev/openpfc-master` and build tree
  `/flash/project_462001519/juaho/build/openpfc-lumi-rocm-master`. Stop
  rsyncing apps onto the frozen `openpfc-0.2` HIP-scaling checkout.

### Added

- `thin_film_fd` (`#124`): a conservative face-flux finite-difference solver
  for the same `h^3` lubrication equation as `thin_film_nonlinear`, distributed
  via `pfc::decomposition` + `pfc::comm::SparseExchange` (the `apps/allen_cahn`
  pattern). Forming the flux at cell faces makes mass conservation exact to
  round-off for any timestep (telescoping, not a `k=0`-mode argument) and,
  with a harmonic-mean face mobility, keeps the film positive through
  rupture where the spectral solver structurally cannot: on the 512²,
  `dx=0.5` science case, the spontaneous run reaches a stable precursor
  plateau (`min h ~ 0.151`-`0.152`, never crossing `h*=0.15`) with 19.6 % hole
  area and its hole count falling from 227 to 204 between `t=345` and
  `t=400` (coalescence), volume conserved to `1.3e-15` relative. The
  arithmetic-mean face average was measured too, as a negative control: it
  overflows within 5 time units of approaching `h*`. A controlled
  single-Fourier-mode comparison against the spectral solver agrees to 2 %
  in `min h` and `1e-6` relative in volume; the broadband-noise flagship
  comparison shows a real, explained timing offset (near-Nyquist noise damps
  ~6x slower in the 2nd-order FD operator than in the spectral scheme's
  exact/dealiased treatment). Stable timestep measured directly:
  `dt=0.001` at this resolution; `dt>=0.003` overflows. Curvature-operator
  order 2 vs 4 compared directly (~1 % difference, not resolution-starved).
  See `apps/thin_film/README.md`, "Two methods, one problem".
- Field-visualisation pipeline for the applications report
  (`docs/report/figures/field_io.py`, `field_plots.py`,
  `make_field_figures.py`): reads OpenPFC's own field output — `.vti` (VTK
  ImageData, appended raw or base64) and headerless `.bin` MPI-IO dumps — and
  renders single-panel, time-series montage, and paired-comparison figures
  with a perceptually-uniform sequential map for one-sided fields and a
  diverging map centred on a physically meaningful midpoint for signed ones.
  Demonstrated end to end on three real LUMI runs (`run_field_demos.sh` is
  the recipe): Fe-32Cr spinodal decomposition and coarsening
  (`cahn_hilliard`), a dewetting lubricating film (`thin_film`), and a
  tungsten PFC seed nucleus sliced from a 256³ run (`tungsten`, reading the
  raw `.bin` path). Five committed SVGs are now shown, with interpretive
  captions, in the `@sec-cahn-hilliard`, `@sec-thin-film`, and
  `@sec-tungsten` chapters.

- Kawahara science case B is now an exact KdV solitary wave rather than a
  Gaussian bump (`#119`). New `kdv_soliton` initial condition, whose width
  \(W=\sqrt{-12\beta/(\alpha A)}\) is *derived* from the model coefficients
  rather than supplied, so an input cannot quietly stop being a solution of
  the equation it is run against; it refuses sign combinations that admit no
  such wave. With `gamma=0` the wave is steady, which turns the comparison
  into a controlled experiment: switching the fifth-order term on costs it
  28% of its amplitude, raises the trailing-radiation RMS 75-fold, and puts a
  wave train within 7% of \(k_{\mathrm{res}}\), the wavenumber where the
  linear phase velocity equals the pulse's own speed — a number that comes
  out of the dispersion relation and is nowhere in the solver.
- EasyBuild recipes for a LUMI-C module install (`#46`): a CPU HeFFTe 2.4.1
  (FFTW backend, `cpeGNU/25.09`) and OpenPFC 0.2.0 on top of it, under
  `easybuild/easyconfigs/`. Verified end to end on LUMI: both install, and
  `srun -n 4 tungsten` runs a 64³ case from the resulting module. Documented
  as an optional alternative to `scripts/build.sh` in `INSTALL.LUMI.md`.
- Surface-diffusion science upgrade (`#115`): an orientation-dependent
  surface stiffness `B(theta) = B0[1 + eps_a cos(m theta)]`,
  `theta = atan2(h_y, h_x)` evaluated spectrally, generalising the isotropic
  Mullins model `dh/dt = -B nabla^4 h` to `dh/dt = div[B(theta)
  grad(lap h)]`; a self-contained spectral-flux ETD1 stepper
  (`surface_diffusion_anisotropic`) since the orientation-dependent
  coefficient cannot be written as a reciprocal-space symbol; and a crossed
  sinusoidal-corrugation science preset run both isotropically and
  anisotropically from the same initial surface, reporting RMS roughness,
  structure-factor dominant wavelength, directional (kx- vs ky-dominated)
  spectral energy, max `|grad h|` and mean height to CSV. The isotropic
  `k^4` single/two-mode exact-decay tests are unchanged and remain the
  numerical oracle; the anisotropic model reduces to them exactly at
  `eps_a = 0` (checked, not assumed). `(B0, eps_a, m)` are illustrative
  parameters, not fitted to a measured material `gamma(theta)` — see
  `apps/surface_diffusion/include/surface_diffusion/anisotropy.hpp`.
- Conservative flux nonlinearity for the applications whose mobility depends on
  the field inside a divergence (`#114`): `pfc::apps::SpectralFlux` and
  `pfc::apps::FluxETD` in `apps/common`, evaluating `div(M(u) grad p)`
  spectrally with Orszag 2/3 dealiasing and an ETD1 update. Now templated on
  `MemorySpace` and running on GPU as well as host -- see the entry below.
- GPU flux path and a heroic-scale GPU dewetting run: `SpectralFlux<MemorySpace>`
  and `FluxETD<MemorySpace>` route every elementwise step (the complex `i*k_d`
  gradient, the dealias mask, the ETD combine) through
  `pfc::sim::SpectralETDOps<MemorySpace>`, which already carried a
  complex-coefficient `combine_raw` overload with real CUDA/HIP kernels behind
  it; the mobility `M(u)` is evaluated on device with the same
  `OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE` mechanism the physics nonlinearities
  use, fused with the gradient multiply in one kernel launch
  (`pfc::apps::MobilityGradPointwise`). New `thin_film_nonlinear_hip` binary
  (same JSON schema and observables as `thin_film_nonlinear`), gated on
  `OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL`,
  plus a HIP-vs-host parity `ctest` case. The host path is unchanged
  arithmetically (same operand order throughout, verified bit-identical by
  construction). Measured on LUMI (128², `dt=0.002`, 10000 steps, 1 GCD vs 1
  CPU rank): `min_h` agrees to a relative `1.6e-13`, liquid volume to
  `4.4e-15` -- FFTW-vs-rocFFT round-off, not a numerical difference.
  Heroic run (LUMI job 21857011): the spontaneous-dewetting preset at 4096²
  (dx=0.5, ~128 fastest-growing wavelengths per side, 64x the validated 512²
  case's area) on 16 GCDs (2 `standard-g` nodes), `t1=140`, `dt=0.002`
  (70000 steps) in 2333.6 s (33.3 ms/step). Volume conserved to a relative
  1.5e-14 through the valid window; the precursor is reached at `t=115`
  (`min_h=0.133`) -- earlier than the 512² case's `t=140`, consistent with
  more independent nucleation sites racing to rupture first over 64x the
  area. The run then hits the documented, out-of-scope divergence near
  `min_h~0.6 h*` around `t=130`-`135`, reproducing the known breakdown at a
  resolution and domain size the CPU verifier never exercised -- confirming
  it is a scale-independent modelling limitation, not a GPU-path defect.
- `thin_film` science case: full `h^3` lubrication mobility, a precursor
  disjoining pressure that survives hole formation, and spontaneous versus
  defect-triggered dewetting presets. A 30 % deep defect brings failure forward
  by about 40 % (precursor reached at t = 85 against t = 140) and relocates it
  to the defect. Volume conserved to round-off; the nonlinear solver reproduces
  the exact `k^4` decay at constant mobility.
- The structure factor moved from `cahn_hilliard` to
  `apps/common/include/openpfc_apps/structure_factor.hpp` now that a second
  application needs it.

- EHD film science upgrade (`#116`): nonlinear compliant lubrication under a
  flexible plate, `apps/ehd_film/src/ehd_film_nonlinear.cpp`, reusing the
  shared conservative flux stepper introduced for the `thin_film` science
  upgrade (`#114`, `apps/common/include/openpfc_apps/spectral_flux.hpp`). The
  cubic mobility `M(h)=M0(h/h0)^3` replaces the constant-mobility linear
  model in a full lubrication coupling `p = B∇⁴h - γ∇²h - Π(h) + p_ext(x,y,t)`,
  with a localized Gaussian load applied and then removed so a run shows
  loading followed by recovery/redistribution. The existing exact `k^6`
  pure-bending decay stays the CI verifier, and the nonlinear flux solver is
  asserted (not assumed) to reproduce it exactly at constant mobility. Two
  256² science presets (`load_relaxation_stiff.json`,
  `load_relaxation_compliant.json`, `B=640` vs `B=100` at fixed tension)
  measure central deflection, pressure extrema, RMS spreading radius and
  displaced volume vs time, and confirm total volume conservation to
  round-off (`~7-8e-15` relative) under the nonlinear mobility, tension and
  the time-dependent load. `EhdFilmPointwise` gained the `h_star`
  adhesion-safe precursor disjoining form ported from `thin_film` (`#114`).
  Host (CPU) only; the flux path has no HIP kernels yet, so the existing
  constant-mobility `ehd_film_hip` binary is unchanged. The adhesive/unstable
  film comparison (issue's case C) is deferred, along with energy
  diagnostics and time-periodic loading.
- `higher_order_pfc` crystal-selection benchmark (`#118`): a real-space
  bond-orientational order metric (`order_parameter.hpp`, \(\psi_4\)/\(\psi_6\)
  from detected density peaks) so square-vs-triangular selection is verified
  structurally, not inferred from reciprocal-space ring power alone.
  \(\psi_4\)/\(\psi_6\) are checked to \(10^{-9}\) against analytically
  constructed ideal square and triangular lattices — the key acceptance test.
  Cites the primary two-mode-PFC source, Wu, Adland & Karma, *Phys. Rev. E*
  **81**, 061601 (2010) (arXiv:1001.1349), and maps OpenPFC's kernel to it
  term by term; calibration is labelled representative, not quantitative (see
  the app README for exactly what was and was not verified against the
  primary source from this machine). Five 2D cases (`lattice_seed.hpp` adds
  the controlled single-crystal seed IC; `diagnostics.hpp` adds free-energy
  density + reciprocal peaks + \(\psi_4\)/\(\psi_6\) to a CSV) show the
  two-mode `q1=√2, r1=0.02` kernel selecting real-space square order
  (\(\psi_4\) local \(0.81\), mean neighbours \(4.04\)) against the
  single-mode kernel's triangular order (\(\psi_6\) local \(0.85\), mean
  neighbours \(5.87\)) from the same box/seed/quench, and confirms that the
  degenerate `r1=0` kernel's \(\sqrt2\)-ring-dominated pattern is real-space
  **triangular**, not square, despite the reciprocal-space power distribution
  alone suggesting otherwise. Two `lattice_seed` single-crystal runs hold
  their imposed symmetry to \(\psi_4\)/\(\psi_6=1.000\) over 2000 ETD steps.
  3D crystal selection is deferred; see the PR.

- Fe–Cr Cahn–Hilliard science upgrade (`#113`): Redlich–Kister excess free
  energy with the assessed bcc Cr–Fe interaction, a code-to-physical scale map
  (1 code length = 1.07 nm, 1 code time = 0.278 h at 475 °C), and
  structure-factor diagnostics — azimuthally averaged `S(k)`, first moment
  `k1`, domain length `L = 2π/k1` and dominant wavelength, all added to the
  diagnostics CSV. Two science presets measure the early-stage band selection
  (`k_peak = 0.368` against a predicted `0.362`) and late-stage coarsening
  (`L ∝ t^0.35` against the Lifshitz–Slyozov `1/3`, reaching 40 nm after
  1100 h of simulated ageing). The reduced regular-solution verifier is the
  `L1 = 0` case of the same expression, not a second code path.

- `gradient_elasticity`: derived stress/strain/energy fields and a
  misfitting-inclusion size-effect study (`#117`). Every run now also
  computes, spectrally from the displacement solution: strain (`exx`/`eyy`/
  `exy`), Cauchy stress (`sxx`/`syy`/`sxy`), the hydrostatic/von-Mises
  invariants (`stress_hydro`/`stress_vm`), and the elastic energy density
  (`energy_density`) -- via the classical local constitutive law evaluated
  on the (already `ell`-regularized) displacement field, documented as such.
  New `circular_inclusion` field modifier (`tanh`-smoothed flat-top disk of
  radius `R`) and a `line_profile` JSON block (single-rank line-cut CSV) feed
  a new `scripts/size_sweep.py` size-effect sweep: peak hydrostatic/von-Mises
  stress and total elastic energy vs. `R/ell`. For this smooth, finite
  inclusion (no classical singularity), the measured, physical direction is
  that peak stress *decreases* monotonically as `R/ell` grows, bounded above
  by a new closed-form "clamped" limit (`R/ell -> 0`, no elastic relaxation)
  and below by the classical "relaxed" limit (`R/ell -> infinity`,
  Eshelby-type) -- both derived, checked by unit tests, and documented in the
  app README, along with a documented and tested periodic-image control
  (box doubling changes peak stress by <2% at `L/max(R,ell)=16`; ratio 8
  measured ~4-5% contamination). The existing `#82` cosine/Gaussian
  analytical verifiers are untouched and stay green. Deferred: the
  dislocation/defect regularization benchmark (issue section B) and a
  systematic 4th-vs-6th-order comparison (section C) beyond the existing
  `alpha(k)` unit tests -- see the app README's "What `#117` does not
  cover".
- `kawahara`: documented capillary-gravity parameter mapping and a wave-packet
  / dispersive-radiation science case on top of the existing arbitrary-
  coefficient verification (`#119`). `capillary_gravity_mapping.hpp` maps
  depth/gravity/Bond number \((h,g,\tau)\) to \((\alpha,\beta,\gamma)\) in a
  frame moving at \(c_0=\sqrt{gh}\); this reduction is attributed in
  secondary literature to Hasimoto (1970)/Kawahara (1972)/Hunter &
  Vanden-Broeck (1983), but was not checked against a primary source
  reachable from this machine, so it is documented as **representative**, not
  quantitative calibration. A new `wave_packet` initial condition (Gaussian-
  envelope carrier at `k0`) plus `WavePacketDiagnostics` (envelope centroid/
  width via spectral low-pass of \(u^2\), exact carrier phase from the
  discrete Fourier coefficient at `k0`, and an automated no-periodic-wrap
  `edge_fraction` sentinel) measure group and phase velocity against
  \(d\omega/dk\) and \(\omega/k\) for `k0` on both sides of the
  \(k_c=\sqrt{-\beta/\gamma}\) crossover; `PulseDiagnostics` compares a
  third-order-only nonlinear pulse against the full third+fifth-order case
  (trailing-radiation RMS). Both diagnostics write CSV through the CPU
  session only (`kind: wave_packet` / `kind: pulse`). Measured on LUMI
  (\(\tau=0.30\), \(k_c=\sqrt{1.5}\approx1.2247\), `t1=250`): group velocity
  (centroid drift) vs \(d\omega/dk\) agree to 2.3% (\(k_0=0.699<k_c\), both
  \(v_g\) and \(c_p\) small and negative) and 0.4% (\(k_0=2.003>k_c\), both
  positive and an order of magnitude larger); carrier phase velocity vs
  \(\omega/k\) agrees to
  \(\sim10^{-14}\) (exact per-mode ETD rotation); `edge_fraction<10^{-7}`
  throughout, i.e. no periodic self-interaction. A nonlinear pulse run
  (`alpha=1.5`, same `beta`, amplitude 0.15, `t1=40`) shows the fifth-order
  term reproducibly lowering both trailing-radiation RMS (\(\sim7\%\) at
  `t=40`) and peak amplitude (\(\sim5.8\%\)) versus the third-order-only
  control at identical mean (conserved to displayed precision in both runs).

- Cray/LUMI site profile for the runtime baker (`#13`): `--site cray`,
  `--host-lib` for stack libraries that share a directory with ordinary system
  libraries (LUMI's `/usr/lib64/libcxi.so.1`, `libxpmem.so.0`), `--runtime` to
  select singularity-ce, a `/var/spool/slurmd` site bind that Cray PMI needs,
  and a measured `host_glibc_requirement` in provenance. Validated on LUMI with
  **4 nodes / 32 ranks** over the Slingshot `cxi` provider: the baked bundle
  reproduces the native run bit-for-bit (job 21830651).

- `higher_order_pfc`: eighth-order two-mode PFC correlation kernel (`#83`).
  \(\Lambda=-\varepsilon+(1+\nabla^2)^2[r_1+(q_1^2+\nabla^2)^2]\) is quartic in
  the Laplacian symbol (\(k^8\)); conserved dynamics makes the evolution
  operator quintic (\(k^{10}\)) with a bit-exact zero at \(k=0\). A small
  generic `PolynomialInKLap<N>` keeps both as Horner loops rather than
  hand-written \(k^8\)/\(k^{10}\) cases. Catch2 checks the kernel against the
  factored analytical form, the derived coefficient expansion, band structure,
  the \(k^4\)-vs-\(k^8\) comparison, \(e^{L(k)t}\) growth, mass conservation
  and ETD stability at 1000x the explicit limit. CPU binary plus
  `higher_order_pfc_hip` when rocFFT HeFFTe is on (LUMI-G `standard-g` jobs
  21829597 on 1 GCD and 21829755 on 2 GCD; conserved `sum` bit-identical to the
  CPU run on one GCD, within 9 ULP on two).

- Multi-app CLI catalog and installed presets for eight JSON-session apps,
  with explicit backend availability and CPU entrypoint smoke tests.
- Cahn–Hilliard seeded broadband coarsening preset and optional collective CSV
  diagnostics: mass, bounds, and current-state total energy including spectral
  gradient energy. Invalid sampled compositions stop the run; CSV files are
  never overwritten. Noise initialization is independent of MPI decomposition.
- Independent Kawahara manufactured-solution tests for third- and fifth-order
  dispersion. Correct the documented PDE to the existing `-beta*u_xxx`
  convention without changing solver coefficients; fix the pulse preset's
  grid count to the documented 256 points.

- CPU case baker (`#13` first slice): stage an Apptainer runtime image from
  a compiled CLI tungsten case, with input/runtime-library hashes and explicit
  host-MPI/PMIx/UCX mounts. Includes a baked-case entrypoint and portability
  checks. Validated with a two-rank Tohtori CPU smoke; GPU and multi-node
  networking are not yet validated.

- `openpfc` CLI (`#41` first slice): create a small tungsten case, compile
  through `scripts/build.sh` with local CPU / Tohtori CUDA / LUMI HIP profiles,
  and run with MPI or Slurm. Build and installed drivers report the project
  version; nested MPI launches are rejected. Container packaging remains #13.
- `gradient_elasticity`: isotropic Helmholtz–Navier strain-gradient
  elasticity on a 2-D periodic eigenstrain (`#82`). Fourth-order
  \(\alpha=1+\ell^2 k^2\) (optional sixth-order stretch); one-shot
  spectral \(2\times 2\) invert per \(\mathbf{k}\), not ETD. Catch2
  analytical Fourier \(\mathbf{u}\), \(\ell\to 0\) classical recovery,
  high-\(k\) reduction, and mean \(\mathbf{u}=0\). CPU binary plus
  `gradient_elasticity_hip` when rocFFT HeFFTe is on (LUMI-G smoke
  job 21820023).
- `ehd_film`: elastohydrodynamic film under a flexible plate on the 0.2
  spectral-ETD session (`#81`). Bending pressure \(B\nabla^4 h\) plus
  lubrication gives \(\lambda=-M_0 B k^6\). Catch2 exact \(k^6\) decay,
  two-mode ratio 64, and mean-gap conservation. CPU binary plus
  `ehd_film_hip` when rocFFT HeFFTe is on (LUMI-G smoke job 21799835).
- `kawahara`: capillary–gravity Kawahara waves on the 0.2 spectral-ETD
  session (`#80`). Odd-order \(L(k)=-i(\beta k^3+\gamma k^5)\) plus
  dealiased \(u^2\). Catch2 phase-velocity / \(k^3\) vs \(k^5\) / mean
  checks. CPU binary plus `kawahara_hip` when rocFFT HeFFTe is on
  (LUMI-G smoke job 21792406).
- `surface_diffusion`: Mullins small-slope surface diffusion on the 0.2
  spectral-ETD session (`#79`). Exact \(k^4\) decay of Fourier modes;
  multi-wavelength JSON demo. Catch2 single-mode \(\exp(-B k^4 t)\) and
  two-mode \((k_2/k_1)^4\) scaling. CPU binary plus `surface_diffusion_hip`
  when rocFFT HeFFTe is on (LUMI-G smoke job 21791182).
- `thin_film`: lubrication dewetting / coating on the 0.2 spectral-ETD
  session (`#78`). Capillary \(k^4\) plus van der Waals/repulsion \(\Pi(h)\);
  `A=0` is a leveling check. JSON + VTK, Catch2 volume / \(\lambda(k)\) /
  leveling. CPU binary plus `thin_film_hip` when rocFFT HeFFTe is on
  (LUMI-G smoke job 21781449).
- `cahn_hilliard`: conserved Fe–Cr-like Cahn–Hilliard on the 0.2 spectral-ETD
  session (`#77`). Regular-solution \(f(c)\) at 475 °C, \(L(k)\propto k^4\),
  JSON + VTK, Catch2 mass / linear-mode / spinodal checks. CPU binary plus
  `cahn_hilliard_hip` when rocFFT HeFFTe is on (same JSON; CPU vs HIP field
  to \(10^{-10}\)).
- `heat3d_fd_hip`: 3D heat-equation FD on HIP (`FDGPUStack`, device halo,
  `for_each_interior_device`). Same CLI as `heat3d_fd`; `HEAT3D_PROFILE_JSON`
  writes schema-v4 `wall_step` frames. LUMI submit helpers:
  `docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh`.
- `heat3d_spectral_hip`: HIP twin of `heat3d_spectral` (implicit Euler,
  2 FFTs/step on `HIPSpectralStack`). Same CLI as the CPU spectral driver;
  `HEAT3D_PROFILE_JSON` / `HEAT3D_SPECTRAL_HIP_CHECKSUM`. LUMI-G 768³
  median `wall_step` 416 / 204 / 127 / 88 / 50 / 39 / 34 ms on
  1 / 2 / 4 / 8 / 16 / 24 / 32 GCDs (52% at 16). Submit:
  `docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh`.
- Spectral r2c with 1D z-slabs puts the complex outbox on y-slabs (full
  z) so HeFFTe's z-FFT does not reshape back to z-slabs after the
  transform. That extra hop was the 16-GCD LUMI-G transpose tax.
  `tungsten_hip` 768³ median `wall_step` 177 / 84 / 72 / 63 ms on
  8 / 16 / 24 / 32 GCDs (64% at 16 vs 1 GCD 851 ms).
- Spectral stacks use a 1D slab process grid at `nproc >= 9`.
  `OPENPFC_FFT_SLAB_AXIS` (`x`/`y`/`z`) forces the split axis.
  `OPENPFC_FFT_NODE_GRID=1` selects a 1×8×N pencil grid (slower than 1D
  slabs on LUMI-G 768³). `OPENPFC_FFT_PROC_GRID=gx,gy,gz` forces that
  Cartesian grid when it factors the rank count. `slab_proc_grid` keeps
  the r2c axis in-plane.
  JSON `num_subranks` overlays HeFFTe `use_num_subranks`.
- GPU ETD multiply/combine/pointwise check launch errors only; they no longer
  `gpuDeviceSynchronize` after every kernel, so the following HeFFTe reshape
  can overlap on the default stream.
- LUMI-G scaling sbatch leaves every GCD visible (`bind_local_device` in the
  binary) so GPU-aware HeFFTe can use intra-node HIP IPC.
- GPU apps pin the local device from `SLURM_LOCALID` *before* `MPI_Init` so
  Cray `MPICH_OFI_NIC_POLICY=GPU` sees the intended GCD. All GCDs stay
  visible for IPC. HeFFTe 2.4.1 `p2p_plined` pack-all patch:
  [`cmake/heffte-2.4.1-p2p-plined-packall.patch`](cmake/heffte-2.4.1-p2p-plined-packall.patch).
- `SpectralETDSession` prints `SPECTRAL_CHECKSUM` / `_HEX` after the run
  so 1-vs-N GCD field sums do not need I/O.

- LUMI `tungsten_hip` 768³ 16/32 GCD points (`#87`): 222 ms (24%
  efficiency) and 103 ms (26%). 16 GCDs is slower than 8 GCDs on this
  grid. Submit helper: `multinode` mode.
- LUMI-G `tungsten_hip` 768³ 1–8 GCD strong-scaling pins (`#87` first
  slice). Median `wall_step` 847 / 455 / 315 / 215 ms; efficiency 49% at
  8 GCDs. See [`docs/hpc/lumi_gpu_scaling.md`](docs/hpc/lumi_gpu_scaling.md).
- JSON/TOML `saveat <= 0` is accepted and disables periodic saves, matching
  `Time` and the spectral config reference. Negative sentinels such as
  `saveat = -1` in LUMI I/O-off inputs no longer fail at parse.
- Tungsten optional `G_grid` / `V_grid` / `x_initial` thermal drive (`#34`).
  Omitting the keys keeps the isothermal CPU goldens; CPU/CUDA/HIP share
  `TungstenPointwise`.
- `FileResultsWriter` expands `$NAME` / `${NAME}` in result filename patterns
  at writer setup, then applies the existing increment template (`#65`).
  Unset, empty, or stray `$` fail closed.
- LUMI-G `tungsten_hip` 1–8 GCD scaling recipe (`#87` first slice): I/O-off
  TOML, parameterized sbatch, and [lumi_gpu_scaling.md](docs/hpc/lumi_gpu_scaling.md).

### Fixed

- Moving-boundary spectral restarts retain the unwrapped front position,
  scan index, and detection state in the atomic checkpoint bundle (`#37`).
  Missing front state or changed BC configuration fails at startup. Tungsten
  JSON/TOML restart inputs now use `restart_from` with a moving-front case.
- Spectral checkpoints publish after scheduled result output, preserving the
  next output index so a resumed run does not reuse its last saved filename.

- LUMI `tungsten_hip` scaling sbatch applies the 8-GCD `map_cpu` bind only
  for 8-rank jobs. Prefixing that map on a 1-GCD `dev-g` allocation made
  `srun` fail (`CPU binding outside of job step allocation`).
- Docs `uv.lock` urllib3 2.7.0 and idna 3.19 (GHSA-qccp-gfcp-xxvc,
  GHSA-mf9v-mfxr-j63j, GHSA-65pc-fj4g-8rjx).
- CUDA Field residency compile-check is an OBJECT library, not a Catch2 TU.
  A namespace-scope constructor used to run CUDA at process start during CTest
  discovery on GPU-less runners (#67).
- `tungsten-golden-4rank` and `aluminum-golden-4rank` honor
  `OpenPFC_MPI_TEST_MAX_WORLD_SIZE` (CI sets 2). They were launching 4 ranks on
  GitHub runners, which Open MPI rejects; gcc-13 jobs hid the same CTest
  failures because `ctest | tee` dropped the exit code.

### Changed

- Historical 0.2 planning documents (architecture audit, execution plan,
  migration map, refactoring roadmap) moved to `docs/archive/`. Current
  architecture is `docs/concepts/architecture.md`.
- Pull-request CI is gcc-13 Debug+Release, code-quality, packaging, and
  path-filtered docs. gcc-11 Debug, CUDA/HIP compile-only, and coverage run on
  `master` pushes (coverage also weekly). Clang-tidy is weekly plus
  `workflow_dispatch`, not per PR.

## [0.2.0] - 2026-09-05

Breaking architecture release. There is no `pfc::Model`, `pfc::Simulator`,
`pfc::ui::App`, or `pfc::World`. Port applications with
[`docs/MIGRATION_0.1_to_0.2.md`](docs/MIGRATION_0.1_to_0.2.md). 0.1.x source
compatibility is not a goal.

**Highlights**

- One `pfc::sim::SpectralETDSystem<Physics, MemorySpace>` and one JSON
  `pfc::ui::SpectralETDSession<Physics, Stack>`.
- Canonical field is `pfc::data::Field<T, MemorySpace>`; modifiers and writers
  take `FieldView` / `FieldOutput`.
- Host halos are `pfc::comm::HaloExchange` / `SparseExchange`; GPU copies live
  under `runtime/gpu/`.
- GitHub CI: CPU gcc-11/13 Debug+Release, 2-rank MPI, packaging smoke, coverage,
  docs, CUDA/HIP compile-only (FD/kernel, no HeFFTe in those jobs).
- In-tree perf pins are schema-v4 summaries, not full frame traces.

The sections below are the development record that landed on `master` with
PR #71.

### Architecture consolidation before 0.2.0 (2026-09-03)

One ETD driver, one JSON session, one field boundary. This iteration removes the
duplication that survived the milestone work (six ETD drivers, four app
sessions, a second field representation at the modifier/writer boundary).

- One `pfc::sim::SpectralETDSystem<Physics, MemorySpace>` (`kernel/simulation/spectral_etd_system.hpp`) replaces the six host/device ETD drivers: `SpectralETDSystem`, `SpectralMeanFieldETDSystem`, `MovingFrameMeanFieldETDSystem` and their `DeviceSpectralETDSystem` / `DeviceSpectralMeanFieldETDSystem` / `DeviceMovingFrameMeanFieldETDSystem` twins are deleted. Optional physics capabilities `nonlinear_symbol(k)`, `filter_mf(k)`, `correlation_kernel(k)`, and `free_energy_density(cell)` are detected at compile time, so plain, mean-field (tungsten), and moving-frame (aluminum) models run through the same class.
- `SpectralETDSystem` exposes `attempt(t)` / `commit()` / `reject()` plus `set_dt(dt)` (re-prepares the ETD coefficients); `step(t)` is attempt + commit. Backend work goes through the memory-space policy `SpectralETDOps<MemorySpace>` (host in `kernel/simulation/spectral_etd_ops.hpp`, CUDA/HIP in `runtime/gpu/spectral_etd_ops_gpu.hpp`), which also stamps field residency so no driver calls `note_*_write` by hand.
- Physics nonlinearities are device-capable functors: `physics.pointwise()` returns a trivially copyable `OPENPFC_HD` struct evaluated per `pfc::sim::SpectralCell` (`kernel/simulation/spectral_pointwise.hpp`). On device it runs in `spectral_pointwise_apply` (`runtime/gpu/spectral_pointwise_gpu.hpp`); aluminum and the plain ETD path no longer round-trip the nonlinearity through the host. Each app instantiates its functor once in `src/gpu/<app>_pointwise.inc` (stamped into `.cu` / `.hip`) with `OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE`; a missing instantiation fails at link time.
- Removed the `nonlinearity_poly()` escape hatch and the `polynomial_nl_cuda_impl` / `polynomial_nl_hip_impl` kernels.
- Physics concepts: `SpectralETDPhysics` (`declare_fields` + `linear_symbol` + `pointwise`) with `HasMeanFieldFilter` / `HasCorrelationKernel` / `HasNonlinearSymbol` / `HasSpectralPointwise` replace `SpectralNonlinearity`, `SpectralDiagonalPhysics`, `MeanFieldETDPhysics`, and `MovingFrameMeanFieldETDPhysics`. `MeanFieldNonlinearityPoly` is gone.
- One generic JSON session `pfc::ui::SpectralETDSession<Physics, Stack>` (`frontend/ui/json_spectral_etd_session.hpp`) replaces `TungstenETDSession`, `TungstenETDGPUSession`, `AluminumETDSession`, and `AluminumETDGPUSession`. It uses the `FieldModifier` catalog for `initial_conditions[]` / `boundary_conditions[]` (`default` target = the physics' primary field), the `ResultsWriter` catalog for `fields[]` (`.vti` / `.vtk` paths default to the `vtk` writer), `CheckpointService` on every memory space (GPU sessions now checkpoint and restart), and the JSON `profiling` section through `pfc::ui::JsonStepProfiler` (`frontend/ui/json_step_profiler.hpp`). Physics provide `static Physics from_json(params, domain, box)`.
- `pfc::ui::run_json_session_main<Session>(argc, argv, label, setup)` (`frontend/ui/json_session_main.hpp`) is the shared `main()` for JSON-driven binaries.
- Tungsten: `tungsten_session.hpp` defines `TungstenSession`, `TungstenCUDASession`, `TungstenHIPSession`; `tungsten_pointwise.hpp` holds the functor; `tungsten::register_catalog()` registers the `fixed` / `moving` BCs. Deleted `tungsten_etd_session.hpp`, `tungsten_etd_gpu_session.hpp`, `tungsten_etd_io.hpp`, `tungsten_etd_profile.hpp`, `tungsten_field_modifiers.hpp`, `common/tungsten_app_main.hpp`, and the `*_etd.cpp` mains. Binary names are unchanged (`tungsten`, `tungsten_etd`, `tungsten_cuda`, `tungsten_etd_cuda`, `tungsten_hip`, `tungsten_etd_hip`).
- Aluminum: `aluminum_session.hpp` defines `AluminumSession`, `AluminumCUDASession`, `AluminumHIPSession`; `aluminum_pointwise.hpp` holds the functor (including `temperature_variation(x, t)`); `seed_grid_fcc` is a catalog `FieldModifier` (`include/aluminum/seed_grid_fcc.hpp`) registered by `aluminum::register_catalog()`. Deleted `aluminum_etd_session.hpp`, `aluminum_etd_gpu_session.hpp`, `aluminum_etd_io.hpp`, `aluminum_field_modifiers.hpp`; mains moved to `src/`. Binary names are unchanged (`aluminumNew`, `aluminum_etd`, `aluminum_etd_cuda`, `aluminum_etd_hip`).
- Tests: `test_spectral_etd_system.cpp` covers plain / mean-field / moving-frame toys plus attempt-reject-commit and `set_dt`; one `test_spectral_etd_system_gpu.cpp` (`CUDA_SpectralETD` / `HIP_SpectralETD`) pins device vs host for all three shapes; `tests/fixtures/spectral_etd_toys*.hpp` hold the shared toys.
- Removed the `pfc::Field` / `RealField` / `ComplexField` `std::vector` aliases (`kernel/data/model_types.hpp` deleted); `pfc::Field<T, MemorySpace>` now names the canonical owning field (`pfc::data::Field`).
- `FieldModifier::apply` takes `field::FieldOutput<double>` (mutable non-owning view) instead of `std::vector<double>&`; a vector lvalue still converts implicitly.
- Added `pfc::apply_field_modifier(modifier, field, t)` (`kernel/simulation/apply_field_modifier.hpp`), which applies any modifier to a host or device `Field` and owns the host-mirror residency bracket.
- `ResultsWriter::write` and `BinaryReader::read` take `field::FieldView` / `field::FieldOutput`; `Field::view()` / `Field::output()` added for host fields; `FieldView` / `FieldOutput` gained `std::vector` constructors and `FieldOutput` gained element access and `begin()` / `end()`.
- `field::apply*` helpers (`operations.hpp`) and `checkpoint::write_real_brick_mpi` take field views, not vectors.
- `CheckpointService::save` / `load` / `maybe_save` / `restore_from_config` are templated on `MemorySpace`, so GPU sessions checkpoint and restart through the host mirror.
- One checkpoint publisher: `publish_checkpoint_directory(final_dir, meta, comm, write_fields)` is MPI-collective and `CheckpointService` writes bricks through it with `brick_io.hpp`; the serial `std::ofstream` brick path and `PublishedFieldBrick` are removed.
- Removed the deprecated `Decomposition`-only constructors of `comm::detail::HostFacesHalo` / `HostFullHalo` / `HostPersistentFaces` and the deprecated `(GridSize, PhysicalOrigin, GridSpacing)` constructors of `SpectralCPUStack` / `FDCPUStack`; use the `Box3i` + `Domain` / `Domain` constructors. No `[[deprecated]]` symbol remains in `include/`, `src/`, `apps/`, `examples/`.
- Host and device `Faces` halo exchangers no longer retain a `const Decomposition&`; the decomposition is consulted only during construction.
- `decide_gpu_aware_mpi` reports `GPUAwareMPIHow::Undetermined` when compiled GPU-aware but no rule decided, instead of the misleading "not compiled GPU-aware".
- Removed the unused `frontend/ui/simulation_wiring.hpp` umbrella and `frontend/utils/field_iteration.hpp`; include the `simulation_wiring_*` slices directly. The stray `scripts/build_LUMI.sbatch` is gone (`scripts/build.sh` generates its own).
- Docs: time-integration contract, custom stepper guide, ADR 0003 and related pages rewritten onto `pfc::sim::run` / `SimulationState`; stale `Simulator` / `App::main` / `Model::step` references removed.

### Changed

- Perf pins in `tests/baselines/perf/` are schema-v4 summaries (mean/median/min/max of frame scalars after warmup). Full per-frame traces stay a cluster export under `results/`; collapse a new pin with `scripts/compare_perf_baseline.py --summarize`.

### Fixed

- GitHub Unit Tests run the monolithic `openpfc-all-tests` CTest entry. Catch2 3.3.2 `catch_discover_tests` drops names that contain `[`, so sharded discovery registered 260 of 932 cases and never ran the suite. Skip `MPI_Init` on Catch2 `--list-*`.
- CUDA/HIP compile-only jobs use official `nvidia/cuda` devel and `rocm/dev-ubuntu-24.04:6.2-complete` images (`nvcc` / `hipcc` already in the image). They do not download or compile HeFFTe; FD/kernel GPU TUs build with `OpenPFC_ENABLE_HEFFTE=OFF` and `OpenPFC_BUILD_TESTS=OFF`. Install `git` before `actions/checkout@v5`.
- Open MPI GPU-aware query: only call `MPIX_Query_cuda_support` / `MPIX_Query_rocm_support` when `OMPI_HAVE_MPI_EXT_CUDA` / `OMPI_HAVE_MPI_EXT_ROCM` is 1. Ubuntu Open MPI ships `<mpi-ext.h>` with CUDA but not HIP; the old `#define` made HIP apps fail to compile.
- HIP TUs compile with `-fPIC` so they link into PIE executables (Ubuntu 24.04 default). Without it, `wave2d_hip` failed `R_X86_64_32 against .rodata`.
- GitHub Actions: `fsfe/reuse-action@v6` (reads `REUSE.toml`); bump checkout/cache/artifact/github-script/codecov to current majors.
- Coverage: stop overwriting `CMAKE_CXX_FLAGS` in CompilerSettings; apply `--coverage` before tests are added so `lcov` finds `.gcda`.
- REUSE: copyright on `grid_field` / `residency` headers and tests; SPDX on wave2d README, docs toolchain files; `REUSE.toml` annotations for `uv.lock`, `.python-version`, `integrator_selection.json`, and vendored stb.
- Doxygen: drop unknown `\\partial_t` / `\\Delta` commands in `json_fd_session.hpp`; replace unresolved `@ref` to C++20 concepts and checkpoint payload types so the unresolved-reference count stays within the legacy baseline.
- GitHub Actions `code-quality` no longer depends on the private `ahojukka5/clang-format-action` (the VTT-ProperTune token cannot resolve it, so the Unit Tests workflow never reached a compiler). The check uses public `jidicula/clang-format-action@v4.18.0` and stays advisory (`continue-on-error`).
- Clang-tidy is a dedicated non-blocking workflow again. `WarningsAsErrors` is not clean on this tree; making it a Unit Tests gate skipped the GCC matrix.
- `docs/user_guide/parameter_validation.md` no longer links to the deleted `appconcretemodelmain-order-of-operations` anchor or the removed `JsonAppRun` / `SpectralSimulationSession` path (Sphinx `myst.xref_missing`).
- `restore_field` reports `BytesSizeMismatch` / `BufferTooSmall` before optional decomposition metadata, so a truncated payload is not classified as `DecompositionMismatch`.
- Gen-1 `Model` stores `World` by value. Tungsten/aluminum Domain constructors were passing a temporary `World` into a dangling reference, so Debug NaN checks aborted `tungsten-all-tests`, `tungsten-golden-4rank`, and `tungsten-cpu-vs-cuda-tests` on garbage k-space (`k_lap = -inf`).
- `OpenPFC_ENABLE_HIP=ON` is fail-closed: configure stops if HIP is not found instead of silently building a CPU tree. `scripts/build.sh --with-rocm` locates `ROCM_PATH` (Tohtori `rocm/7.2.1` does not set `CMAKE_PREFIX_PATH`; `hipcc` may be `/usr/bin/hipcc`) and refuses to continue if the CMake summary does not report HIP available. CMake 3.21–3.24 uses ROCm Clang as `CMAKE_HIP_COMPILER`, not the `hipcc` wrapper.
- `strong_types.hpp` skips `<compare>` / defaulted `operator<=>` on HIP device TUs (`__HIPCC__` / `__HIP__`) as well as CUDA. HIP clang (`-x hip`) cannot find `<compare>`.
- `DeviceFullHalo` keeps the 26-direction widening passes when GPU-aware MPI is off. Real-neighbor axes pack on device, MPI on host, and unpack; `*_FORCE_PACKED_HALO=1` remains 6-face-only. Tohtori HIP job 1618752: 47/47 including 2- and 4-rank Full hash tests.
- Device `SparseExchange` host-stages send/recv slabs when GPU-aware MPI is off (gather/scatter stay on device). Allen-Cahn and wave2d CPU-vs-HIP run on Tohtori HIP (job 1618762).
- `HaloExchangeOptions::selector` applies a per-rank `HaloDirectionSelector` on the host and device facades. Device Faces/Full use `pfc::halo::opposite_slot` instead of a local copy.
- Host Faces `HaloExchange::exchange()` / `finish()` posts every bound field, then one `MPI_Waitall`. Full stays sequential (each axis pass must complete before the next). Persistent stays per-field `exchange_halos()` (self-wrap persistent is MPI-implementation-sensitive on this Open MPI). 4-rank multi-field batch still equals two singles.
- `make_structured_halos` and device `SparseExchange` copy already-sorted host indices into the target `SparseVector` instead of `get_index` plus a second sort.
- Unused vendor thin includes `runtime/{cuda,hip}/padded_device_halo_exchange.hpp` and `full_padded_device_halo.hpp` are deleted. Device Faces/Full live in `runtime/gpu/`.
- Host Faces MPI (`HostFacesHalo`) is inlined into `comm_halo_exchange.hpp`. `padded_halo_exchange.hpp` is deleted.
- Host Full MPI (`HostFullHalo`) is inlined into `comm_halo_exchange.hpp`. `full_padded_halo_exchange.hpp` is deleted.
- Host persistent Faces MPI (`HostPersistentFaces`) is inlined into `comm_halo_exchange.hpp`. `halo_persistent.hpp` is deleted.
- `OpenPFC_ENABLE_HEFFTE=OFF` is an FD-only / kernel-only configure: HeFFTe is not required, `fft.cpp` is omitted, spectral apps/examples are skipped, and FD apps still link. Catch2 TUs that include `fft_fftw.hpp` / `<heffte.h>` are skipped; FD/halo/k-space tests still build.
- `session_stack_factory.hpp` includes the spectral CPU stack only when HeFFTe is on, so FD factory tests build in an FD-only tree.
- `RandomSeeds` / `SeedGrid` use `std::numbers::pi` instead of `atan(1.0)` (k-space folding stays in `kspace.hpp`).
- `HostPersistentFaces` marks unused `domain` / `patterns` so `-Wunused-parameter` is clean.
- GPU FFT factories live in `runtime/gpu/fft_gpu.hpp` / `fft_gpu.cpp`. Vendor `fft_cuda.hpp` / `fft_hip.hpp` are thin includes; `fft_cuda.cpp` / `fft_hip.cpp` are deleted.
- Host FFT call sites use `pfc::fft::IHostFFT`. The temporary `IFFT` alias is removed.
- `ImexEulerComposer::attempt` returns `StepAttemptResult`. Solve extras are `last_solve_*` accessors (`ImexStepAttemptResult` deleted).
- Device Faces `HaloExchange::exchange()` posts every bound field, one `MPI_Waitall`, then unpacks. Full stays sequential. CUDA 1-rank two-field wrap and 2-rank batch-equals-singles in `test_comm_halo_exchange_gpu.cpp`.

### Added

- M12: deleted Gen-1 `pfc::Model` / `pfc::Simulator` / A1 `LegacyModelPhysics` and their tests. JSON wiring returns FieldModifier and writer lists; checkpoints restore only through `CheckpointService`. API examples 03/07/09/10 use `SpectralCPUStack` + `pfc::sim::run`. Migration guide: `docs/MIGRATION_0.1_to_0.2.md`.
- M12 A0: deleted `pfc::World` (`world.hpp` / `world::create` / `from_json<World>` / stack `world()` / decomposition subworld shim). Callers use `Domain` + `Box3i`. LUMI HIP job 21686027: 59/59 CTest (1 skip `HIP_ExchangeFailClosed`).
- `kobayashi_fd_hip` writes schema-v2 profiling JSON with `--output PATH.json` (`--warmup N` untimed steps). Positional HEX CLI is unchanged. LUMI `dev-g` job 21689652: `tests/baselines/perf/lumi-dev-g-kobayashi-hip-1rank-release-256.json` (1 rank, 256²/200 steps, warmup 20, mean `wall_step` ~0.30 ms).
- Shared `test_step_protocol` covers all seven leaves: EmbeddedRKStepper extra-`dt` and ImexEulerStepper extra-`StageContext` attempt isolation/commit. CTest `test_step_protocol`.
- Examples `04_diffusion_model`, `05_simulator`, and `12_cahn_hilliard` build spectral operators with `for_each_kpoint` instead of a hand-rolled Nyquist fold.
- CTest `aluminum-golden-4rank`: 4-rank 16³/20-step `AluminumETDSession` vs Gen-1 ≤1e-10 (local max) / 1e-12 relative Σψ².
- CPU-only checksum goldens for the CPU-vs-GPU parity configs: CTest `tungsten-cpu-golden`, `allen-cahn-cpu-golden`, `wave2d-cpu-golden` (1e-10 relative). Capture on tohtori `g0005` is in `tests/baselines/BASELINES.md`.
- Named CTest `session-matrix` (CPU JSON session) and `session-matrix-cuda` (GPU spectral stack JSON, same `method`/`backend` keys). HIP twin `session-matrix-hip` is wired for amdgpu/LUMI.
- CTest `tungsten-etd-cpu-golden` and `tungsten-etd-cpu-vs-cuda`: 0.2 `TungstenETDSession` vs `TungstenETDCUDASession` on the Gen-1 CPU-vs-CUDA 32³/10-step sine IC, max abs ≤1e-10. CPU checksum matches Gen-1 `tungsten-cpu-golden` on g0005.
- Production `tungsten` / `tungsten_cuda` / `tungsten_hip` binaries drive 0.2 ETD sessions (same JSON/TOML CLI). `TungstenETDSession` accepts `seed_grid` IC, moving BC, VTK via `.vti`/`.vtk` paths, and JSON `profiling` (`wall_step` compatible with Gen-1). `tungsten_scalability` uses the same sessions (double only).
- Deleted Gen-1 tungsten `Model` triplet (`cpu`/`cuda`/`hip` `tungsten_model.hpp`, vendor kernels, `tungsten_ops.hpp`, `tungsten_etd_workspace.hpp`, `run_tungsten_gpu_vtk.hpp`). CPU-vs-CUDA CTest names now run the 0.2 ETD session.
- Device mean-field ETD evaluates `N(ψ,ψ_MF)` with a GPU polynomial kernel when physics exposes `nonlinearity_poly()` (tungsten). Host-view fallback remains for other physics.
- Examples 04, 05, 10, 12, `diffusion_model*`, and world strong-types/query helpers use `Domain` + `SpectralCPUStack` / `pfc::sim::run` instead of `Model` / `World` / `App`.
- `GPUSpectralStack` binds this MPI rank to `local_rank % n_devices` so multi-rank CUDA/HIP spectral sessions use one GPU per rank (same mapping as allen_cahn / kobayashi).
- `fft::create_cuda` / `create_hip` and `GPUSpectralStack` take `heffte::plan_options`. JSON `plan_options` overlay (`use_gpu_aware`, `p2p_plined`, pencils) applies to GPU spectral sessions and `make_simulation_session<GPUSpectralStack<…>>`, matching the CPU stack.
- 0.2 ETD Release remasure on tohtori `g0005` after GPU `plan_options` overlay + per-rank device bind: CUDA 1-rank 256³ `wall_step` 0.00543 s (−79% vs Gen-1 0.0253 s); CUDA 8-rank 0.00253 s (−89% vs Gen-1 0.0240 s). Previous 8-rank 0.0864 s was GPU 0 contention plus HeFFTe defaults. Table in `tests/baselines/BASELINES.md`.
- 0.2 ETD Release perf JSON on tohtori `g0005` vs Gen-1 (pre-overlay): CUDA 1-rank 256³ **faster** (`wall_step` 0.00659 s vs 0.0253 s); CPU 1-rank 64³ +35%; CUDA 8-rank 256³ slower (no HeFFTe `plan_options` overlay on `GPUSpectralStack`). Superseded by the remasure above.
- CTest `aluminum-etd-cpu-golden`: 0.2 `AluminumETDSession` 32³/5-step `seed_grid_fcc` matches Gen-1 ≤1e-10; CPU checksum pin on g0005. Host-buffer `fill_seed_grid_fcc` matches `SeedGridFCC::apply`.
- Production `aluminumNew` drives `AluminumETDSession` (same JSON/TOML CLI). `aluminum_etd` is an alias.
- Deleted Gen-1 `Aluminum.hpp` and Model-based `SeedGridFCC`/`SlabFCC` modifiers. Tests use `AluminumPhysics` + session only. `grep -rn 'public pfc::Model' apps/` is empty.
- Gen-1 aluminum `prepare_operators` uses `for_each_kpoint`. SeedGridFCC/SlabFCC use `std::numbers::pi`.
- Example `03_parallel_fft` uses `std::numbers::pi` instead of `atan(1.0)`.
- `JsonWiringSession` constructor parameters no longer shadow the catalog members (`-Wshadow`).
- `scripts/build.sh --no-heffte` configures an FD-only / kernel-only tree (`-DOpenPFC_ENABLE_HEFFTE=OFF`).
- `scripts/compare_perf_baseline.py` compares `wall_step` means against a stored baseline (schema v2/v3 traces or v4 summaries; pass ≤5% regression, warn >5%, fail >15%). Canary input: `tests/baselines/perf/inputs/tungsten_canary.json`.
- Tungsten CUDA Release 256³/20-step perf JSON on tohtori `g0005`: 1-rank and 8-rank (`tests/baselines/perf/tohtori-g0005-tungsten-cuda-*-release-256.json`). Input: `tests/baselines/perf/inputs/tungsten_release_256.json`. Compare with `--warmup-frames=1`.
- Tungsten CPU Release 64³/20-step strong-scaling JSON on tohtori `g0005`: 1/4/16 ranks (`tests/baselines/perf/tohtori-g0005-tungsten-cpu-*-release-64.json`). Compare with `--warmup-frames=1`.
- `examples/23_halo_microtiming` times `HaloExchange` (host / `--cuda` / `--hip`) and writes schema-v2 JSON. CTest `halo-microtiming-host-2rank` smokes a 16³/4-iter run.
- Halo-exchange microtiming JSON on tohtori `g0005` Release (128³ Faces, 2/4/8 ranks, host and CUDA): `tests/baselines/perf/tohtori-g0005-halo-*-release-128.json`. Compare with `--warmup-frames=5`.
- `scripts/sbatch_openpfc_hip_amdgpu.sbatch` builds HeFFTe-ROCm (Open MPI 5 + ROCm 7) if needed and runs the HIP Debug suite on Tohtori `amdgpu` with four MPI slots. Tohtori `g0004` job 1618727: 47/47 CTest batches (`HIP_SPECTRAL=ON`, `MPI_HIP_AWARE=OFF`).
- Kobayashi CUDA `KOBAYASHI_VERIFY_HEX` 32²/4-step smoke is CTest (`kobayashi-cuda-hex-smoke`, and `kobayashi-cuda-hex-2rank` when MPI suites are on). Pin matches CPU HEX except `sum_T` (1 ULP), recorded in `tests/baselines/BASELINES.md`.
- CUDA `padded_halo_faces.cu` is compiled into `openpfc_gpu_kernels` (same as HIP `padded_halo_faces.hip` in `openpfc_hip_kernels`). `openpfc-tests`, `kobayashi_fd_cuda`, and `test_fd_gpu_stack_cuda` no longer recompile the TU.
- M4 leftover device-halo tests use `pfc::comm::HaloExchange<CUDASpace/HIPSpace>` instead of `PaddedDeviceHaloExchanger` / `FullPaddedDeviceHalo`. Unique 26-direction hash fill (1/2/4 ranks, hw=2, two fields), Full+Axes3D face-only, and Faces self-wrap (hw=1 and packed hw=2) live in `test_full_padded_device_halo*.cpp` / `test_padded_device_halo_self_wrap*`. CUDA execute on tohtori; HIP execute is M-LUMI.
- M4 leftover host halo tests use `pfc::comm::HaloExchange` instead of `PaddedHaloExchanger` / `FullPaddedHaloExchanger` / `HaloExchanger` / `PersistentHaloExchanger`. Unique hw=2, 2x2x1, Full 1/2-rank hash, and Axes2D slab coverage lives in `test_comm_halo_exchange.cpp`, `test_comm_halo_exchange_modes.cpp`, and `test_halo_direction_set.cpp`. `HaloDirectionSelector` remains old-API-only; neighbour mismatch still goes through `validate_neighbour_direction_agreement`.
- M11 JSON `restart_from` is exclusive of leftover `simulator.increment` / `result_counter`. ETD sessions honor `checkpoint.every` and `restart_from` via `CheckpointService`.
- M11 heat3d FD and tungsten ETD restart-equivalence tests use `CheckpointService`. App-local `heat3d`/`wave2d` `state_capture.hpp` adapters are deleted.
- M11 coupling surface `pfc::coupling::FieldHandle` (`kernel/simulation/coupling.hpp`): host export of name + `FieldView` + owned box + spacing/origin. Example `22_external_coupling` is a mock FEM loop with `Time::clip_attempt_dt` and a FieldModifier-shaped source adapter. Docs: `docs/extending_openpfc/external_coupling.md`.
- M11 `CheckpointService` (`kernel/simulation/checkpoint_service.hpp`) owns filesystem restart: JSON `checkpoint.every` / `checkpoint.directory` / `restart_from`. Bundles are `<directory>/step_<increment>/` with versioned `metadata.json` and collective MPI-IO `fields/<name>.bin` (kernel `brick_io.hpp`, not frontend `BinaryWriter`). `from_json(CheckpointMetadata)` rejects schema-version mismatch. Load restores owned fields, `Time` increment/time, result counter, and method identity; grid or method mismatch is a hard error. Interrupted writes never leave a loadable `final_dir`.
- M10 JSON FD CPU session (`json_fd_session.hpp`): `SimulationSession<FDCPUStack>` + RK composer from `Time::method()`, Laplacian heat RHS via `stack.du()`. Session-matrix FD JSON runs two Euler steps on a sine eigenmode (amplitude decays).
- M10 `compose_etd1` / `compose_imex_euler` construct `ETD1Stepper` / `ImexEulerStepper`. JSON `"etd1"` / `"imex_euler"` remain identity on `Time`; `compose_scalar` fail-closes and names those entry points. `compose_etd1` matches the closed-form diagonal update; `compose_imex_euler` runs a diagonal implicit step.
- M10 JSON sessions use `SimulationSession<SpectralCPUStack>` (`make_simulation_session` overlays HeFFTe `plan_options`). The frontend `spectral_cpu_stack.hpp` / `_detail.hpp` twin and Gen-1 `SpectralSimulationSession` / `App` are deleted.
- M10 `StagePreparationService` is the single pre-stage BC protocol: FD `apply_dirichlet_ghosts` (node-centered odd reflection on a non-periodic axis) and spectral penalty writes share the injectable hook. `ExecutionService::prepare_boundaries` and `StageContext::needs_boundary_update` are removed; drivers pass `needs_boundary` to `requirements_from`. Dirichlet sine on a non-periodic x-axis is a discrete Laplacian eigenmode; wave2d mixed BC uses the same service.
- M10 `HDF5Writer` writes multi-rank `/field` + XDMF. Parallel HDF5 (`H5_HAVE_PARALLEL`) uses collective MPI-IO hyperslabs; serial HDF5 gathers bricks to rank 0. Buffer-size mismatches fail closed via `MPI_Allreduce` like `BinaryWriter`. Two-rank x-split round-trip is `[hdf5][MPI]`.
- M10 JSON `"writer": "hdf5"` (`HDF5Writer`, `OpenPFC_ENABLE_HDF5`) writes a 3D `/field` dataset and an XDMF sidecar. Complex fields fail closed.
- M10 `apply_writer_domain` maps `Domain` + owned `Box3i` (or a `Field`) onto `ResultsWriter::set_domain`. Simulator and ETD I/O use it; Gen-1 still supplies the FFT real-space box as the owned box.
- M10 App path names drop `spectral_`: `JsonAppRun` (`app_json_run.hpp`) and `configure_json_driver_hooks`. Gen-1 `SpectralSimulationSession` is unchanged.
- M10 GPU session-matrix JSON: `make_simulation_session` on CUDA/HIP `GPUSpectralStack` / `FDGPUStack` binaries (`backend` `cuda`/`hip`, HIP `rocm` alias).
- M10 JSON integrator tokens `imex_euler` and `etd1` parse onto `Time::method()`. They are identity only (`make_tableau` throws); unknown tokens still use `format_config_error`. Short `"imex"` remains invalid.
- M10 `ResultsWriterCatalog` matches `FieldModifierCatalog`: `register_writer` / `create_writer` (throws `format_config_error`) / `registered_writer_types`. Unknown writer keys fail before mkdir.
- M10 `FileResultsWriter` holds filename-pattern templating. `BinaryWriter` and `VTKWriter` derive from it; kernel `ResultsWriter` no longer requires a dummy path.
- M10 `FixedBC` / `MovingBC` moved out of the kernel into `apps/common`. Tungsten and aluminum register `"fixed"` / `"moving"` at startup. The built-in modifier catalog is ICs only.
- M10 heat3d FD order sweep: even orders 2, 8, 10 reduce Gaussian L2 vs the analytic solution on `FDCPUStack`.
- M10 JSON wiring takes only `JsonWiringContext` / `JsonWiringSession`. The `(comm, rank, rank0)` overload family is removed.
- M10 `SimulationSession<Stack>` owns `SessionSelection`, `Time`, and a stack from `stack_builder`. JSON helper `make_simulation_session`. Session-matrix tests cover spectral/fd × cpu (JSON) and CUDA stacks (cluster binaries).
- M10 unknown IC/BC `"type"` and a missing Model field for modifiers or results writers are hard errors (`format_config_error` / `invalid_argument`). JSON `"writer": "vtk"` writes a `.vti` in the driver-seam test. `apply_simulator_section_from_json` overlays `simulator.integrator.method` on `Time`.
- M10 GPU stack factory (`runtime/gpu/session_gpu_stack_factory.hpp`) builds `GPUSpectralStack` / `FDGPUStack` from `SessionSelection`. Omitted JSON `backend` maps to the binary's device; explicit `cpu` still fails closed. Tungsten/aluminum GPU ETD sessions use `make_gpu_spectral_stack`.
- M10 CPU stack factory (`session_stack_factory.hpp`) builds `SpectralCPUStack` / `FDCPUStack` / `FDPaddedCPUStack` from `SessionSelection`. FD halo width is `fd_order/2`. Mismatched method/backend fail closed. Tungsten/aluminum CPU ETD sessions use `make_spectral_cpu_stack`.
- M10 JSON `method` (`spectral`|`fd`), `backend` (`cpu`|`cuda`|`hip`, with `fftw`/`rocm` aliases), and `fd_order` (even 2–20) parse to `SessionSelection`. Unknown values throw `format_config_error`. Intended stack name is recorded for the session-matrix test.
- M10 unknown JSON `"writer"` is a hard error (`format_config_error`). Built-in catalog registers `vtk` as well as `binary`.
- M10 `SimulationDriver` / `pfc::sim::run` (`kernel/simulation/simulation_driver.hpp`): thin Time loop matching `Simulator::step` (on_start at increment 0, next, conditions, physics, save). Tungsten/aluminum CPU and GPU ETD sessions use it. Gen-1 `Simulator` remains.
- M9 `apps/common/` (`openpfc_apps_common`): shared CLI helpers, MPI timing/SUM reduce, and rank-0 XY gather. heat3d / wave2d / allen_cahn / kobayashi keep per-app `RunConfig` and usage text.
- M9 Kobayashi OpenMP engine is the single-rank `FDPaddedCPUStack` path with OpenMP over owned cells (torus wrap indexing retired). Shared host stencils with `kobayashi_fd_manual`. Thread-parity 1 vs 4 retained; 32²/4-step HEX pinned against MPI nproc=1.
- M9 aluminum 5-step SeedGridFCC golden: `MovingFrameMeanFieldETDSystem` vs Gen-1 ≤1e-10 on 32³. Free-energy density formula pinned vs Gen-1; `last_free_energy_sum` is finite on the golden run. Pre-M0 sumsq norms unchanged.
- M9 `FDPaddedCPUStack` (`kernel/simulation/stacks/fd_padded_cpu_stack.hpp`): host padded `Field` + `HaloExchange` twin of `FDGPUStack`. Kobayashi CPU driver allocates fields and the two halo groups from the stack. `FDCPUStack` remains unpadded `SparseExchange` for heat3d/wave2d.
- M9 `FDGPUStack` (`runtime/gpu/fd_gpu_stack.hpp`): padded device `Field` + `HaloExchange` + extra-field/`make_exchange` factory. Kobayashi CUDA/HIP drivers allocate fields and the two halo groups from the stack.
- M9 GPU `DeviceMovingFrameMeanFieldETDSystem` (`runtime/gpu/moving_frame_mean_field_etd_gpu.hpp`) plus `AluminumETDGPUSession` and A/B binaries `aluminum_etd_cuda` / `aluminum_etd_hip`. Host vs device ≤1e-10 (`CUDA_MovingFrameETD` / `CUDA_AluminumETD`; HIP twins). Gen-1 `aluminumNew` remains.
- M9 host `MovingFrameMeanFieldETDSystem` (`kernel/simulation/moving_frame_mean_field_etd.hpp`): \(\chi(k)\) and \(P(k)\) iFFTs, \(N(\psi,\psi_{\mathrm{MF}},P*\psi,T_{\mathrm{var}}(x,t))\), shared ETD cache, free-energy `integrate_owned`. A/B CPU binary `aluminum_etd` (`AluminumETDSession`). Gen-1 `aluminumNew` remains.
- M9 start: `aluminum::AluminumPhysics` (`apps/aluminumNew/include/aluminum/aluminum_physics.hpp`) plus `MovingFrameMeanFieldETDPhysics` (`filter_mf`, `correlation_kernel` \(P(k)\), \(N(\psi,\psi_{\mathrm{MF}},P*\psi,T_{\mathrm{var}})\), free-energy density). Schema keys match `aluminumNew.json`. Gen-1 `Aluminum` remains.
- TungstenPhysics ETD weights (zero mode, near-zero `opCk`, long-dt) pinned against `legacy_etd_weights_for_mode`.
- Pre-M0 App-GPU-IC re-pointed at `TungstenETDGPUSession`: JSON `single_seed` IC, two ETD steps, device vs host ≤1e-10 (`HIP_TungstenETD` / `CUDA_TungstenETD`). Host session vs Gen-1 with the same IC.
- Type names spell abbreviations in full caps: `CPUFFT`, `CPUTag`, `HIPSpace`, `CUDASpace`, `GPUSpectralStack`, `FDCPUStack`, `ETD1Stepper`, `SpectralETDSystem`, `TungstenETDSession` (was `CpuFft` / `CpuTag` / `HipSpace` / `GpuSpectralStack` / `FdCpuStack` / `Etd1Stepper` / `…Etd…`).
- Tungsten golden A/B: 1-rank 8³/100 ETD steps and 4-rank 16³/20 steps (`tungsten-golden-4rank`) compare `TungstenETDSession` to Gen-1 within 1e-10. Pre-M0 dump was never captured.
- Tungsten ETD sessions write binary `psi` dumps on `Time::do_save()` from JSON `fields[]` (same schedule as Gen-1 `Simulator`).
- M8 A/B GPU session `TungstenETDGPUSession` (`GPUSpectralStack` + `DeviceSpectralMeanFieldETDSystem`). Binaries `tungsten_etd_hip` / `tungsten_etd_cuda`. HIP/CUDA tests `HIP_TungstenETD` / `CUDA_TungstenETD` vs host session within 1e-10.
- M8 A/B CPU binary `tungsten_etd`: JSON → `TungstenETDSession` (`SpectralCPUStack` + `TungstenPhysics` + `SpectralMeanFieldETDSystem`). Gen-1 `tungsten` remains. Constant IC two-step parity vs Gen-1. Writers not wired yet.
- `GPUSpectralStack<MemorySpace>` (`runtime/gpu/gpu_spectral_stack.hpp`): device counterpart of kernel `SpectralCPUStack` (Domain + Decomposition + `IDeviceFFT` + `Field<double, MemorySpace>`). HIP/CUDA tests `HIP_GPUSpectralStack` / `CUDA_GPUSpectralStack`. Lives in runtime because device FFT factories are runtime.
- Tungsten CPU/CUDA/HIP `from_json` share `tungsten::apply_tungsten_json` (`ParameterSchema` parse + setters). Frontend `ParameterValidator` still prints the summary.
- Device `DeviceSpectralMeanFieldETDSystem<Physics, MemorySpace>` (`runtime/gpu/spectral_mean_field_etd_gpu.hpp`): `IDeviceFFT` choreography, host `N(ψ,ψ_MF)`, device `χ(k)` multiply and ETD1 with `k_lap*phi1` weights. HIP/CUDA tests `HIP_SpectralMeanFieldETD` / `CUDA_SpectralMeanFieldETD` vs host within 1e-10.
- M8 start: `tungsten::TungstenPhysics` (`apps/tungsten/include/tungsten/tungsten_physics.hpp`, 153 lines) plus host `SpectralMeanFieldETDSystem` (mean-field filter, two-argument N, `k_lap*phi1` ETD weights). One-step parity vs Gen-1 `Tungsten` and JSON schema round-trip.
- Toy Swift–Hohenberg physics in one header (`tests/fixtures/swift_hohenberg.hpp`, 80 lines): schema + `rhs(t, SHGrads)` + spectral `linear_symbol`/`nonlinearity`. Three-way CPU test: Gen-1 `Model` ETD, point-wise SpectralGradient Euler, and `SpectralETDSystem`. Descriptor HIP parity is `HIP_SpectralETD`.
- Observable reduction (`kernel/simulation/observable_reduce.hpp`): owned-cell sum × cell volume, `MPI_Allreduce` SUM. Constant-field volume and 4-rank vs 1-rank Gaussian discrete sum to 1e-12; HIP/CUDA `HIP_ObservableReduce` / `CUDA_ObservableReduce`.
- Device `DeviceSpectralETDSystem<Physics, MemorySpace>` (`runtime/gpu/spectral_etd_system_gpu.hpp`): `IDeviceFFT` choreography, host `N(psi)`, device ETD1 combine. HIP/CUDA tests `HIP_SpectralETD` / `CUDA_SpectralETD` vs host within 1e-10.
- Adapter **A1** `pfc::compat::LegacyModelPhysics` wraps a Gen-1 `Model&` as `SteppablePhysics` (delegates `step`). Adapter **A2** `Simulator::step_with_physics` has a dedicated parity test (`[a1]`/`[a2]`): mock call counts plus bitwise diffusion-fixture trajectories.
- Host `SpectralETDSystem<Physics>` (`kernel/simulation/spectral_etd_system.hpp`): owns `psi_hat`/`N`/`N_hat` on `SimulationState`, prepares `L(k)` with `for_each_kpoint`, advances via `ETD1Stepper` on the complex hat, optional 2/3-rule dealias. Device `IDeviceFFT` path is a later M7 slice.
- `ParameterSchema<Params>` (`kernel/simulation/parameter_schema.hpp`): member-pointer bindings generate parse/`from_json`, collect validation errors in `format_config_error` form, and emit a docs table. `HasParameterSchema` is the physics hook. Frontend `ParameterValidator` is unchanged.
- Physics concepts in `kernel/simulation/physics_concepts.hpp`: `DeclaresFields` / `add_declared_field`, `PointwiseRhs` (`rhs(t, G)`), `SpectralDiagonalPhysics` (`linear_symbol(k)` + `nonlinearity(psi)`), and combined `PointwisePhysics` / `SpectralETDPhysics`. `HasParameters` is the nested-type hook for `ParameterSchema`.
- `PackedEulerStepper<Rhs, Scalars...>` advances a mixed-scalar pack (e.g. `double` + `std::complex<double>`). `PackedStageFunction` / `PackedStepAttempt` / `commit_step_attempt` cover that result. Variadic `commit_step_attempt` takes the `PackedStepAttempt` first (pack last, for deduction); two-field form is buffers-first like `MultiStepAttemptResult`. Homogeneous `MultiEulerStepper<Rhs, N, Scalar>` is unchanged.
- Multi-field steppers (`MultiEuler`, `MultiETD1`, `MultiImex`, `MultiExplicitRK`) accept homogeneous `HostFieldPack` of host fields via `vec()`, in addition to `std::vector<Scalar>` packs. `HostFieldPack<Scalar, Fs...>` is in `state_concepts.hpp`.
- Stepper host-field overloads are constrained by `pfc::field::HostFieldState<F, Scalar>` (`state_concepts.hpp`): `Field` plus host `vec()`. `pfc::data::Field<Scalar>` still models it.
- Device-resident ETD1 combine: host `pfc::integrator::apply_etd1_update` plus `apply_etd1_update_{cuda,hip}` over device pointers (real and complex). Real two-term combine added to the GPU elementwise kernels. HIP/CUDA tests `HIP_ETD1Apply` / `CUDA_ETD1Apply`.
- `ImexEulerStepper` / `MultiImexEulerStepper` take `Scalar` (default `double`) and host `Field<Scalar>`. Real diagonal implicit coeffs still live in `operator_context` as `vector<double>`.
- `ExplicitRKStepper` / `MultiExplicitRKStepper` and `EmbeddedRKStepper` take `Scalar` (default `double`) and host `Field<Scalar>`. Real Butcher weights stay `ButcherTableau<double>`.
- `RK2HeunStepper` and `RK3HeunStepper` take `Scalar` (default `double`) and host `Field<Scalar>`.
- `EulerStepper<Rhs, Scalar>` and `MultiEulerStepper<Rhs, N, Scalar>` advance `double` or `std::complex<double>` (default `double`). Host `Field<Scalar>` overloads stay; vector path remains.
- `MultiETD1Stepper<Rhs, N, Scalar>` advances `double` or `std::complex<double>` packs with real per-field ETD coefficients. `MultiStageFunction` and `MultiStepAttemptResult` take an optional `Scalar` (default `double`).
- Deleted unused `fd_stencils.hpp` back-compat shims (`detail::EvenFDStencil1d`, `fd_even_order_lookup`).
- One workspace type: `pfc::sim::steppers::StageWorkspace<T>` is an alias of `pfc::integrator::Workspace<T>` (stage vectors + scratch, move-only, `clear()`/`reset()`).
- One `StageContext`: `pfc::sim::StageContext` is an alias of `pfc::integrator::StageContext`. The struct carries integrator flags plus optional `ExecutionService*`; solvers use `time` (was `evaluation_time`) and `service()`.
- `MultiEulerStepper` isolates candidates via `attempt` (any N). Public `euler_attempt.hpp` is deleted.
- `Time` stores `pfc::sim::steppers::RKIntegratorMethod`. The two-value `IntegratorMethod` enum in `time.hpp` is removed; JSON `from_json<Time>` accepts the full RK method token set.
- `AdaptiveTimeController` maps error evidence (or an embedded-error vector) to accept/reject and next `dt`, then commits or rejects a `Time` attempt. Example `21_adaptive_stepping` and `test_adaptive_controller`.
- `ETD1Stepper<Rhs, Scalar>` advances `double` or `std::complex<double>` fields with real ETD coefficients. `StepAttempt<Scalar>` is the general result; `StepAttemptResult` is `StepAttempt<double>`.
- All seven single-field leaves (`Euler`, `RK2`/`RK3` Heun, `ExplicitRK`, `EmbeddedRK`, `ImexEuler`, `ETD1`) accept host `pfc::data::Field<double>` via `vec()`. Vector path remains.
- Deleted unused `IntegratorBase` / `EulerIntegrator` / `RK2HeunIntegrator` and the on-hold `IntegratorResult` DTO (and their unit tests).
- Non-diagonal dense `SolveFunction` mock under `ImexEulerStepper` (`imex_euler_nondiagonal_dense_solve`).
- `MultiStageFunction<Rhs, N>` (default N=2) and `MultiETD1Stepper` accept any N≥1 via a variadic `attempt`.
- `RK3HeunStepper`, `ExplicitRKStepper`, `EmbeddedRKStepper`, `ImexEulerStepper`, and `ETD1Stepper` return `StepAttemptResult` (multi-field: `MultiStepAttemptResult`). Dropped `EmbeddedStepAttemptResult`, `ImexStepAttempt`, and `ETD1StepAttempt`. Embedded extras are `u_high()` / `u_low()` / `error()` / `last_rhs_evals()`; IMEX extras are `last_solve_*`; ETD extras are `last_reason()`.
- LUMI/Cray GNU compile of M5 FFT/ETD headers: `SpectralExpCoefficientCache<>` at call sites, vector overload of `fill_spectral_exp_coeffs`, no `override`+`requires` on `FFT_Impl`, dummy `device_fft_buffers` fallback, and `r2c_direction` dropped from comm-only `create`/`create_cuda`/`create_hip` (MPI_Comm is `int` on Cray).
- `EulerStepper` and `RK2HeunStepper` implement `attempt` / `commit_step_attempt` (`StepAttemptResult`). In-place `step()` is that pair. ADR 0003 accepted as the M6 protocol.
- Optional 2/3-rule dealiasing mask (`kernel/fft/dealias.hpp`), off by default. Documented in `docs/science/numerics_limits.md`.
- `SpectralGradient` binds a `FieldView` and zeros odd-derivative symbols at the even-grid Nyquist mode (Audit K1).
- `k_component_odd` / `is_nyquist_index`; device kernels use `runtime/gpu/kspace_iterator_gpu.hpp`.
- `spectral_exp_coeffs` / `SpectralExpCoefficientCache` are templated on `Real` (default `double`).
- `pfc::fft::kspace::for_each_kpoint` walks a local FFT outbox with `(idx, kx, ky, kz, i, j, k)`. `SpectralGradient` builds its operator tables with it.
- Convenience FFT factories (`create`, `create_with_backend`, `create_cuda`, `create_hip`) take an optional `r2c_direction` (default 0).
- GPU FFT workspaces allocate each precision on first use (ADR 0006) instead of always owning both float and double buffers.
- `pfc::fft::IHostFFT` and `pfc::fft::IDeviceFFT<MemorySpace>` (ADR 0005). `create_with_backend` rejects CUDA/HIP at construction; `create_cuda` / `create_hip` return `FFT_CUDA` / `FFT_HIP` which implement `IDeviceFFT`.
- Removed unused `apps/kobayashi/src/cuda/kobayashi_batched_halo.hpp` after the CUDA driver moved onto `HaloExchange`.
- FD MPI leftover tests (`test_fd_heat_mpi`, `test_fd_xy_mpi`, `test_halo_exchange_driver`) use `HaloExchange` instead of `HaloExchanger` / `PersistentHaloExchanger`. `test_sparse_halo_exchange` uses `SparseExchange`.
- `HaloExchangeOptions::directions` — optional `HaloDirectionSet` (empty means `Axes3D()` for Faces, `Full3D()` for Full). Kobayashi CPU/HIP/CUDA pass `Axes2D()` so the `nz=1` slab skips ±Z.
- Kobayashi CUDA driver uses two multi-field `pfc::comm::HaloExchange<CUDASpace>` objects on device-resident Fields (state then aux), matching HIP/CPU. Execute on tohtori.
- `scripts/build.sh --machine=lumi` — LUMI HIP/ROCm path: loads `LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath` and `heffte-rocm`, configures on the login node, then submits compile + ctest to `standard-g` (default) or `dev-g` under Slurm account `project_462001519`. CUDA is rejected on LUMI.
- wave2d CUDA driver and CPU-vs-CUDA test use `SparseExchange<CUDASpace>` with device y-face BC patches. Execute on tohtori.
- Allen–Cahn CUDA driver and CPU-vs-CUDA test use `SparseExchange<CUDASpace>` (device gather/MPI/face recv). Execute on tohtori.
- FD MPI integration tests (`test_fd_xy_mpi`, `test_fd_heat_mpi`) use `SparseExchange` for separated-face Laplacian cases.
- 4-rank `HaloExchange` mode suite: host blocking == split-phase == multi-field batch; Full 26-direction corners; HIP 4-rank Faces + Full (`test_comm_halo_exchange_modes.cpp` / `test_comm_halo_exchange_gpu.cpp`). Persistent multi-rank is still 1-rank only on LUMI.
- In-repo min-surface brick splitter (`brick_split.hpp`) replaces `heffte::split_world` / `proc_setup_min_surface` in the decomposition TU (ADR 0007). Equivalence is pinned against live HeFFTe in `test_brick_split.cpp`.
- Neighbour-direction agreement Allgather is off by default in release builds (`NDEBUG`); set `OPENPFC_VALIDATE_NEIGHBOUR_AGREEMENT=1` to force it, or `=0` to skip it in debug.
- wave2d host tests and `step_wave_separated_order2_cpu` use `HaloExchange` / `SparseExchange` instead of `PaddedHaloExchanger` / `SparseHaloExchanger`.
- Examples `15_finite_difference_heat` and `19_explicit_stepper_fd` use `SparseExchange` / `HaloExchange` instead of the old exchanger classes.
- heat3d Catch2 manual/scratch paths use `pfc::comm::HaloExchange<HostSpace>` instead of `PaddedHaloExchanger`.
- `StagePreparationService` binds `pfc::comm::HaloExchange<HostSpace>` instead of `PaddedHaloExchanger`.
- wave2d HIP driver and CPU-vs-HIP test use `pfc::comm::SparseExchange<HIPSpace>` on unpadded device Fields; y-face BC patches and Dirichlet walls stay on device (no per-step full-field D2H).
- Allen–Cahn HIP driver and CPU-vs-HIP test use `pfc::comm::SparseExchange<HIPSpace>` on an unpadded device Field; the kernel reads device `face_recv_ptrs()` (no per-step full-field D2H).
- Kobayashi HIP driver uses two multi-field `pfc::comm::HaloExchange<HIPSpace>` objects on device-resident Fields (state then aux) instead of per-step `hipMemcpy` + host `PaddedHaloExchanger`.
- Kobayashi CPU driver uses two multi-field `pfc::comm::HaloExchange<HostSpace>` objects (state then aux) instead of six `PaddedHaloExchanger`s.
- Allen–Cahn CPU driver and CPU comparison-test path use `pfc::comm::SparseExchange<HostSpace>` instead of `SparseHaloExchanger`.
- wave2d CPU drivers (`wave2d_fd`, `wave2d_fd_manual`) use `pfc::comm::HaloExchange<HostSpace>` instead of `PaddedHaloExchanger`.
- heat3d CPU drivers (`heat3d_fd`, `heat3d_fd_manual`, `heat3d_fd_scratch`) use `pfc::comm::HaloExchange<HostSpace>` instead of `PaddedHaloExchanger`.
- `FDCPUStack` uses `pfc::comm::SparseExchange<HostSpace>` instead of constructing `SparseHaloExchanger` itself.
- `pfc::comm::SparseExchange<HostSpace>` — host index-set facade over `SparseHaloExchanger` (structured `make_structured_halos` or a custom `RemoteHalo` list; `exchange`/`start`/`finish`). Device `SparseExchange<CUDASpace/HIPSpace>` gathers, posts device-pointer MPI, and scatters without a full-field D2H. CUDA execution: verify on tohtori.
- Device halo default transport is pack-to-contiguous + device-pointer MPI when GPU-aware (`PaddedDeviceHaloExchanger` / `FullPaddedDeviceHalo`). `OPENPFC_{CUDA,HIP}_USE_SUBARRAY_HALO=1` restores derived-type MPI; `*_FORCE_PACKED_HALO=1` still host-stages. Post-exchange sync is stream-scoped, not `cudaDeviceSynchronize` / `hipDeviceSynchronize`.
- `pfc::comm::HaloExchange<CUDASpace/HIPSpace>` — device facade in `runtime/gpu/comm_halo_exchange_gpu.hpp` (Faces / Full, blocking `exchange()`, residency sync). Persistent and split-phase fail closed (device exchangers are blocking-only). CUDA execution: verify on tohtori.
- `pfc::gpu::runtime_mpi_gpu_aware()` — shared GPU-aware MPI decision (assume override, Open MPI MPIX query, Cray `MPICH_GPU_SUPPORT_ENABLED=1`, optional device-pointer probe). Halo and SparseVector exchange use this instead of an Open-MPI-only query.
- `pfc::mpi::communicator::duplicate()` — opt-in `MPI_Comm_dup` isolation.
- `pfc::comm::HaloExchange<HostSpace>` — unified host halo facade (Faces/Full, `exchange`/`start`/`finish`, optional persistent, multi-field tag blocks). Device specialization is remaining M4 work.
- `pfc::halo` geometry helpers in `kernel/decomposition/halo_geometry.hpp` (M4): face slots, `opposite_slot` / `opposite_direction`, per-field MPI tag blocks, and padded send/recv slabs. `halo_directions.hpp` uses these instead of its own slot/tag copies.
- `pfc::Domain` — canonical coordinate/geometry type replacing the templated `World`
- `pfc::Box3i` — single canonical inclusive integer index box
- `pfc::data::Field<T, MemorySpace>` — canonical owning field container unifying LocalField/PaddedBrick
- `include/openpfc/runtime/gpu/gpu_api.hpp` — vendor shim (`gpuMalloc`, `gpuMemcpyAsync`, `gpuStream_t`, `GPU_CHECK`, `GPU_LAUNCH_KERNEL`) selected by CUDA vs HIP (`OPENPFC_HD` already covers `__HIPCC__` in `host_device.hpp`)
- `include/openpfc/runtime/gpu/databuffer_gpu.hpp` — single-source GPU `DataBuffer` for CUDA and HIP; `databuffer_cuda.hpp` / `databuffer_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/deep_copy_gpu.hpp` — single-source GPU `deep_copy(buffer, scalar)` fill; `deep_copy_cuda.hpp` / `deep_copy_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/memory_space_gpu.hpp` — single-source `CUDASpace` / `HIPSpace`; `memory_space_cuda.hpp` / `memory_space_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/backend_tags_gpu.hpp` — single-source `CUDATag` / `HIPTag`; `backend_tags_cuda.hpp` / `backend_tags_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/memory_traits_gpu.hpp` — single-source GPU `backend_traits`; `memory_traits_cuda.hpp` / `memory_traits_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/gpu_check.hpp` — single-source `cuda_check` / `hip_check`; `cuda_check.hpp` / `hip_check.hpp` are thin includes
- `include/openpfc/runtime/gpu/exchange_gpu.hpp` — single-source GPU SparseVector MPI exchange; `exchange_cuda.hpp` / `exchange_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/fd_gradient_device_gpu.hpp` — single-source GPU FD gradient evaluator (CUDA composite + HIP padded-Field factory); vendor `fd_gradient_device.hpp` re-export into `pfc::cuda` / `pfc::hip`
- `include/openpfc/runtime/gpu/for_each_interior_device_gpu.hpp` — single-source GPU interior driver (single-field + multi-field N=2–4 + autotune hook); vendor headers re-export into `pfc::sim::cuda` / `pfc::sim::hip`
- `include/openpfc/runtime/gpu/sparse_vector_gpu.hpp` — single-source GPU SparseVector copy-to-device; `sparse_vector_cuda.hpp` / `sparse_vector_hip.hpp` are thin includes
- `include/openpfc/runtime/gpu/sparse_vector_ops_gpu.hpp` — single-source GPU SparseVector gather/scatter; vendor `sparse_vector_ops.hpp` is a thin include; kernels live in `src/openpfc/runtime/gpu/sparse_vector_ops_gpu.inc` compiled from `sparse_vector_ops.cu` / `.hip`
- `include/openpfc/runtime/gpu/padded_device_halo_exchange_gpu.hpp` — single-source GPU 6-face padded device halo exchanger (HIP Field overloads stamped for CUDA); vendor `padded_device_halo_exchange.hpp` are thin includes; env/timer names stay `OPENPFC_CUDA_*` / `OPENPFC_HIP_*`
- `include/openpfc/runtime/gpu/full_padded_device_halo_gpu.hpp` — single-source GPU 26-direction padded device halo (CUDA `m_use_full_widening` stamped for HIP); vendor `full_padded_device_halo.hpp` are thin includes
- `src/openpfc/runtime/gpu/padded_halo_faces_gpu.inc` — single-source padded face pack/unpack kernels; compiled from `padded_halo_faces.cu` and `.hip`
- `include/openpfc/runtime/gpu/elementwise_ops_gpu.hpp` — generic device elementwise ops (complex×real multiply, two-term diagonal combine, axpy-style fill `out = alpha * x + beta`); compiled from `src/openpfc/runtime/gpu/elementwise_ops.cu` / `.hip`
- HIP-parity gpu_validation tests: `test_multi_field_device.hip` and `test_composite_gradient_pod_size_hip.hip` (HIP twins of the CUDA-only multi-field `for_each_interior_device` and composite-gradient POD layout cases)
- HIP FFT unit test `tests/unit/runtime/gpu/test_fft_hip.cpp` (`HIP_FFT`), gated on `OpenPFC_ENABLE_HIP_SPECTRAL` — twin of CUDA `test_fft_cuda.cpp` using `pfc::fft::create_hip` and `HIPTag` DataBuffers
- HIP FFT integration roundtrip `tests/integration/scenarios/gpu_validation/test_hip_roundtrip.cpp` — twin of CUDA `test_cuda_roundtrip.cpp` (float/double DataBuffer forward/backward)
- HIP CPU-vs-GPU Laplacian integration tests `test_hip_vs_cpu_laplacian.cpp` and `test_hip_vs_cpu_laplacian_mpi.cpp` — twins of the CUDA Laplacian gpu_validation scenarios
- HIP vs CPU diffusion smoke `test_hip_vs_cpu.cpp` is compiled into `openpfc-tests` and constructs `create_hip` (previously an unwired stub)
- HIP backend instantiation smoke in `test_gpu_backend_instantiation.cpp` — separate Catch2 case so a CUDA skip cannot hide HIP; compares `create_hip` inbox/outbox sizes to the CPU FFT
- `examples/fft_backend_benchmark` benchmarks HIP (rocFFT) as well as CUDA, using `runtime/gpu/` DataBuffer/tags
- HIP `FullPaddedDeviceHalo` 26-direction integration twin `test_full_padded_device_halo_hip.cpp` of the CUDA `test_full_padded_device_halo.cpp` cases
- `scripts/check_gpu_memcpy_single_source.sh` — CI guard that `cudaMemcpy` / `hipMemcpy` in `include/` and `src/` stay under `runtime/gpu/`
- `pfc::fft::Backend::HIP` and `backend_from_string("hip"` / `"rocm")` when `OpenPFC_ENABLE_HIP_SPECTRAL` is on; JSON `from_json<fft::Backend>` and `create_with_backend` accept HIP the same way they already accept CUDA

### Fixed

- Device-link `openpfc_gpu_kernels` (`CUDA_RESOLVE_DEVICE_SYMBOLS ON`) so CUDA
  executables no longer fail host-link on undefined
  `__cudaRegisterLinkedBinary_*` from the static archive (Tohtori/nvcc 13.1).
- CUDA-only `backend_from_string` test called `pfc::runtime.backend_from_string`
  instead of `pfc::runtime::`.
- Brick-split HeFFTe compare skips `(size, nparts)` pairs where HeFFTe 2.4.1
  `proc_setup_min_surface` finds no grid and `assert`s (Debug/CUDA HeFFTe).
- HIP GPU autotune device queries used the deprecated `gcnArch` field (an `int` on ROCm 6, so `std::string(prop.gcnArch)` was not an architecture name) and `pciDeviceId` (the struct spells `pciDeviceID`). They now use `gcnArchName` and `pciDeviceID`; the HIP autotune test requires a `gfx` prefix when a device is present.
- GPU autotune unit test includes Catch2 string matchers so `REQUIRE_THROWS_WITH` compiles.
- HIP FFT unit test did not link HeFFTe, so `heffte.h` was not on the include path. Both CUDA and HIP FFT unit-test binaries now link `Heffte::Heffte`.
- HIP configure failed: `openpfc_gpu_compile_defs` is an INTERFACE library, but GPU macros were applied with `PUBLIC` (`target_compile_definitions` only allows `INTERFACE` on INTERFACE targets).
- HIP configure failed at export: FetchContent `nlohmann_json` is not in `OpenPFCTargets`. The library now uses that header-only tree via `BUILD_INTERFACE` includes instead of linking the FetchContent target. GPU kernel libraries get the same include path so autotune JSON parses under HIP.
- Removed `tests/unit/kernel/data/test_field.cpp`; it included the deleted Gen-1 `kernel/data/field.hpp`. Canonical coverage is `test_grid_field.cpp`.
- Removed `tests/unit/kernel/data/test_multi_index.cpp`; `multi_index.hpp` was deleted in M2.
- **Sticky CUDA/HIP error from a handled allocation failure:** `DataBuffer`'s CUDA/HIP specializations checked `cudaMalloc`/`hipMalloc`'s return value but never called `cudaGetLastError()`/`hipGetLastError()` to clear the driver's sticky error flag before throwing. A deliberately-triggered allocation failure (e.g. in a resize-failure test) left that flag poisoned for the rest of the process, later misattributed to an unrelated kernel launch elsewhere as a false "out of memory".
- **`FullPaddedDeviceHalo` skipped its corner/edge fill without GPU-aware MPI:** the 3-pass widening algorithm was gated entirely behind GPU-aware MPI availability, even for self-only periodic axes that never touch MPI device pointers at all. It now runs the full algorithm whenever no active axis has a real (non-self) neighbor, or GPU-aware MPI is genuinely available; only real cross-rank axes without GPU-aware MPI fall back to the face-only path.
- **`test_stage_preparation.cpp` checked halo cells `PaddedHaloExchanger` never fills:** its comparison helpers walked the full padded range on orthogonal axes, but the exchanger is documented face-only (corners/edges untouched). Restricted the checks to the owned range, matching the already-correct pattern in `test_padded_halo_exchange.cpp`.
- Missing `pfc::ui::from_json<Domain>` specialization (declared, never defined) caused a link error in any CUDA/HIP app driver calling it directly instead of through `SpectralSimulationSession`.
- `TungstenCUDA`/`TungstenHIP` had no constructor matching the generic `(fft::IFFT&, const World&, MPI_Comm)` session-wiring signature, only failing to compile when a CUDA/HIP app target was actually built.

### Changed

- Example `09_parallel_fft_high_level` stores the FFT outbox in `pfc::data::Field<std::complex<double>>` instead of legacy `Array`.
- `pfc::field::for_each*` in `brick_iteration.hpp` no longer has `PaddedBrick` overloads; padded `pfc::data::Field` is the only container (tests already used Field).
- `pfc::communication::PaddedHaloExchanger` no longer binds `PaddedBrick`; padded `pfc::data::Field` (or explicit `Box3i` + `Domain`) is the only container binding. Callers already used the Field constructors.
- HIP `pfc::hip::FDGradientDevice` factory binds `pfc::data::Field` (any memory space) instead of legacy `PaddedBrick`, matching the CUDA twin. Unpadded Fields (`storage_halo == 0`) are rejected; padded `Field<double, HIPSpace>` is the device path.
- CPU `pfc::gradient::FDGradient` no longer has a `PaddedBrick` constructor or factory; padded `pfc::data::Field` is the only container binding (`test_multi_field_device.cu` migrated).
- `scripts/build.sh` (Tohtori) auto-detects a custom CUDA-aware Open MPI build (see `scripts/build_tohtori.sh --cuda`) and uses it in place of the site `openmpi/5.0.10` module when present, defaulting `MPI_CUDA_AWARE` to `ON` in that case. Without it, the default stays `OFF`: the site module links a UCX built without `--with-cuda`, and passing device pointers to it segfaults despite Open MPI's own `MPIX_Query_cuda_support()` probe claiming support.
- HIP packed-halo pinned host buffers use `hipHostMalloc` / `hipHostFree` instead of the deprecated `hipMallocHost` / `hipFreeHost`.
- `OpenPFC_ENABLE_CUDA` / `OpenPFC_ENABLE_HIP` and `OpenPFC_MPI_CUDA_AWARE` / `OpenPFC_MPI_HIP_AWARE` are PUBLIC compile definitions on `openpfc` (and the vendor kernel libraries) instead of directory-scope `add_compile_definitions`, so `find_package(OpenPFC)` consumers see the same macros as the in-tree build.
- Device kernel TUs (`sparse_vector_ops.cu/.hip`, `padded_halo_faces.cu/.hip`) live under `src/openpfc/runtime/gpu/` instead of `include/` (and instead of `src/openpfc/runtime/cuda/` for the CUDA halo-face TU). CUDA `padded_halo_faces.cu` remains linked per executable because of separable-compilation device-link.
- `deep_copy(buffer, scalar)` for CUDA/HIP `DataBuffer` runs a device fill kernel (`runtime/gpu/fill_gpu`) instead of staging a host vector. Device scalar fill supports `float` and `double`; include `deep_copy_gpu.hpp` (or the vendor shim). Device `View` fill and View-to-View device copies are not provided. GPU fill tests cover `DataBuffer` and raw `fill_*_impl` pointers.
- Tungsten CUDA/HIP `multiply_complex_real` and `apply_time_integration` call `runtime/gpu/elementwise_ops_gpu` instead of duplicating those kernels. Nonlinear and stabilization kernels stay in the Tungsten TUs.
- `cmake --install` no longer ships FetchContent `nlohmann_json` headers or the `openpfc-tests` binary. `find_package(OpenPFC)` always `find_dependency(nlohmann_json)`.
- `OpenPFC_ENABLE_GPU_AUTOTUNING` is a PUBLIC compile definition on `openpfc` (and the vendor kernel libraries) instead of directory-scope `add_compile_definitions`.
- Heat3D/Wave2D structural `rhs()` tests renamed off `vs_legacy_step` (they were never numerical vs-legacy baselines). Checkpoint headers/docs state that restart loading is not implemented.
- Bare `cmake` on a single-config generator defaults to `RelWithDebInfo`, not Debug (`cmake/ProjectSetup.cmake`; documented in `INSTALL.md`).
- `tests/benchmarks/README.md` no longer lists machine-specific nanosecond/millisecond claims; measure locally in Release.
- State-access design docs describe GPU storage as `pfc::core::DataBuffer` (`CUDATag`/`HIPTag`); `pfc::gpu::GPUVector` is gone.
- `NAN_CHECK_ENABLED` is a PUBLIC compile definition on `openpfc` when Debug is selected or `OpenPFC_ENABLE_NAN_CHECK=ON`, instead of directory-scope `add_compile_definitions`.
- GPU SparseVector host-to-device copy failures report `"HIP copy failed: …"` with the runtime string, matching CUDA; unused duplicate `sparse_vector_ops_cuda.hpp` / `sparse_vector_ops_hip.hpp` shims removed (`sparse_vector_ops.hpp` remains).
- `for_each_interior_device` launch/sync failures use `GPU_CHECK` (`"GPU error: …"`) instead of a per-overload hand-rolled check. Kernel `.inc` files still prefix CUDA/HIP; co-enabled TUs still use `cuda_check` / `hip_check`.
- CUDA `openpfc_gpu_kernels` and HIP `openpfc_hip_kernels` share one CMake source list (`sparse_vector_ops`, `fill`, `elementwise_ops`). HIP still adds `padded_halo_faces.hip`; CUDA halo-face kernels stay linked per executable.
- GPU kernel `.inc` sources live under `src/openpfc/runtime/gpu/` next to the vendor TUs that include them (not under `include/`; not installed).
- Architecture, styleguide, and `DataBuffer` diagnostics name `runtime/gpu/` as the CUDA/HIP implementation layer; vendor `runtime/cuda` / `runtime/hip` trees are documented as thin includes plus FFT until M5.
- Shared Tungsten GPU headers and vendor FFT headers include `runtime/gpu/` DataBuffer/tags directly instead of hopping through CUDA/HIP shims.
- Dual CUDA/HIP unit tests include `runtime/gpu/` SparseVector exchange and DataBuffer headers instead of duplicated vendor shims (`test_sparse_vector_exchange_device.cpp` included).
- CUDA/HIP fail-closed SparseVector exchange tests include `runtime/gpu/` exchange, SparseVector, and check headers instead of vendor shims (keep native `cuda_check` / `hip_check` calls).
- CUDA/HIP SparseVector unit tests (`test_sparsevector_cuda.cpp`, `test_sparsevector_hip.cpp`) include `runtime/gpu/` SparseVector headers instead of vendor shims (keep native `cudaMemcpy` / `hipMemcpy`).
- CUDA/HIP padded device-halo tests (`test_padded_device_halo_self_wrap.cpp`, `test_padded_device_halo_self_wrap_hip.hip`, `test_full_padded_device_halo.cpp`) include `runtime/gpu/` halo headers instead of vendor shims
- HIP fd-gradient gpu_validation test includes `runtime/gpu/` DataBuffer and HIPSpace headers instead of vendor shims; vendor `fd_gradient_device.hpp` / `for_each_interior_device.hpp` re-exports stay (call sites use `pfc::hip::` / `pfc::sim::hip`)
- Kobayashi CUDA driver includes `runtime/gpu/` padded device-halo headers instead of the vendor shim (`pfc::cuda::PaddedDeviceHaloExchanger` stays; the GPU header already stamps it)
- `SimulationState` device-field compile coverage includes `HIPSpace` (CUDA twin already existed) and includes `runtime/gpu/` memory-space headers.
- SparseVector `on_host` coverage includes `HIPTag` (CUDA twin already existed) and includes `runtime/gpu/` SparseVector headers.
- Halo-exchange concept docs list canonical GPU sources under `runtime/gpu/` (vendor CUDA/HIP headers are thin includes); HIP packed-halo env and kernel-library split are documented alongside CUDA.
- FD/halo Doxygen `@see` comments and per-point-gradient docs point at `runtime/gpu/` device twins, not CUDA-only vendor headers.

### Removed

- Public `pfc::SparseHaloExchanger`. Host sparse exchange is `pfc::comm::SparseExchange`. The implementation is `pfc::comm::detail::HostSparseHalo` in `sparse_halo_exchange.hpp`.
- Public `pfc::cuda::PaddedDeviceHaloExchanger` / `pfc::hip::PaddedDeviceHaloExchanger` and `pfc::cuda::FullPaddedDeviceHalo` / `pfc::hip::FullPaddedDeviceHalo`. Device Faces/Full exchange is `pfc::comm::HaloExchange<CUDASpace/HIPSpace>`. The implementations are `pfc::gpu::DeviceFacesHalo` / `pfc::gpu::DeviceFullHalo`.
- Public `pfc::PersistentHaloExchanger`. Host persistent Faces exchange is `pfc::comm::HaloExchange` with `HaloExchangeOptions::persistent`. The implementation is `pfc::comm::detail::HostPersistentFaces` in `halo_persistent.hpp`.
- Public `pfc::FullPaddedHaloExchanger` / `pfc::communication::FullPaddedHaloExchanger`. Host Full (26-direction) exchange is `pfc::comm::HaloExchange` with `HaloConnectivity::Full`. The implementation is `pfc::comm::detail::HostFullHalo` in `full_padded_halo_exchange.hpp`.
- Public `pfc::PaddedHaloExchanger` / `pfc::communication::PaddedHaloExchanger`. Host Faces exchange is `pfc::comm::HaloExchange`. The implementation is `pfc::comm::detail::HostFacesHalo` in `padded_halo_exchange.hpp`.
- Unpadded in-place `pfc::HaloExchanger` (`kernel/decomposition/halo_exchange.hpp`). It overwrote outermost owned cells and had no remaining callers. Use a padded `Field` + `pfc::comm::HaloExchange` (Faces) or `pfc::comm::SparseExchange` for separated cores.
- `pfc::field::LocalField` (`kernel/field/local_field.hpp`). Use `pfc::data::Field` with `field_from_subdomain_unpadded` (unpadded storage) or `field_from_inbox` for spectral inboxes. Stepper factories already bound `Field`.
- `pfc::field::PaddedBrick` (`kernel/field/padded_brick.hpp`). Use padded `pfc::data::Field` via `field_from_subdomain(decomp, rank, halo)`. Halo exchangers, FDGradient, and `brick_iteration` already bind Field.
- `pfc::DiscreteField` (`kernel/data/discrete_field.hpp`) and `pfc::interpolate`. Use `pfc::data::Field` with `coords()` / `apply()`. The quarantined DiscreteField unit tests were deleted with the type.
- `pfc::Array` (`kernel/data/array.hpp`). Use `pfc::data::Field`. The Array unit tests were deleted with the type.
- `pfc::field::Field<T>` (`kernel/data/field.hpp`). Use `pfc::data::Field`. Steppers, stacks, and factories already bound the canonical type; the functional container had no remaining callers.
- `pfc::field::make_legacy_modifier` (`kernel/field/legacy_adapter.hpp`). Wrap a lambda in a `FieldModifier` and call `pfc::field::apply` instead.
- `pfc::field::apply(Model&, name, fn)` and the matching `apply_with_time` / `apply_inplace*` Model overloads. Call `apply(get_real_field(m, name), get_world(m), get_fft(m), fn)` instead so `operations.hpp` no longer includes the simulation layer.
- `pfc::gpu::GPUVector` (`runtime/cuda/gpu_vector.hpp`), `kernels_simple` (`add_scalar` / `multiply_scalar`), and their CUDA unit tests. Use `pfc::core::DataBuffer` (or `pfc::data::Field`) for device storage.
- `pfc::create_mirror` / `pfc::create_mirror_view` (`kernel/execution/create_mirror.hpp`). Host `View` copies use `deep_copy`; device storage is `DataBuffer`.
- GPU View execution-space mapping (`runtime/gpu/view_gpu.hpp` and vendor `view_cuda.hpp` / `view_hip.hpp`). `View` is host-only; device storage is `DataBuffer`.
- GPU `parallel_for` / `fence` (`runtime/gpu/parallel_gpu.hpp` and vendor shims) and `Cuda`/`HIP` execution-space tags (`execution_space_gpu.hpp`). Host `parallel_for` remains Serial/OpenMP-only.
- Kokkos-facsimile `View`, host `parallel_for`/`fence`, `RangePolicy`/`MDRangePolicy`, layouts, `deep_copy` View overloads, `Serial`/`OpenMP` execution-space tags, and `tests/unit/kernel/execution/test_kokkos_like.cpp`. Device storage is `DataBuffer`; `deep_copy(buffer, scalar)` remains.
- GPU autotune demo keys `add_scalar` / `multiply_scalar` (registry + fallback defaults). Remaining defaults are `for_each_interior_3d`, `gather`, and `scatter`.

## [0.1.5] - 2026-07-23

Final stable 0.1.x release: a correctness and packaging pass ("Pre-M0
stabilization") completed before the breaking OpenPFC 0.2 architecture refactor.
This is the last release on the 0.1 architecture.

### Fixed (Pre-M0 stabilization — final 0.1.x correctness pass before the 0.2 refactor)

Each fix has a regression test; the full CPU suite (26 ctest batches + Python)
passes, and CUDA/HIP compile cleanly. GPU *runtime* behavior for the device-only
fixes is marked `// TODO: not tested` pending a GPU-node run.

- **FD dispatcher fail-closed (audit §4.3):** `field::fd::laplacian_interior(int order, …)` now throws `std::invalid_argument` on an unsupported order instead of silently doing nothing.
- **Periodicity honored (audit §4.4):** `world::create` / `from_bounds` now store the requested per-axis periodicity (was always all-periodic); added `world::get_periodic` / `world::is_periodic`.
- **Subdomain bounds (audit §4.5):** `world::get_lower_bounds` / `get_upper_bounds` now respect a subdomain's index offset instead of reporting the global origin.
- **Coordinate→index convention (audit §4.6):** `csys::to_index` now rounds (matching the documented `to_indices` contract and `DiscreteField`), not truncates.
- **Dead API removed (audit §4.8):** deleted the undefined `utils::compute_upper_bounds/compute_spacing` declarations and the constructors that called them.
- **Dangling reference removed (audit §4.10):** `field::Field<T>` stores `World` by value.
- **Checked MPI + fail-closed cleanup (audit §4.7, §4.11):** checked `MPI_Comm_size` in the decomposition factory and the GPU packed-halo `Irecv/Isend`; unified destructor/move-assign cleanup on a single `abort_on_mpi_error` (log + `MPI_Abort`, never throw from a destructor).
- **Device `parallel_for` trap (audit §4.2):** CUDA/HIP `parallel_for` is now a compile-time error instead of silently running device work on the host.
- **GPU initial-condition residency (audit §4.1):** `Model` gained `prepare/finalize_for_field_modifiers` hooks that the `Simulator` calls around modifier application and result writing, so App-driven GPU runs seed the device field (previously integrated from an unseeded buffer).
- **Single save scheduler (audit §4.12):** the Tungsten GPU driver uses `Time::do_save()` instead of a divergent `round(saveat/dt)` rule.
- **HeFFTe box-order invariant (audit §4.9):** `Decomposition` asserts at construction that `heffte::split_world` box order matches the x-fastest neighbor convention.

### Fixed — packaging / build (audit §11)

- Code-coverage instrumentation no longer defaults ON and is `PRIVATE`, so it cannot leak into Release builds or the installed/exported package.
- The HIP kernel library is now part of the install/export set; `OpenPFCConfig.cmake` declares its transitive dependencies (CUDAToolkit / hip / HDF5 / nlohmann_json); backend-enable definitions are exported; installs ship headers only. Guarded by a new `find_package(OpenPFC)` packaging smoke test in CI (`tests/packaging/consumer`).

### Documentation

- **Onboarding spine:** `docs/start_here_15_minutes.md`, `docs/spectral_stack.md`, `docs/recipes/`, `docs/gpu_path_decision.md`, `docs/hpc_operator_guide.md`.  
- **Quality / teaching:** `docs/when_not_to_use_openpfc.md` (fit + FD vs spectral direction), `docs/documentation_versioning.md`, `docs/from_paper_to_run.md`, `docs/workshop/`, `docs/adr/`, `docs/operator_playbooks.md`, `docs/science_numerics_limits.md`, optional printable handbook (`docs/handbook_build.md`, `scripts/build_handbook.sh`).  
- **MkDocs (optional):** `uv` project under `docs/` (`mkdocs`, Material theme); root `mkdocs.yml` builds a browsable prose site — see `docs/mkdocs_preview.md`.  
- **CI:** `scripts/check_doc_bash_syntax.py` validates fenced `bash`/`sh` blocks under `docs/`.  
- **Binary MPI-IO fields:** `docs/binary_field_io_spec.md` (layout, filename templates, collectives).  
- **Spectral `App` config keys:** `docs/spectral_app_config_reference.md` (world, time, `plan_options`, `fields`, IC/BC).  
- **HPC:** `docs/tutorials/hpc_slurm_day_one.md`, `docs/mpi_io_layout_checklist.md`.  
- **Science notes:** `docs/science_tungsten_quicklook.md`, `docs/science_cahn_hilliard_vs_allen_cahn.md`.  
- Indexes and cross-links updated (`docs/README.md`, `learning_paths.md`, `tutorials/README.md`, …).

**Upgrade discipline (maintainers):** When a change alters **CMake options**, **JSON/TOML keys**, **default writers**, or **on-disk formats**, add an explicit **migration** bullet under `[Unreleased]` (usually `### Changed` or `### Removed`) with what to do instead and a link to the relevant `docs/*.md`.

### Added

- **Sparse, grid-agnostic halo exchanger** (`include/openpfc/kernel/decomposition/sparse_halo_exchange.hpp`): new `pfc::SparseHaloExchanger<T>` and `pfc::halo::RemoteHalo<T>` accept arbitrary `(peer_rank, send_indices, recv_indices, send_tag, recv_tag)` tuples — no grid, axis, or face semantics. Drives one `MPI_Isend`/`MPI_Irecv` pair per `RemoteHalo` over the existing `core::SparseVector` + `exchange::isend_data` / `irecv_data` plumbing; supports optional `core::scatter` after the wait. The new `pfc::halo::make_structured_halos<T>(decomp, rank, hw, dirs = Axes3D())` builds the `RemoteHalo` list for the standard structured face/edge/corner exchanges driven by a `HaloDirectionSet`, and `pfc::halo::copy_to_face_layout` (in `halo_face_layout.hpp`) refills the `std::array<std::vector<T>, 6>` layout that `field::fd::laplacian_periodic_separated<Order>` expects. Foundation for future FEM / unstructured / multi-block patterns. See [`docs/concepts/halo_exchange.md`](docs/concepts/halo_exchange.md).
- **Customizable halo direction sets** (`include/openpfc/kernel/decomposition/halo_directions.hpp`): new `pfc::halo::HaloDirectionSet` and named presets **`Axes2D` (4) / `Full2D` (8) / `Axes3D` (6) / `Full3D` (26)**, plus a per-rank `HaloDirectionSelector` callback. Every face exchanger gained a new ctor that accepts a `HaloDirectionSet` (default preserves historical behaviour: `Axes3D()` for face exchangers, `Full3D()` for `FullPaddedDeviceHalo`); excluded slots are skipped in both GPU-aware and packed branches. **`apps/kobayashi/src/cuda/kobayashi_fd_cuda.cpp`** now uses **`Axes2D()`** so the 2D slab driver no longer touches **±Z** halos. See [`docs/concepts/halo_exchange.md` § 5.4](docs/concepts/halo_exchange.md) and [`apps/kobayashi/docs/cuda_halo_lessons_h100.md`](apps/kobayashi/docs/cuda_halo_lessons_h100.md).
- **CUDA padded halos**: `pfc::cuda::PaddedDeviceHaloExchanger` (`runtime/cuda/padded_device_halo_exchange.hpp`) and face pack/unpack kernels (`src/openpfc/runtime/gpu/padded_halo_faces.cu`, linked into **`kobayashi_fd_cuda`**) — same MPI derived types as `PaddedHaloExchanger<double>` with a **device** base pointer; GPU-aware MPI when supported, else narrow face slabs + pinned host. Env **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** forces the packed path.
- **Applications**: `kobayashi_fd_manual` (`apps/kobayashi/`) — Kobayashi phase-field + temperature coupling on a periodic 2D slab, explicit finite differences matching the historical Julia `kobayashi_v1` layout; PNG snapshots of \(\phi\); optional **`KOBAYASHI_VERIFY`** / **`KOBAYASHI_VERIFY_HEX`** stdout lines and **`OPENPFC_KOBAYASHI_SKIP_PNG`** / **`OPENPFC_KOBAYASHI_QUIET`** env toggles; Slurm scaling under `apps/kobayashi/slurm/` (`gen05_epyc`) with **`summarize_scaling.py`** and **`plot_strong_scaling.py`** (SVG strong-scaling figure from **`summary.tsv`**). **`kobayashi_fd_openmp`** — same numerics on one node with periodic **index wrapping** (no MPI halos) and **OpenMP** parallelism; Catch2 **`test_kobayashi_fd_openmp`**; Slurm **`kobayashi_openmp_scaling_gen05_epyc.sbatch`** and **`summarize_openmp_scaling.py`**.
- **Documentation / clusters:** Kobayashi **`apps/kobayashi/slurm/`** README and **`kobayashi_rebuild_openpfc_gen05_epyc.sbatch`** for rebuilding against **`openmpi/5.0.10`** (Slurm PMI/`srun`).
- **HeFFTe builds**: Optional vendoring of HeFFTe 2.4.1 via CMake FetchContent (`FetchHeffte.cmake`), HeFFTe discovery hints, and a pinned GCC 11 + OpenMPI toolchain preset for fixed cluster layouts (e.g. tohtori).
- **HIP / AMD GPUs**: CMake HIP/ROCm detection, HeFFTe ROCm backend when HIP is enabled, rocFFT-backed `fft::create_hip`, and Tungsten on HIP (model, kernels, applications, scalability and VTK-focused tests).
- **MPI halos**: `PersistentHaloExchanger` for six-face persistent MPI halo exchange (with integration coverage against the non-persistent path).
- **Profiling**: Kernel `ProfilingSession` library, wiring through the JSON/TOML `App`, MPI path instrumentation, and documentation plus Tungsten input/schema alignment for performance runs.
- **Kokkos-like API**: Experimental View types, execution and memory-space tags, `parallel_for` / `fence`, and host–device copy/mirror helpers for Kokkos-style structuring of numerical code.
- **Field modifiers**: Multi-field initial/boundary condition targets on `FieldModifier`, with `Simulator` validating each name against `has_field`.
- **GPU MPI**: CMake option `OpenPFC_MPI_CUDA_AWARE` for GPU-aware MPI when using CUDA.
- **Documentation**: Developer style guide (`docs/styleguide.md`), separate CPU vs CUDA/HIP build layout guide (`docs/build_cpu_gpu.md`), LUMI-G build notes, and expanded performance/profiling material.

### Changed

- **CUDA GPU-aware MPI (configure + Slurm):** `find_package(MPI REQUIRED COMPONENTS C CXX)`; configure probe runs **`try_run`** on **`cmake/openpfc_mpix_cuda_probe.c`** (links **`MPI::MPI_C`**) so **`MPIX_Query_cuda_support()`** is validated on the configure host. **`kobayashi_rebuild_openpfc_cuda_h100.sbatch`** defaults to **`-DOpenPFC_MPI_CUDA_AWARE=ON`** and uses **`mpicxx`** as **`CMAKE_CXX_COMPILER`**; **`KOBAYASHI_REBUILD_CUDA_MPI_AWARE=0`** forces a packed-only compile path.
- **`PaddedDeviceHaloExchanger` (GPU-aware):** periodic face neighbors that map to **the same MPI rank** (e.g. ±Z when the process grid is **1** deep in Z, as in **`nz = 1`** Kobayashi slabs) no longer use **`MPI_Irecv` / `MPI_Isend` on device buffers to self**; those faces use **device pack/unpack** into a small **`m_d_scratch`** buffer instead, avoiding pathological stalls / low GPU utilization on some Open MPI + UCX builds.
- **`kobayashi_fd_cuda`**: **`MPI_COMM_WORLD` size 1** uses **device-only periodic halos** (`device_periodic_local`) instead of **`PaddedDeviceHaloExchanger`** in the timestep loop (avoids MPI progress on the host and redundant global CUDA sync). **`nproc > 1`** still uses **`PaddedDeviceHaloExchanger`**; rank 0 prints **`KOBAYASHI_CUDA_HALO_MODE`**.
- **CUDA build**: `padded_halo_faces.cu` is compiled into **`kobayashi_fd_cuda`** instead of **`libopenpfc_gpu_kernels`** so separable CUDA compilation registers correctly at the final device link (static archives + mixed host link previously produced undefined `__cudaRegisterLinkedBinary_*` symbols).
- **Clusters (tohtori):** Default Open MPI **5.0.10** in **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`**, **`CMakePresets.json`**, and **`scripts/build_tohtori.sh`**; **`INSTALL.md`**, **`cmake/README.md`**, **`scripts/README.md`**, dependency matrix, and related docs aligned (override with **`OPENMPI_ROOT`** when needed).
- **Profiling**: `ProfilingSession` is frame-only generic (`begin_frame`, `set_frame_metric`, `set_frame_metric_elapsed_since_begin`, `end_frame`); OpenPFC step/MPI/memory wiring uses **`openpfc_frame_metrics.hpp`**. JSON **`frame_metric_names`** use **`heap_secondary_bytes`** instead of **`fft_heap_bytes`** for the second heap column.
- **Profiling export**: JSON and HDF5 use **schema version 2** with a per-MPI-rank hierarchy (`ranks[]` in JSON; `openpfc/profiling/ranks/<id>/` in HDF5). See **`docs/profiling_export_schema.md`**. **`ProfilingPrintOptions::wall_denominator_metric`** configures the %tot denominator (default **`wall_step`**). **`print_profiling_timer(std::ostream &, MPI_Comm, …)`** with **`mpi_aggregate_stdout`** prints a rank-0 table combining per-rank timer totals (**`mpi_aggregate_stat`**: mean/sum/min/max/median). **`App`** enables this when **`profiling.print_report`** is true (all ranks participate in the gather).
- **Profiling export (schema v3)**: Optional **`ProfilingExportOptions::run_id`** and **`export_metadata`**; **`App`** reads **`profiling.run_id`**, **`profiling.export_metadata`**, and environment (**`SLURM_JOB_ID`**, **`OPENPFC_PROFILING_RUN_ID`**, domain sizes, Slurm layout). When **`run_id`** is set, HDF5/JSON use a merge-friendly layout under **`openpfc/profiling/runs/<id>/`**. **`experiments/scalability/`** documents a Slurm driver for scaling studies (site/workload profiles, **`scala`** CLI).
- **Layout & includes**: Clearer **kernel / runtime / frontend** layering, consistent `<openpfc/...>` includes across the library and unit tests, and FFT layout helpers split into `fft_layout.hpp`.
- **SparseVector / MPI**: Zero-copy face exchange and non-blocking MPI paths for sparse halo communication where applicable.
- **CI**: GitHub Actions runners pinned to **Ubuntu 24.04 LTS** (main matrix, coverage, docs, clang-tidy, code quality). LLVM apt repos use **noble** for Clang 14/16; removed the gcc-13 toolchain PPA. The coverage job runs tests via **CTest** like the main workflow; workflow README aligned with HeFFTe 2.4.1.
- **UI**: `list_valid_field_modifiers()` reads registered names from `FieldModifierRegistry` (sorted) instead of a duplicated literal list.

### Fixed

- **MPI `SparseVector` neighbor tests** (`tests/unit/kernel/decomposition/test_sparse_vector_neighbor_exchange.cpp`): rank-two-only cases now **skip** when `MPI_Comm_size != 2` (they previously called matching `receive` on ranks ≥2 and hung). Ring / multi-neighbor exchanges use **safe blocking order** (odd/even send–recv), the **2×2 grid** case exchanges horizontal then vertical halves without deadlock, and **multiple-neighbor** recv peers / expected values were corrected (left-going payload is received from the **right** neighbor). Eliminates long **`mpirun -n 4 … '[MPI]'`** hangs / SIGTERM from these tests.
- **`PersistentHaloExchanger`**: persistent requests now use the same **MPI tag pairing** as `HaloExchanger` zero-copy (`MPI_Recv_init` uses **opposite face slot**, `MPI_Send_init` uses the **local slot**), and requests are registered **all recv then all send** like `start_halo_exchange`. Integration parity runs on **`MPI_Comm_size == 2`** with **`{1,1,2}`** Z-splitting so ±Z neighbors are never **self** (avoids fragile persistent self-message ordering); **`mpi_4procs_grid_multiple`** no longer includes this case (it remains under **`mpi_2procs_all`**).
- **`[MPI]` tag hygiene:** `test_halo_exchange_driver.cpp` (hard-wired `{2,1,1}` / two subdomains) and the **`[profiling][MPI]`** JSON/timer assertions now **return early unless `MPI_Comm_size == 2`**, so a broad filter like **`mpirun -n 4 ./openpfc-tests '[MPI]'`** no longer runs them with **world size ≠ decomposition domains** or **≠ 2 assumed ranks**.
- **Decomposition**: Halo loops aligned with **inclusive** `World` bounds so face exchanges match the intended domain.
- **MPI timer**: `pfc::mpi::timer::toc()` no longer reads uninitialized state when called before `tic()`; misuse now throws `std::logic_error`, and `reset()` clears an in-progress lap.
- **Logging**: If `gmtime_r` / `gmtime_s` fails, log lines use a `<time-unavailable>` placeholder instead of formatting an uninitialized `tm`.
- **Memory reporter**: `get_system_memory_bytes()` only uses parsed `MemTotal` kB when stream extraction succeeds, avoiding read of uninitialized `mem_kb` on malformed lines.
- **`PaddedDeviceHaloExchanger` (packed fallback):** same-rank periodic faces (e.g. ±Z when **local nz = 1**) no longer use **`MPI_Irecv` / `MPI_Isend` to self** on **nx×ny** face buffers (~128 MiB per message at 4096²); they use **device pack/unpack** like the GPU-aware path, avoiding multi‑second-per-step stalls when **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`**.

### Removed

- **`pfc::SeparatedFaceHaloExchanger<T>`** and `include/openpfc/kernel/decomposition/separated_halo_exchange.hpp`. The face-only exchanger has been superseded by the fully sparse `pfc::SparseHaloExchanger<T>` plus `pfc::halo::make_structured_halos<T>` for the structured shortcut. **Migration:** replace `SeparatedFaceHaloExchanger<T> ex(decomp, rank, hw, comm);` + `ex.exchange_halos(u.data(), u.size(), face_halos);` with `SparseHaloExchanger<T> ex(comm, rank, halo::make_structured_halos<T>(decomp, rank, hw));` + `ex.exchange_halos(u.data(), u.size()); halo::copy_to_face_layout(ex, face_halos);`. `pfc::halo::FaceHaloCounts`, `face_halo_counts`, and `allocate_face_halos` are unchanged. **Drive-by fix:** `pfc::halo::Connectivity::Edges` no longer aliases `Faces` in `create_halo_patterns` — it now correctly returns the 18-direction faces+edges subset (corners excluded).
- **Nix / flake support**: Removed `flake.nix`, `flake.lock`, and the `nix/` packaging tree; dropped the Nix job from CI. Use CMake and `INSTALL.md` for builds.

## [0.1.4] - 2025-12-18

### Added

- **GPU/CUDA Support**: Complete CUDA implementation enabling GPU-accelerated PFC simulations.
  Added `DataBuffer` for backend-agnostic memory management with CPU/GPU memory traits,
  CUDA FFT integration via HeFFTe, GPU kernels for element-wise operations, and `GPUVector`
  RAII container. Implemented full Tungsten model on GPU with optimized kernel launches and
  CPU-GPU synchronization for FieldModifiers and VTK output. Runtime backend selection API
  allows choosing between CPU and CUDA FFT backends via configuration. Comprehensive test
  coverage includes GPU device detection, memory allocation, FFT operations, and CPU vs CUDA
  result comparison. Build system supports optional `OpenPFC_ENABLE_CUDA` flag.
- **VTK Output**: New VTK ImageData writer in `include/openpfc/results/vtk_writer.hpp` and
  `src/results/vtk_writer.cpp` for parallel visualization output. Generates `.vti` files
  for each rank and `.pvti` parallel metadata files for ParaView/VisIt. Includes comprehensive
  test suite with MPI-aware tests and single-invocation test model to prevent cleanup races.
- **TOML Configuration**: Added TOML config file support alongside JSON. New
  `feat(utils): Add TOML to JSON conversion utility` enables `.toml` input files with
  automatic conversion. Integrated tomlplusplus library via CMake find module. All example
  configurations converted to TOML format. Unit tests validate conversion accuracy.
- **Modular CMake Architecture**: Refactored monolithic CMakeLists.txt into 12 focused modules
  in `cmake/` directory: ProjectSetup, CompilerSettings, CUDASupport, Dependencies,
  LibraryConfiguration, BuildOptions, CodeCoverage, Installation, PackageConfig, BuildSummary.
  Improves maintainability and reusability. Documented in `cmake/README.md`.
- **CI/CD Pipelines**: Comprehensive GitHub Actions workflows for build matrix (GCC/Clang,
  multiple OS), documentation deployment, code coverage analysis with Codecov integration,
  and REUSE license compliance. Status badges added to README. Documentation includes
  workflow descriptions and troubleshooting guides.
- **Parameter Validation System**: New UI subsystem for configuration validation with
  `ParameterMetadata`, `ParameterValidator`, and `ValidationResult` classes. Supports nested
  path validation, finite checks, type validation, and helpful error messages. Integrated
  into Tungsten app with comprehensive test coverage (300+ assertions).
- **FFT Backend Selection**: Runtime FFT backend selection API allowing users to choose
  between available HeFFTe backends (FFTW, MKL, cuFFT) via configuration. New
  `examples/fft_backend_benchmark.cpp` demonstrates performance comparison. Backend field
  added to config schema with parsing and validation.
- **SparseVector & MPI Exchange**: New `SparseVector` container with halo exchange patterns
  for domain decomposition. Includes gather/scatter operations, neighbor exchange with MPI,
  and halo pattern creation utilities. Comprehensive test suite validates exchange correctness.
- **Testing Infrastructure**: First integration test suite for diffusion model validating
  complete simulation pipeline against analytical solutions (4 test cases, 331 assertions).
  Added benchmark subdirectory with microbenchmarks for World coordinate operations.
  Comprehensive unit tests for UI validation (300+ assertions), VTK writer (MPI-aware),
  DataBuffer, GPUVector, and SparseVector. Switched to single-invocation test model to
  prevent MPI initialization issues. Test coverage improvements across all modules.
- **World API**: Type-safe World construction using strong types from `strong_types.hpp`.
  Added new `create(GridSize, PhysicalOrigin, GridSpacing)` overload preventing parameter
  confusion at compile time. Old `create(Int3, Real3, Real3)` API deprecated. Zero overhead -
  strong types compile away completely. Updated all examples and helper functions. Test suite
  with 71 assertions covering type safety, zero overhead, and backward compatibility.
- **Documentation**: Added 10 comprehensive API examples (World, FFT, Simulator, Time,
  Decomposition, ResultsWriter, FieldModifier, DiscreteField, Model, custom field initializer).
  Added CITATION.cff for standardized citations. Improved Doxygen configuration. README
  sections on configuration validation, FFT backend selection, and extending OpenPFC.
- **Research Tools**: Added power consumption benchmarks for FFT operations (CPU and GPU),
  multi-GPU HeFFTe examples, and scalability testing applications for Tungsten model.

### Changed

- **CMake Structure**: Root `project()` moved to top-level CMakeLists.txt. Build options
  reorganized into logical modules. Test discovery switched to single-invocation model.
  Benchmark compilation now optional via `OpenPFC_BUILD_BENCHMARKS`.
- **Tungsten Structure**: Split monolithic tungsten code into modular headers and separate
  JSON inputs into `inputs_json/` subdirectory. Restructured JSON schema to nested format.
  Renamed 'origo' field to 'origin' for consistency.
- **UI Module**: Split monolithic `ui.hpp` into modular components. Made `plan_options`
  optional in app config. Added error formatting utilities for better user messages.
- **World Module**: Split `world.hpp` into modular headers. Added query helper examples.
  Updated coordinate benchmark documentation.
- **Test Organization**: Split monolithic parameter validation tests. Serialize VTK writer
  tests to prevent cleanup races. Make `MPI_Worker` static to persist MPI per process.
  Normalize test commands under single-invocation model.
- **Build Warnings**: Enabled additional compiler warnings for code quality in Debug builds.
  Added `-Werror=format-security`. Made GCC-specific warnings conditional. Format check
  warns instead of fails in Nix builds.
- **Dependencies**: Updated nixpkgs from 23.11 to 24.05. Added git and tomlplusplus to
  Nix build dependencies. Integrated Catch2 test discovery.

### Fixed

- **Build System**: Fixed CMake warnings by moving `project()` to root. Fixed Catch2 test
  discovery and optional MPI suites. Made documentation comment posting optional in CI.
  Cleaned up clang-format artifacts before REUSE checks. Improved error reporting in Nix tests.
- **Test Fixes**: Fixed narrowing conversions in sparse vector tests. Fixed GridSpacing
  initializers in FFT tests. Fixed syntax errors in world benchmark and CUDA tests. Added
  missing `pfc` namespace qualifiers. Suppressed unused variable/parameter warnings with
  `[[maybe_unused]]`. Fixed incorrectly converted `world::create` calls.
- **Application Fixes**: Fixed missing `set_fft()` call in diffusion example causing runtime
  errors. Removed unused fields (verbose in Diffusion, m_first in Aluminum). Fixed array
  initialization in SeedFCC. Added MPI-aware main to tungsten CPU vs CUDA test.
- **MPI Fixes**: Fixed `MPI_Worker` to be safe for test frameworks. Query current MPI
  rank/size when generating PVTI instead of using stale values. Synchronize ranks before
  cleanup in VTK writer test to prevent races.
- **Memory Safety**: Initialize all params struct members in aluminum to prevent undefined
  behavior. Add explicit template instantiation for World constructor. Fix CPU FFT
  `std::vector` interface to call HeFFTe directly.
- **Code Quality**: Removed redundant const qualifiers. Added missing override keywords.
  Fixed variable shadowing in multiple files. Removed variable shadowing in timing collection.
  Fixed clang-format violations across codebase.
- **CI/CD**: Removed ubuntu-20.04 from test matrix. Removed Cachix binary cache step.
  Initialized git submodules in all workflows. Made clang-format check warning instead of
  error. Used forked clang-format-action with fail-on-error option.
- **Documentation**: Removed internal tracking references from code. Added SPDX headers for
  REUSE compliance to all test READMEs. Fixed Doxygen file headers for better doc generation.

### Deprecated

- **World API**: Old `world::create(Int3, Real3, Real3)` deprecated in favor of type-safe
  `create(GridSize, PhysicalOrigin, GridSpacing)`. Migration guide in documentation.

### Breaking Changes

None - all deprecated APIs remain functional with warnings.

## [0.1.3] - 2025-11-25

### Added

- **Examples**: Custom coordinate system example in `examples/17_custom_coordinate_system.cpp`
  demonstrating OpenPFC's extensibility via ADL (Argument-Dependent Lookup). Implements
  complete polar (2D: r, θ) and spherical (3D: r, θ, φ) coordinate systems with coordinate
  transformations (`polar_to_coords()`, `polar_to_indices()`, `spherical_to_coords()`,
  `spherical_to_indices()`). Includes comprehensive Doxygen documentation (615 lines),
  round-trip transformation verification, and 4-step recipe showing users how to add
  custom coordinate systems without modifying OpenPFC source code. Embodies "Laboratory,
  Not Fortress" philosophy - users can extend with cylindrical, spherical, or custom
  geometries through tag-based dispatch and free functions. Example compiles cleanly
  with zero warnings and demonstrates working coordinate conversions with correct output.
- **Documentation**: Comprehensive API documentation for top 10 most-used public
  APIs with detailed @example blocks and usage patterns. Enhanced documentation
  covers World (domain creation and coordinate transforms), Model (physics
  implementation), Simulator (time integration orchestration), FFT (spectral
  transforms), Time (time stepping), Decomposition (parallel decomposition),
  ResultsWriter (output formats), FieldModifier (IC/BC extensibility), and
  DiscreteField (coordinate-aware fields). Added 10 standalone example programs
  (4,570+ lines) demonstrating complete usage workflows from basic setup to
  production PFC simulations. Includes build system integration via
  docs/api/examples/CMakeLists.txt with BUILD_API_EXAMPLES option. Documentation
  warnings reduced from 9 to 1 (89% improvement). All examples validated and
  test suite confirms no regressions (73 test cases, 5,836 assertions passing).
- **FFT**: K-space helper functions in `include/openpfc/fft/kspace.hpp` providing
  zero-cost abstractions for wave vector calculations in spectral methods.
  Added 4 inline helper functions: `k_frequency_scaling(world)` for computing
  frequency scaling factors (2π/L), `k_component(index, size, freq_scale)` for
  wave vector components with Nyquist folding, `k_laplacian_value(ki, kj, kk)`
  for computing -k² Laplacian operator, and `k_squared_value(ki, kj, kk)` for
  magnitude squared. Eliminates 120+ lines of duplicated k-space calculation
  code across examples (04_diffusion_model.cpp, 12_cahn_hilliard.cpp, tungsten.cpp,
  etc.). All functions are inline, noexcept, and compile to identical machine
  code as manual implementation (zero runtime overhead). Comprehensive test
  coverage (177 assertions in 6 test cases). Example program added in
  `examples/fft_kspace_helpers_example.cpp` demonstrating before/after comparison.
- **DiscreteField**: Converted `interpolate()` from member function to free function
  `pfc::interpolate(field, coords)` aligning with OpenPFC's "structs + free functions"
  design philosophy. Added both mutable and const overloads for type safety. Member
  function deprecated with `[[deprecated]]` attribute for v1.x backward compatibility
  (will be removed in v2.0). Free function enables ADL-based extension allowing users
  to provide custom interpolation schemes without modifying OpenPFC. Updated all
  11 call sites across tests, examples, and documentation to use new API. Added
  comprehensive test coverage (95+ new test lines) including mutable/const overloads,
  ADL lookup verification, and nearest-neighbor rounding behavior tests. All 222
  assertions pass. Zero runtime overhead maintained (inline functions).

## [0.1.2] - 2025-11-21

### Added

- **Core**: World construction helper functions in `include/openpfc/core/world.hpp`
  providing ergonomic, zero-cost abstractions for common grid creation patterns.
  Added 5 inline helper functions: `uniform(int)` and `uniform(int, double)` for
  N³ grids, `from_bounds(...)` for automatic spacing computation from physical
  bounds (periodic/non-periodic aware), `with_spacing(...)` for custom spacing
  with default origin, and `with_origin(...)` for custom origin with unit spacing.
  All helpers include input validation with clear error messages. Reduces
  boilerplate from `world::create({64,64,64}, {0,0,0}, {1,1,1})` to
  `world::uniform(64)`. Backward compatible - existing `create()` API unchanged.
  Comprehensive test coverage (32 new assertions). Example program added in
  `examples/world_helpers_example.cpp`.
- **Core**: Mathematical constants in `include/openpfc/constants.hpp` for
  compile-time evaluation with zero runtime overhead. Added 12 constants: π,
  2π, π/2, π/4, 1/π, √π, √2, √3, e, ln(2), ln(10), and φ (golden ratio).
  All constants are `constexpr double` with 16+ decimal digits precision.
  Comprehensive Doxygen documentation included. Constants accessible via both
  `pfc::constants::pi` and `pfc::pi` namespaces. API matches C++20
  `std::numbers` for future migration.
- **Testing**: Comprehensive test suite for mathematical constants in
  `tests/unit/core/test_constants.cpp` with 13 test cases and 41 assertions
  covering precision verification, derived constants, compile-time evaluation,
  and integration scenarios (FFT wave numbers, crystal geometry).
- **Testing**: Pre-commit hook for automatic clang-format checking to prevent
  formatting issues before pushing to CI. Hook available in `scripts/pre-commit-hook`
  with installation instructions in `scripts/README.md`.
- **Testing**: Comprehensive test coverage improvements achieving 90.7% line
  coverage and 94.8% function coverage. Added tests for `utils.hpp`,
  `world.cpp`, and `fixed_bc.hpp`.
- **Build system**: Added `-Werror=format-security` compiler flag to catch
  format string vulnerabilities locally before CI, matching CI behavior.
- **Documentation**: Added SPDX license headers to test README files
  (`tests/`, `tests/benchmarks/`, `tests/fixtures/`, `tests/integration/`,
  `tests/unit/`) for REUSE compliance (174/174 files now compliant).
- **Documentation**: Added comprehensive `@file` documentation tags to all 41
  public header files in `include/openpfc/` achieving 100% coverage. Each header
  now includes brief description, detailed explanation, practical usage examples,
  and cross-references to related components. Reduced Doxygen @file warnings
  from 47 to 0. Improves API discoverability for new users and enables better
  IDE/LLM assistance.

### Fixed

- **Examples**: Replaced runtime pi calculation (`std::atan(1.0) * 4.0`) with
  compile-time `pfc::constants::pi` in `diffusion_model.hpp`,
  `12_cahn_hilliard.cpp`, and `05_simulator.cpp` for zero runtime overhead in
  FFT wave number calculations. Removed unused global PI constants.
- **CMake build system**: Fixed Catch2 detection in `FindCatch2.cmake` by
  explicitly setting `Catch2_FOUND` variable after `FetchContent_MakeAvailable`.
  This enables the test suite to build when `OpenPFC_BUILD_TESTS=ON`.
- **CMake build system**: Fixed HeFFTe detection in `FindHeffte.cmake` by
  setting `Heffte_FOUND=TRUE` after FetchContent to prevent fatal errors when
  HeFFTe is downloaded instead of using system-installed package.
- **tungsten application**: Added explicit `find_package(Heffte REQUIRED)` and
  corrected target link to `Heffte::Heffte` to ensure proper linkage with
  separately installed HeFFTe v2.4.1.
- **Code quality**: Fixed format-security compiler error in `utils.hpp` by
  adding overload for `string_format()` with no variadic arguments.
- **Code formatting**: Removed trailing whitespace in `test_fft.cpp` to pass
  clang-format checks.

### Breaking Changes

- **Model::rank0 is now private**: The public member variable `rank0` has been
  moved to private section and renamed to `m_rank0`. Use the `Model::is_rank0()`
  method instead.
  - **Migration**: Replace `model.rank0` with `model.is_rank0()` in your code
  - **Reason**: Better encapsulation and consistent API with other query methods
    like `get_world()` and `get_fft()`
  - **Impact**: All examples and applications updated to use the new API
  - **Note**: The method `is_rank0()` is now `const` and `inline` for zero overhead

## [0.1.1] - 2024-06-13

- Make some changes to tungsten and aluminum models to be more consistent with
  the use of minus signs in different operators: move minus sign from peak
  function to opCk operator (commits 8685f7a and b4392b3).
- Bug fixes and changes in CMakeLists.txt: conditionally install nlohmann_json
  headers (issue #16), do not add RPATH to binaries when installing them,
  (commit 6c91de3) and also install binaries to INSTALL_PREFIX/bin (issue #14).
- Start using clang-format in the project (ci pipeline). (Issue #43)
- Add possibility to add initial and boundary conditions to fields with other
  name than "default". (Commit c65fb23)
- Add schema file for the input file. (Commit 6eeeab9)
- Fix license headers in source files, add license header checker to GH Action
  and in general improve licensing information. (Issues #25, #39, #40)
- Replace `#pragma once` with a proper include guard in all header files. (Issue
  #48)
- Fix bug with clang-tidy configuration preventing compilation. (Issue #52)
- Major updates to README.md: update citing information, add description of
  application structure, add new images, scalability results, and add example
  simulation of Cahn-Hilliard equation. (Issues #5, #19, #22, #23, #27, #28,
  #40)

## [0.1.0] - 2023-08-17

- Initial release.
