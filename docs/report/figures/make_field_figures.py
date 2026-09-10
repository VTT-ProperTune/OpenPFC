#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate the field-visualisation figures for the applications report.

Reads the `.vti` / `.bin` output of a handful of real runs (see
Reads the `.vti` / `.bin` / `.png` output of real runs (see
Reads the `.vti` / `.bin` output of a set of real runs (see
`run_field_demos.sh` for how to reproduce them) and renders SVGs into this
directory using `field_io.py` (readers) and `field_plots.py` (panels,
montages, comparisons, line plots). The report itself has no compute
engine: it reads these committed SVGs, not raw simulation output.

    FIELD_DATA_DIR=<where run_field_demos.sh wrote output> \\
        /flash/project_462001519/juaho/venv-pytest/bin/python \\
        docs/report/figures/make_field_figures.py

Requires matplotlib + numpy. `module load cray-python` on LUMI does not
provide matplotlib; use a virtual environment
(`/flash/project_462001519/juaho/venv-pytest/bin/python` on the shared
allocation this was developed on, or your own venv elsewhere).
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from field_io import (  # noqa: E402
    GridSpec, read_gray_png, read_vti, slice_bin, with_spacing,
)
from field_plots import (  # noqa: E402
    render_comparison, render_line_comparison, render_montage, render_panel,
)
from field_io import Field2D, GridSpec, read_vti, slice_bin  # noqa: E402
from field_plots import render_comparison, render_montage, render_panel  # noqa: E402

DATA_DIR = Path(os.environ.get("FIELD_DATA_DIR", Path.cwd() / "_field_demo_data"))


def _vti_series(run_dir: Path, pattern: str, times):
    return [read_vti(run_dir / pattern.format(i), time=t) for i, t in times]


def _center_crop(field: Field2D, n: int) -> Field2D:
    """The central `n` x `n` cells of a field, with the extent kept honest.

    Needed where the *local* structure is the claim and the whole field is
    too finely patterned to read at print size -- a lattice whose period is
    a handful of cells, for instance. `field_plots` has no crop of its own
    (its three entry points all render a whole `Field2D`), so slicing the
    array and recomputing the extent from the original one's cell size is
    the way to hand it a smaller field without lying about coordinates.
    """
    ny, nx = field.data.shape
    if n > nx or n > ny:
        raise ValueError(f"cannot crop {n}x{n} out of a {nx}x{ny} field")
    i0, j0 = (nx - n) // 2, (ny - n) // 2
    x0, x1, y0, y1 = field.extent
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    return Field2D(
        data=field.data[j0:j0 + n, i0:i0 + n],
        extent=(x0 + i0 * dx, x0 + (i0 + n) * dx, y0 + j0 * dy, y0 + (j0 + n) * dy),
        name=field.name,
        time=field.time,
        units=field.units,
    )


def figure_cahn_hilliard_coarsening_montage():
    """Broadband spinodal decomposition of Fe-32Cr: one figure, six times.

    This is the "phase-separating alloy" the report abstract promises: a
    single composition field starting from small random noise, spinodally
    unmixing into Cr-rich and Fe-rich domains, then coarsening. The colour
    scale is diverging and centred on c0=0.32 (the alloy's mean
    composition) because c above/below the mean is the physically
    meaningful signal, not the absolute composition.
    """
    run_dir = DATA_DIR / "cahn_hilliard_coarsening" / "results" / "cahn_hilliard"
    saveat = 5.0
    # Spinodal decomposition is fast: mean-field diagnostics
    # (results/cahn_hilliard/diagnostics.csv) show c already spanning
    # [0.14, 0.87] by t=5 and [0.04, 0.98] by t=10, so evenly spaced
    # indices would spend most of the montage on the slow coarsening tail
    # and miss the nucleation itself. Densely sample the first three
    # saves, then spread out.
    indices = [0, 1, 2, 4, 8, 20]
    fields = _vti_series(run_dir, "c_{:04d}.vti", [(i, i * saveat) for i in indices])
    fig = render_montage(
        fields,
        kind="diverging",
        center=0.32,
        panel_titles=[f"t={f.time:g}" for f in fields],
        suptitle="Fe-32Cr spinodal decomposition (broadband noise IC)",
        cbar_label="Cr mole fraction c",
        axis_units="grid units",
    )
    out = HERE / "cahn_hilliard_coarsening_montage.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_cahn_hilliard_comparison():
    """Single Fourier mode vs broadband noise, same alloy, same t=10.

    Same physics (`fe_cr_spinodal.json` vs `coarsening.json`), same time,
    different initial condition: one seeded Fourier mode grows into a
    single, still nearly sinusoidal, stripe pattern, while broadband noise
    has already nucleated many separate domains. Shared colour scale makes
    the difference in domain count and sharpness directly comparable.

    t=10 (not `fe_cr_spinodal.json`'s full t1=20) is deliberate: this
    single-mode case keeps growing past the point where the composition
    should saturate near 0 or 1, and by t=12 it has overshot those physical
    bounds (ETD here is not unconditionally stable at large amplitude, see
    `apps/cahn_hilliard/README.md`). t=10 is the latest sample that is still
    physically sane (`0 <= c <= 1`).
    """
    single = read_vti(
        DATA_DIR / "cahn_hilliard_single" / "results" / "cahn_hilliard" / "c_0005.vti", time=10.0
    )
    coarsening = read_vti(
        DATA_DIR / "cahn_hilliard_coarsening" / "results" / "cahn_hilliard" / "c_0002.vti",
        time=10.0,
    )
    fig = render_comparison(
        single,
        coarsening,
        kind="diverging",
        center=0.32,
        label_a="single-mode seed (fe_cr_spinodal.json)",
        label_b="broadband noise (coarsening.json)",
        suptitle="Same alloy, same t=10: initial condition sets the morphology",
        cbar_label="Cr mole fraction c",
        axis_units="grid units",
    )
    out = HERE / "cahn_hilliard_comparison.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_thin_film_dewetting_montage():
    """Lubrication dewetting of a thin liquid film: one figure, six times.

    The "dewetting film" the report abstract promises: a near-uniform film
    (h0=1, +-2%) with a cosine seed near the fastest-growing unstable
    wavelength grows a periodic thickness modulation -- ridges thickening,
    troughs thinning -- as `apps/thin_film`'s linear-instability analysis
    predicts. By t=84 the trough has thinned by 15% and the ridge has
    thickened by 9%; this run (`dewetting.json` with `t1`/`saveat`
    extended to 120/12, see `run_field_demos.sh`) becomes numerically
    unstable soon after (the disjoining-pressure nonlinearity gets stiff as
    h drops further), so this montage stops at the last physically sane
    frame rather than showing true rupture. `h` is one-sided (thickness
    cannot go negative), so the colour map is sequential, not diverging.
    """
    run_dir = DATA_DIR / "thin_film_dewetting" / "results" / "thin_film"
    saveat = 12.0
    indices = [0, 2, 4, 5, 6, 7]
    fields = _vti_series(run_dir, "h_{:04d}.vti", [(i, i * saveat) for i in indices])
    fig = render_montage(
        fields,
        kind="sequential",
        panel_titles=[f"t={f.time:g}" for f in fields],
        suptitle="Growing dewetting instability (cosine seed near k_peak)",
        cbar_label="film thickness h",
        axis_units="grid units",
    )
    out = HERE / "thin_film_dewetting_montage.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_thin_film_comparison():
    """Dewetting (A>0, unstable) vs leveling (A=0, stable) at matched t=84.

    Same lubrication model, same mean thickness h0=1; the only difference
    is the disjoining-pressure amplitude A. With A>0 (`dewetting.json`) the
    initial +-2% ripple has grown into a -15%/+9% modulation by t=84; with
    A=0 (`leveling.json`, run to the same t1/saveat so the two are sampled
    at identical times) there is no destabilising pressure and the larger
    initial roughness (+-5%) has instead relaxed to +-0.2%. Shared colour
    scale makes that divergence directly comparable.
    """
    dewetting = read_vti(
        DATA_DIR / "thin_film_dewetting" / "results" / "thin_film" / "h_0007.vti", time=84.0
    )
    leveling = read_vti(
        DATA_DIR / "thin_film_dewetting" / "results" / "thin_film" / "leveling_h_0007.vti",
        time=84.0,
    )
    fig = render_comparison(
        dewetting,
        leveling,
        kind="sequential",
        label_a="dewetting, A=0.05 (unstable)",
        label_b="leveling, A=0 (stable)",
        suptitle="Same film, same t=84: the disjoining pressure decides the fate",
        cbar_label="film thickness h",
        axis_units="grid units",
    )
    out = HERE / "thin_film_comparison.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_tungsten_seed_panel():
    """A tungsten PFC seed nucleus, mid-depth slice, after 12 time units.

    The 3D `psi` density field (256x256x256, `tungsten_single_seed_256_cuda.json`'s
    domain) holds a single crystalline seed -- `SingleSeed`'s hard-coded
    radius is 64 reduced length units -- embedded in a uniform undercooled
    liquid at psi=n0=-0.4. A raw `.bin` dump has no header, so the grid
    geometry (`GridSpec`) comes from the run's JSON `domain`, out of band.
    The colour scale is diverging and centred on n0, the liquid baseline
    density -- what the reader should read off the figure is where the
    field departs from the background, not its absolute value.

    Why 256^3 and not the 32^3 grid in `tungsten_single_seed.json`: at
    32^3 the domain's half-width (17.8 reduced units) is far smaller than
    the seed radius (64), so the "seed" fills the entire periodic box and
    there is no liquid left to show a nucleus *in*. `run_field_demos.sh`
    runs the 256^3 config with a short `t1=12` (this isothermal case is
    close to a steady coexistence point -- the seed does not visibly grow
    or shrink on this timescale, it only relaxes at the facet edges) and
    writes `.bin` instead of the config's default `.vti` to exercise the
    raw-binary reader.
    """
    grid = GridSpec(
        nx=256, ny=256, nz=256,
        dx=1.1107207345395915, dy=1.1107207345395915, dz=1.1107207345395915,
        origin="center",
    )
    run_dir = DATA_DIR / "tungsten_seed" / "results" / "tungsten"
    field = slice_bin(run_dir / "psi_0006.bin", grid, axis="z", name="psi", time=12.0)
    fig = render_panel(
        field,
        kind="diverging",
        center=-0.4,
        title="tungsten PFC seed, mid-depth slice, t=12",
        cbar_label="density field ψ",
        axis_units="reduced PFC units",
        figsize=(4.8, 4.2),
    )
    out = HERE / "tungsten_seed_panel.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_surface_diffusion_comparison():
    """Isotropic vs anisotropic anneal of the *same* nanosurface, at t=8.

    `nanosurface_isotropic.json` and `nanosurface_anisotropic.json` start
    from the identical crossed corrugation -- 16 periods along x superposed
    on 16 periods along y, amplitude 0.05 each -- and differ in exactly one
    number, the anisotropy strength `eps_a`. Everything visible between the
    two panels is therefore the anisotropy.

    Isotropically the linear symbol depends only on |k|, so both ridge sets
    decay at the same rate and the egg-crate pattern survives, only fainter.
    With `eps_a=0.5, m=6`, `B(theta) = B_0[1 + eps_a cos(m theta)]` makes
    the two orientations inequivalent: the x-varying ridges, whose gradient
    points along x and so samples the stiff end of B near theta=0, are
    erased, while the y-varying ridges near the soft theta=pi/2
    (cos(3 pi) = -1) survive. The right panel is what
    `energy_ky_frac = 0.751` in `nanosurface_anisotropic.csv` looks like as
    a surface.

    Do not read the panel amplitudes off the isolated single-orientation
    rates B_0(1 +- eps_a)k^4: theta is the orientation of the *combined*
    gradient of both ridge sets, so the two do not decay independently
    (`docs/report/07_surface_diffusion.qmd` makes the same point about the
    measured 3:1 split).

    t=8 is the shipped `t1`, not an extension: h decays as exp(-B k^4 t),
    so the +-0.1 initial corrugation is down to +-0.005 here and running
    further leaves nothing to see. `h` is a signed height about a conserved
    mean of zero, so the map is diverging and centred on 0 -- the shared
    scale also carries the *amplitude* difference (RMS roughness 0.00238 vs
    0.00155), which a per-panel autoscale would have hidden.
    """
    run_dir = DATA_DIR / "surface_diffusion_nanosurface" / "results" / "surface_diffusion"
    isotropic = read_vti(run_dir / "nanosurface_isotropic_0016.vti", time=8.0)
    anisotropic = read_vti(run_dir / "nanosurface_anisotropic_0016.vti", time=8.0)
    fig = render_comparison(
        isotropic,
        anisotropic,
        kind="diverging",
        center=0.0,
        label_a="isotropic ($\\epsilon_a=0$)",
        label_b="anisotropic ($\\epsilon_a=0.5$, $m=6$)",
        suptitle="Same corrugated surface, same t=8: sixfold stiffness picks an orientation",
        cbar_label="surface height h",
        axis_units="grid units",
    )
    out = HERE / "surface_diffusion_nanosurface_comparison.svg"
def figure_kawahara_solitary_radiation():
    """A KdV solitary wave with and without the fifth-order term.

    The only 1-D application in the report, and the reason
    `field_plots.render_line_comparison` exists: a `512 x 1 x 1` snapshot
    drawn as an image would be a one-pixel stripe, and the signal here is
    an *amplitude* (a shed wave train at a few percent of the pulse
    height), which a colour bar reads worst and a linear u axis reads best.

    Both panels are the same solitary wave, the same `alpha` and `beta`
    (the tau=0.30 capillary-gravity mapping), the same 20000 steps. The
    control (`nonlinear_pulse_kdv_only.json`, `gamma=0`) is an *exact*
    solution of the equation it is run against, so it is not merely a
    baseline: every difference in the right-hand panel is attributable to
    `gamma`. Read across at t=100: the control's peak is unmoved to 1.4
    parts in 1e4 and its line is flat everywhere else, while the full run
    has given up 28% of its peak and filled the line with a wave train.

    x is in physical units, which takes a deliberate correction: the JSON
    session's VTK writer emits `Spacing="1 1 1"` regardless of
    `domain.dx`, so `with_spacing(..., dx=0.25)` restates the extent as the
    128-long line the input actually describes (see `field_io.with_spacing`).
    The axis is checkable, and checks out: the t=0 peak lands at x=32.0,
    the input's `x0`, and the t=100 peak at x=34.5, the `peak_x` the run's
    own `diagnostics.csv` records.
    """
    run_dir = DATA_DIR / "kawahara_solitary" / "results" / "kawahara"
    saveat, dx = 2.0, 0.25
    indices = [0, 25, 50]

    def series(stem):
        return [
            with_spacing(
                read_vti(run_dir / f"{stem}_u_{i:04d}.vti", time=i * saveat), dx=dx
            )
            for i in indices
        ]

    fig = render_line_comparison(
        series("nonlinear_pulse_kdv_only"),
        series("nonlinear_pulse_kawahara"),
        label_a="KdV control, $\\gamma=0$ (exact solitary wave)",
        label_b="Full Kawahara, $\\gamma=1/90$",
        labels=[f"t={i * saveat:g}" for i in indices],
        suptitle="The fifth-order term turns an exact solitary wave into a radiating one",
        xlabel="x (code length units)",
        ylabel="surface displacement u",
        annotate_a="tail RMS 4.19e-05 — the sech² skirt, no wave train",
        annotate_b="tail RMS 3.13e-03 (75×), peak down 28%",
    )
    out = HERE / "kawahara_solitary_radiation.svg"
def figure_higher_order_pfc_lattice_comparison():
    """Which 2D lattice the PFC correlation kernel selects, out of pure noise.

    The crystal-selection claim of `apps/higher_order_pfc` in one picture:
    same 128x128 periodic box, same `seeded_noise` initial condition (seed
    42, amplitude 0.01 about psi_bar=-0.15), same quench (eps=0.25, g=0.5),
    same t1=400. The only difference between the two panels is the
    correlation kernel -- `single_mode_triangular.json` has one band, at
    |k|=1; `two_mode_square.json` adds a second at |k|=q1=sqrt(2) with
    r1=0.02 -- and the lattice that grows out of the noise changes with it.

    Why a 64x64 detail and not the whole box: the lattice period is 8 cells
    (|k|=1 sits on grid mode 16 of 128), so at print size the whole field
    is a texture rather than a countable arrangement of peaks. The central
    quarter is where a reader can actually count neighbours around a bright
    peak -- six at 60 degrees on the left, four at 90 on the right -- which
    is the whole point. What the crop hides is that both runs are
    *polycrystalline*: the diagnostics for these exact runs report
    psi6=(global 0.335, local 0.852) for the single-mode case and
    psi4=(0.341, 0.806) for the two-mode one, and a global average well
    below the local one means grains at different orientations partially
    cancelling. Grain boundaries are visible in the detail as the faint
    seams where the packing slips.

    The colour scale is diverging and centred on psi_bar=-0.15, the
    conserved mean density, because a PFC density is only meaningful as a
    departure from its own mean: the peaks are lattice sites, the troughs
    the interstitial space between them.
    """
    tri = read_vti(
        DATA_DIR / "higher_order_pfc_single_mode" / "results" / "higher_order_pfc"
        / "single_mode_0020.vti",
        time=400.0,
    )
    square = read_vti(
        DATA_DIR / "higher_order_pfc_two_mode" / "results" / "higher_order_pfc"
        / "two_mode_0020.vti",
        time=400.0,
    )
    fig = render_comparison(
        _center_crop(tri, 64),
        _center_crop(square, 64),
        kind="diverging",
        center=-0.15,
        label_a="single-mode $k^4$: six neighbours (triangular)",
        label_b="two-mode $k^8$, $r_1{=}0.02$: four (square)",
        suptitle="Same box, same noise, same quench, $t=400$: the kernel picks the lattice",
        cbar_label="density $\\psi$",
        axis_units="grid cells",
    )
    out = HERE / "higher_order_pfc_lattice_comparison.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_wave2d_wall_reflection():
    """The same acoustic pulse against a pressure-release and a rigid wall.

    `apps/wave2d`'s chapter asks whether the character of the reflection
    depends on the wall type, and this is that question rendered: one
    Gaussian pulse released at the centre of a 192x96 slab, periodic in x
    and walled in y, run twice with only `y_bc` changed. At t=50 the
    outgoing ring has reached both walls and come back. Along y=0 and
    y=95 the Dirichlet run shows the reflected crest as a *trough* (the
    odd mirror inverts it, and u is pinned to 0 on the wall itself) while
    the Neumann run shows it as a crest of nearly twice the incident
    amplitude (the even mirror adds to it). The shared diverging scale
    centred on u=0 is what makes "inverted" legible as a colour flip
    rather than as two separately normalised pictures.

    t=50 (step 1000 of 1400) rather than the end of the run: by t=70 the
    reflections from both walls have crossed and the pattern is a
    reverberation rather than a reflection, which answers a different
    question. dt=0.05 is well inside @eq-w2-cfl; the drivers step with
    explicit Euler, which is only weakly unstable for this system, and at
    this dt the growth over 1400 steps stays in the last digits of the
    smooth initial data.
    """
    run_dir = DATA_DIR / "wave2d_walls" / "results" / "wave2d"
    step, dt = 1000, 0.05
    dirichlet = read_vti(run_dir / f"dirichlet_u_{step:04d}.vti", time=step * dt)
    neumann = read_vti(run_dir / f"neumann_u_{step:04d}.vti", time=step * dt)
    fig = render_comparison(
        dirichlet,
        neumann,
        kind="diverging",
        center=0.0,
        label_a="Dirichlet y-walls (pressure release)",
        label_b="Neumann y-walls (rigid)",
        suptitle="Same pulse, same t=50: only the wall condition differs",
        cbar_label="displacement u",
        axis_units="grid units",
        figsize=(9.2, 3.4),
    )
    out = HERE / "wave2d_wall_reflection.svg"
def figure_gradient_elasticity_inclusion_comparison():
    """What an internal length does to the stress around a misfitting inclusion.

    `circular_inclusion.json`'s 256^2 box, a tanh-smoothed disk of radius
    R=32 with a dilatational eigenstrain eps0=0.01, solved once (this
    application has no time loop). Left: the shipped ell=8, i.e. R/ell=4.
    Right: the same problem with ell=0, the classical Navier limit the
    Verification section tests against. Everything else -- R, the box, the
    moduli, the interface width -- is identical, so the pair is a
    size-effect statement, not two unrelated pictures.

    What to look at is the *rim*, not the core. Classically the inclusion
    is a nearly uniform patch of compression with a smooth edge and no
    tensile region at all (the whole field is negative). Switching the
    internal length on adds an oscillatory boundary layer of width ~ell at
    the interface: a tensile ring outside the inclusion and a deeper
    compressive ring just inside it. That layer is why the peak
    |sigma_h| goes *up*, from 0.01466 to 0.02133 (+45%), and it is the
    counter-intuitive direction worth pausing on -- gradient elasticity
    regularises fields that were singular, but this inclusion has no
    singularity to regularise, so the (1 + ell^2 k^2) factor simply
    stiffens the medium at the short wavelengths the interface is made of.
    The far field is untouched, exactly as the model promises.

    The colour scale is diverging and centred on zero because the sign of
    the hydrostatic stress is the physics: negative is compression,
    positive tension, and whether a tensile ring exists at all is the
    difference between the two panels.

    One caveat on those two peak numbers. `scripts/size_sweep.py` sizes its
    box as L = 16*max(R, ell); the shipped `circular_inclusion.json` is
    256^2 with R=32, so it sits at a ratio of 8, where the chapter's own
    periodic-image control measures a few per cent of error on the peak.
    Both panels use the same box, so the comparison between them is clean;
    the absolute values are not sweep-grade.
    """
    gradient = read_vti(
        DATA_DIR / "gradient_elasticity_gradient" / "results" / "gradient_elasticity"
        / "inclusion_stress_hydro_0000.vti"
    )
    classical = read_vti(
        DATA_DIR / "gradient_elasticity_classical" / "results" / "gradient_elasticity"
        / "inclusion_stress_hydro_0000.vti"
    )
    fig = render_comparison(
        gradient,
        classical,
        kind="diverging",
        center=0.0,
        label_a="$\\ell=8$ ($R/\\ell=4$): peak $0.0213$, tensile ring",
        label_b="$\\ell=0$ (classical): peak $0.0147$",
        suptitle="Misfitting inclusion, $R=32$: the internal length rebuilds the interface",
        cbar_label="hydrostatic stress $\\sigma_h$",
        axis_units="cells, $\\Delta x = 1$",
    )
    out = HERE / "gradient_elasticity_inclusion_comparison.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_ehd_film_comparison():
    """Compliant vs stiff plate under the same load, at the instant it lifts.

    `load_relaxation_compliant.json` (B=100) and
    `load_relaxation_stiff.json` (B=640) apply the identical Gaussian press
    (p0=0.5, width a=8) over 0 <= t < 60 to an initially uniform gap
    h0=1, and differ only in the plate's bending stiffness. Save index 6 is
    t=60, the moment the load comes off -- also where both runs' CSV records
    their minimum central gap, so this is the deepest the dent ever gets.

    The two panels answer the chapter's question in one look. The compliant
    plate dents *deeper* (h_centre 0.659 against 0.747) and *narrower* (RMS
    spreading radius 12.3 against 14.2); the stiff plate spreads the same
    displaced volume over a wider, shallower depression, ringed by the
    slight bulge where the liquid pushed out has to go. Both dents sit well
    inside the 256-cell periodic box, which is the visual form of the
    chapter's claim that the measured spreading radius is measuring the
    disturbance and not the domain.

    A gap thickness cannot go negative, so the map is sequential; the shared
    scale is what makes "deeper" and "shallower" comparable rather than two
    separately autoscaled blobs that would look identical.
    """
    run_dir = DATA_DIR / "ehd_film_load" / "results" / "ehd_film_nonlinear"
    compliant = read_vti(run_dir / "load_relaxation_compliant_0006.vti", time=60.0)
    stiff = read_vti(run_dir / "load_relaxation_stiff_0006.vti", time=60.0)
    fig = render_comparison(
        compliant,
        stiff,
        kind="sequential",
        label_a="compliant plate ($B=100$)",
        label_b="stiff plate ($B=640$)",
        suptitle="Same press, same instant t=60: plate stiffness sets depth against width",
        cbar_label="gap thickness h",
        axis_units="grid units",
    )
    out = HERE / "ehd_film_load_comparison.svg"
def figure_allen_cahn_growth_montage():
    """A favoured Allen-Cahn grain taking over, with its own observable on it.

    `apps/allen_cahn` writes no `.vti` and no `.bin`: its only field output
    is the grayscale PNG pair that `pfc::io::write_mpi_scalar_field_png_xy`
    emits, so this figure is rendered through `field_io.read_gray_png`,
    which inverts that writer's fixed `[-1, 1] -> [0, 255]` map. The
    inversion is exact enough to be checked: the superlevel-set areas
    recovered from the PNGs (872, 3324, 6176, 9896, 14556 cells) are the
    *same integers* the program printed for its own exit-code criterion.

    Two honest caveats, both from the writer's clipping rather than from
    this reader. The driving force F=10 shifts the wells, so the run's
    phi actually spans [-0.69, +1.15]: the grain interior is saturated at
    +1 in the PNG and cannot be recovered. That is harmless here because
    the figure's subject is the interface and the area inside it, and the
    level set that defines the observable, phi=0, sits in the middle of
    the recoverable range. The matrix relaxing from -1 (panel 1) to -0.69
    (later panels) is the same well shift, and is real, not an artefact.

    The application has no output cadence -- at most two PNGs per run, the
    initial and the final state -- so the time series is five *separate
    runs* at increasing `n_steps`. The initial condition is deterministic
    and there is no noise anywhere in the model, so those five runs sample
    one trajectory rather than five.

    Why 256^2 and not the default 64^2: at 64^2 the grain reaches the
    periodic boundary and merges with its own images by 40000 steps
    (the superlevel area saturates at the full 4096 cells), which is
    exactly the ceiling the chapter warns about, and makes the last panel
    a picture of the box rather than of a grain.
    """
    run_dir = DATA_DIR / "allen_cahn_growth"
    dt = 9e-5
    # (n_steps, measured A(t)/A(0) printed by the run itself)
    samples = [(0, 1.0), (10000, 3.81), (20000, 7.08), (30000, 11.35), (40000, 16.69)]
    fields = [
        read_gray_png(run_dir / f"phi_s{n:05d}.png", vmin=-1.0, vmax=1.0,
                      name="phi", time=n * dt)
        for n, _ in samples
    ]
    fig = render_montage(
        fields,
        kind="diverging",
        center=0.0,
        panel_titles=[
            f"t={n * dt:g}\n$A/A_0$ = {r:g}" if n else f"t=0\n$A_0$ = 872 cells"
            for n, r in samples
        ],
        suptitle="Allen-Cahn: the favoured phase takes over (256², F=10)",
        cbar_label="order parameter φ",
        axis_units="grid units",
        panel_size=(2.5, 2.7),
    )
    out = HERE / "allen_cahn_growth_montage.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


def figure_kobayashi_dendrite_montage():
    """A sixfold Kobayashi dendrite, and the point at which the box stops it.

    The chapter's question is a morphology question -- disc or arms? -- and
    the answer is in the first panel already: the 2.2-cell nucleus (too
    small to see at t=0, so t=0 is not shown) throws out six primary arms in
    the directions the sixfold anisotropy prefers, then decorates them with
    side branches. Nothing in the model prescribes six arms; they come from
    epsilon(theta) = eps_bar[1 + delta*cos(6(theta - theta_0))] alone.

    The last panel is included on purpose rather than cropped away. The box
    is a periodic torus with no heat sink, so integrating the temperature
    equation gives d<T>/dt = kappa d<phi>/dt exactly: the melt warms in
    strict proportion to the solidified fraction, and growth arrests when
    <T> reaches T_eq = 1, i.e. at a solid fraction of 1/kappa = 0.556 --
    independent of box size. This run confirms both halves of that on its
    own numbers: at t=1 the recovered solid fraction is 0.4319 and the
    printed <T> is 0.7773 = 1.8 x 0.4319 to four figures. By that point the
    arms have reached the periodic boundary and are about to merge with
    their own images, which is exactly the ceiling the chapter warns about,
    so t=1 is where a reader should stop treating this as one dendrite in a
    melt.

    Rendered through `field_io.read_gray_png`: this application writes no
    `.vti` and no `.bin`, only the `[0, 1]`-clipped grayscale PNG series
    that `pfc::io::write_mpi_scalar_field_png_xy` emits every `kNsave`
    steps. phi is bounded in [0, 1] by construction here, so unlike the
    Allen-Cahn figure nothing is lost to the writer's clipping; the only
    cost is 1/255 quantisation, and the recovered solid fraction still
    matches the run's own `sum_phi` to four figures. The map is sequential,
    not diverging: phi is one-sided (0 = liquid, 1 = solid) with no
    physically meaningful midpoint to centre on.

    512^2 rather than the default 256^2: see `run_field_demos.sh`. At 256^2
    the same physics reaches the arrest fraction by running its arms into
    the boundary, and the late frames show a lattice rather than a dendrite.
    """
    N, dx, dt, nsave = 512, 0.03, 1e-4, 2000
    grid = GridSpec(nx=N, ny=N, dx=dx, dy=dx, origin="corner")
    run_dir = DATA_DIR / "kobayashi_dendrite" / "results" / "kobayashi"
    # (frame index, solid fraction measured from the run's own sum_phi)
    frames = [(1, 0.024), (2, 0.076), (3, 0.160), (4, 0.278), (5, 0.432)]
    fields = [
        read_gray_png(run_dir / f"phi_{i:04d}.png", vmin=0.0, vmax=1.0, grid=grid,
                      name="phi", time=i * nsave * dt)
        for i, _ in frames
    ]
    fig = render_montage(
        fields,
        kind="sequential",
        panel_titles=[
            f"t={i * nsave * dt:g}\n{frac:.1%} solid" for i, frac in frames
        ],
        suptitle=(
            "Sixfold dendrite in undercooled melt; growth arrests at 55.6% solid "
            "(1/κ) because the torus has no heat sink"
        ),
        cbar_label="phase field φ  (0 = liquid, 1 = solid)",
        axis_units="model length units",
        panel_size=(2.5, 2.7),
    )
    out = HERE / "kobayashi_dendrite_montage.svg"
def figure_aluminum_fcc_nucleus_comparison():
    """Is an FCC seed in an aluminium melt above or below the critical size?

    `inputs_json/fcc_seed_nucleus.json`: 192^3 cells at dx = 2*pi*sqrt(3)/8,
    i.e. 261 reduced length units per side and 24 FCC lattice constants
    (a = 2*pi*sqrt(3) = 10.88), holding one randomly oriented FCC seed in a
    melt at n0=-0.006, isothermal at T_const=980 (G_grid = V_grid = 0, as
    in every shipped aluminium input). Two runs, identical but for the seed
    radius, both at t=200, both sliced through mid-depth.

    Both seeds shrink at first, and that is the point: `seed_grid_fcc`
    writes a diffuse profile (amplitude 0.4, peak psi 3.16) that has to
    relax onto the model's own much larger solid amplitude, and the radius
    it loses paying for that relaxation is what decides its fate. The
    radius-60 seed survives the transient: its effective radius falls from
    59.6 to a minimum of 52.0 around t=500, then turns around and *grows*
    (54.6 at t=1000, dr/dt still rising), while the peak density climbs
    3.16 -> 4.69 -> 5.00. The radius-30 seed does not: its envelope decays
    monotonically -- 0.84, 0.50, 0.29, 0.11, 0.03 at t=0,50,100,150,200 --
    and by t=200 the box is uniform melt again. Radius 50 is on the losing
    side too, taking until t~700 to vanish, so the critical radius at these
    parameters sits between 50 and 60 reduced units: about five FCC lattice
    constants. (Radii measured as the volume where |psi - n0|, box-blurred
    over one lattice constant, exceeds half its own 99.9th percentile, then
    turned into a sphere-equivalent radius. All four radii were run to
    t=1000 to establish this; the shipped preset stops at t=200 because
    that is where the figure is taken.)

    This is worth a figure because it sharpens the natural reading of the
    application. The shipped aluminium configuration is usually described
    as a seeded *growth* run. With the thermal drive off it is really a
    *nucleation* one: the melt is barely undercooled, growth is slow and
    only begins after a long amplitude transient, and whether a seed grows
    at all depends on its radius.

    Read from `.bin` rather than `.vti` on purpose: `pfc::VTKWriter` emits
    `Origin="0 0 0" Spacing="1 1 1"` whatever the domain's real origin and
    dx, so a `.vti` can only be plotted in cells. The raw brick plus an
    explicit `GridSpec` puts reduced PFC length units on the axes, in which
    the reader can measure the lattice spacing off the figure. The colour
    scale is diverging and centred on n0=-0.006, the melt density: what
    should stand out is where psi departs from the surrounding liquid.

    Two presentation choices, both deliberate. The central 144 of 192 cells
    are shown, because the outer ring is undisturbed melt -- beyond 100
    reduced units from the centre, |psi - n0| peaks at 0.106 against the
    crystal's 4.4, so it contributes nothing but white space. And the
    colour limits are clamped to +-2.0 rather than the data range: the
    solid's density peaks reach 4.42 but only on 0.9% of cells, and letting
    them set the scale washes the whole lattice out to pale pink. Clipping
    that 0.9% is what makes the lattice sites countable; the numbers the
    reader might want off the scale are in this docstring and the caption
    instead.
    """
    grid = GridSpec(
        nx=192, ny=192, nz=192,
        dx=1.3603495232, dy=1.3603495232, dz=1.3603495232,
        origin="corner",
    )
    supercritical = slice_bin(
        DATA_DIR / "aluminum_supercritical" / "results" / "aluminum" / "psi_0004.bin",
        grid, axis="z", name="psi", time=200.0,
    )
    subcritical = slice_bin(
        DATA_DIR / "aluminum_subcritical" / "results" / "aluminum" / "psi_0004.bin",
        grid, axis="z", name="psi", time=200.0,
    )
    fig = render_comparison(
        _center_crop(supercritical, 144),
        _center_crop(subcritical, 144),
        kind="diverging",
        center=-0.006,
        vmin=-2.0,
        vmax=2.0,
        label_a="seed radius 60: a stable nucleus",
        label_b="seed radius 30: re-melted",
        suptitle="Aluminium FCC seed at $t=200$, isothermal, mid-depth slice",
        cbar_label="density $\\psi$",
        axis_units="reduced PFC units",
    )
    out = HERE / "aluminum_fcc_nucleus_comparison.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    return out, fig


FIGURES = [
    figure_cahn_hilliard_coarsening_montage,
    figure_cahn_hilliard_comparison,
    figure_thin_film_dewetting_montage,
    figure_thin_film_comparison,
    figure_tungsten_seed_panel,
    figure_surface_diffusion_comparison,
    figure_ehd_film_comparison,
    figure_kawahara_solitary_radiation,
    figure_wave2d_wall_reflection,
    figure_allen_cahn_growth_montage,
    figure_kobayashi_dendrite_montage,
    figure_higher_order_pfc_lattice_comparison,
    figure_gradient_elasticity_inclusion_comparison,
    figure_aluminum_fcc_nucleus_comparison,
]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if not DATA_DIR.exists():
        raise SystemExit(
            f"FIELD_DATA_DIR={DATA_DIR} does not exist. Run run_field_demos.sh first "
            "(see docs/report/README.md)."
        )
    for make in FIGURES:
        out, fig = make()
        plt.close(fig)
        print("wrote", out.relative_to(HERE.parent.parent.parent))
