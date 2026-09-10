#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate the field-visualisation figures for the applications report.

Reads the `.vti` / `.bin` output of a handful of real runs (see
`run_field_demos.sh` for how to reproduce them) and renders SVGs into this
directory using `field_io.py` (readers) and `field_plots.py` (panels,
montages, comparisons). The report itself has no compute engine: it reads
these committed SVGs, not raw simulation output.

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

from field_io import GridSpec, read_vti, slice_bin  # noqa: E402
from field_plots import render_comparison, render_montage, render_panel  # noqa: E402

DATA_DIR = Path(os.environ.get("FIELD_DATA_DIR", Path.cwd() / "_field_demo_data"))


def _vti_series(run_dir: Path, pattern: str, times):
    return [read_vti(run_dir / pattern.format(i), time=t) for i, t in times]


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
