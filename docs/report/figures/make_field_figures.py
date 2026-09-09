#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate the field-visualisation figures for the applications report.

Reads the `.vti` / `.bin` output of three real runs (see
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


FIGURES = [
    figure_cahn_hilliard_coarsening_montage,
    figure_cahn_hilliard_comparison,
    figure_thin_film_dewetting_montage,
    figure_thin_film_comparison,
    figure_tungsten_seed_panel,
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
