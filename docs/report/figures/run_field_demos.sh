#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Regenerate the raw field data (.vti / .bin / .png) used by make_field_figures.py.
#
# This is the "run recipe" for the field-visualisation figures: the report
# commits the rendered SVGs, not multi-megabyte simulation output, so a
# reader who wants to reproduce a figure from scratch runs this script
# against a build tree, then re-runs make_field_figures.py.
#
# Usage:
#   ./scripts/build.sh --machine=lumi --cpu --no-submit --no-test \
#       --build-dir=<your build dir>          # see AGENT_NOTES / root README
#   docs/report/figures/run_field_demos.sh <build dir>
#
# Environment:
#   FIELD_DATA_DIR   Where run output is written (default: ./_field_demo_data
#                     next to wherever you invoke this from). Point this at
#                     shared/fast storage on a cluster.
#   RUNNER            Launcher command (default: "mpirun -n 1"). Set it to
#                     the empty string to run the binaries directly, with no
#                     launcher at all -- which is what a LUMI login node
#                     needs, since it has srun but no mpirun and these runs
#                     are single rank anyway. (Empty is honoured: the
#                     default below uses `${RUNNER-...}`, not
#                     `${RUNNER:-...}`.) On LUMI,
#                     with the shared allocation described in AGENT_NOTES.md:
#                       SLURM_JOB_ID=<job> TMPDIR=<shared tmp> \
#                       RUNNER="srun --overlap -n 1" \
#                       docs/report/figures/run_field_demos.sh <build dir>
#                     (SLURM_JOB_ID/TMPDIR are read from the environment by
#                     srun itself; export them before calling this script.)
#
# Every app here is a single-field, single-rank demo: `-n 1` is enough and
# keeps each run trivial to place in its own output directory. Three of them
# (cahn_hilliard, thin_film, tungsten) go through the JSON
# `SpectralETDSession`; the other two (surface_diffusion_anisotropic,
# ehd_film_nonlinear) are standalone science drivers whose `fields[]` output
# is written by `openpfc_apps/field_snapshots.hpp` instead, using the same
# JSON spelling.
# Every run here is single-rank (`-n 1`): the spectral-ETD demos are small
# enough not to need more, `apps/kawahara`'s cases are 1-D lines that HeFFTe
# cannot split at all (`Ny = Nz = 1`), and the finite-difference CLI demos
# are cheap. One rank also keeps each run trivial to place in its own output
# directory.
#
# Three of the runs below (wave2d, allen_cahn, kobayashi) are command-line
# applications with no JSON input at all, so they take positional arguments
# rather than a config file and $RUNNER is applied to them directly.
# Every run here is single rank (`-n 1`): enough for these grid sizes, and it
# keeps each run trivial to place in its own output directory. It is also a
# hard requirement for `higher_order_pfc`, whose real-space order metric needs
# the whole grid on one rank (see `apps/higher_order_pfc/diagnostics.hpp`).
#
# Rough single-core cost on a LUMI login node, for planning: the four 2-D runs
# (cahn_hilliard x2, thin_film x2) are seconds; higher_order_pfc x2 and
# gradient_elasticity x2 are seconds to tens of seconds; tungsten (256^3) and
# aluminum (192^3 x2) are the expensive ones, a few minutes each.

set -Eeuo pipefail

if [ -z "${BASH_VERSION-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${1:-build}"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
DATA_DIR="${FIELD_DATA_DIR:-$(pwd)/_field_demo_data}"
RUNNER="${RUNNER-mpirun -n 1}"

mkdir -p "$DATA_DIR"

run_case() {
  local app="$1" bin="$2" config="$3" outdir="$4" resultsdir="$5"
  local rundir="$DATA_DIR/$outdir"
  mkdir -p "$rundir/$resultsdir"
  echo "==> $outdir"
  (cd "$rundir" && $RUNNER "$BUILD_DIR/apps/$app/$bin" "$config")
}

# --- cahn_hilliard: Fe-Cr spinodal decomposition -------------------------
# Two initial conditions from the shipped inputs, run into separate
# directories so both sets of results/cahn_hilliard/c_%04d.vti survive:
#   single-mode growth (fe_cr_spinodal.json) vs broadband coarsening
#   (coarsening.json) -- the pair render_comparison() puts side by side.
run_case cahn_hilliard cahn_hilliard \
  "$REPO_ROOT/apps/cahn_hilliard/inputs_json/fe_cr_spinodal.json" \
  cahn_hilliard_single results/cahn_hilliard
run_case cahn_hilliard cahn_hilliard \
  "$REPO_ROOT/apps/cahn_hilliard/inputs_json/coarsening.json" \
  cahn_hilliard_coarsening results/cahn_hilliard

# --- thin_film: dewetting vs leveling -------------------------------------
# dewetting.json (A=0.05, unstable) and leveling.json (A=0, stable) write
# distinct filenames (h_%04d.vti / leveling_h_%04d.vti) so they can share
# one results/thin_film/ directory.
#
# Both configs are patched to run further than their shipped t1 so the
# instability visibly deepens (the shipped t1=40 default only shows a
# +-2% -> +-4% ripple). t1=120/saveat=12 is close to the largest range
# that stays numerically sane for `dewetting.json`: by t=96 the disjoining-
# pressure nonlinearity gets stiff enough that this explicit-nonlinearity
# ETD scheme diverges (see docs/report/figures/make_field_figures.py). Do
# not raise t1 further without checking min/max stay in (0, 2*h0).
# `leveling.json` (A=0, no destabilising term) is run to the same t1/saveat
# purely so the two land on identical sample times for the comparison
# figure; it has no such instability.
mkdir -p "$DATA_DIR/thin_film_dewetting/results/thin_film"
echo "==> thin_film (dewetting, t1/saveat extended to 120/12)"
python3 -c '
import json
d = json.load(open("'"$REPO_ROOT"'/apps/thin_film/inputs_json/dewetting.json"))
d["timestepping"]["t1"] = 120.0
d["timestepping"]["saveat"] = 12.0
json.dump(d, open("'"$DATA_DIR"'/thin_film_dewetting/dewetting_extended.json", "w"), indent=2)
'
(cd "$DATA_DIR/thin_film_dewetting" && $RUNNER "$BUILD_DIR/apps/thin_film/thin_film" \
  dewetting_extended.json)
echo "==> thin_film (leveling, t1/saveat matched to 84/12)"
python3 -c '
import json
d = json.load(open("'"$REPO_ROOT"'/apps/thin_film/inputs_json/leveling.json"))
d["timestepping"]["t1"] = 84.0
d["timestepping"]["saveat"] = 12.0
json.dump(d, open("'"$DATA_DIR"'/thin_film_dewetting/leveling_extended.json", "w"), indent=2)
'
(cd "$DATA_DIR/thin_film_dewetting" && $RUNNER "$BUILD_DIR/apps/thin_film/thin_film" \
  leveling_extended.json)

# --- tungsten: 3D PFC single-seed nucleus ---------------------------------
# Deliberately NOT apps/tungsten/inputs_json/tungsten_single_seed.json: that
# file's 32^3 domain has a half-width (17.8 reduced units) far smaller than
# `SingleSeed`'s hard-coded seed radius (64, see
# include/openpfc/kernel/simulation/initial_conditions/single_seed.hpp), so
# the "seed" fills the entire periodic box -- no liquid is left to show a
# nucleus in. apps/tungsten/tungsten_single_seed_256_cuda.json's 256^3
# domain (half-width 142) is the smallest shipped config where the seed
# radius leaves a visible liquid background; its timestepping is shortened
# here (t1=12, saveat=2) since this is an illustration, not a scalability
# run, and its output is redirected from `.vti` to `.bin` to exercise the
# raw-binary reader.
mkdir -p "$DATA_DIR/tungsten_seed/results/tungsten"
python3 -c '
import json
d = json.load(open("'"$REPO_ROOT"'/apps/tungsten/tungsten_single_seed_256_cuda.json"))
d["fields"] = [{"name": "psi", "data": "results/tungsten/psi_%04d.bin"}]
d["timestepping"] = {"t0": 0.0, "t1": 12.0, "dt": 1.0, "saveat": 2.0}
json.dump(d, open("'"$DATA_DIR"'/tungsten_seed/run_config.json", "w"), indent=2)
'
echo "==> tungsten_seed (256^3, t1=12)"
(cd "$DATA_DIR/tungsten_seed" && $RUNNER "$BUILD_DIR/apps/tungsten/tungsten" run_config.json)

# --- surface_diffusion: isotropic vs anisotropic nanosurface anneal --------
# The shipped science pair, run unmodified: both presets start from the
# *identical* crossed corrugation (16 periods per axis, amplitude 0.05) and
# differ only in eps_a, so any difference in the final surface is the
# anisotropy and nothing else. They write distinct filenames
# (nanosurface_isotropic_%04d.vti / nanosurface_anisotropic_%04d.vti) so both
# share one results/surface_diffusion/ directory.
#
# Unlike the thin_film runs above, `t1` is NOT extended past the shipped
# value: t1=8 is already where the two runs differ most before the surface
# flattens into round-off. h decays as exp(-B k^4 t) and by t=8 the initial
# +-0.1 corrugation is down to +-0.005 -- the orientation contrast is still
# clearly visible, but running much further leaves nothing to look at.
run_case surface_diffusion surface_diffusion_anisotropic \
  "$REPO_ROOT/apps/surface_diffusion/inputs_json/nanosurface_isotropic.json" \
  surface_diffusion_nanosurface results/surface_diffusion
run_case surface_diffusion surface_diffusion_anisotropic \
  "$REPO_ROOT/apps/surface_diffusion/inputs_json/nanosurface_anisotropic.json" \
  surface_diffusion_nanosurface results/surface_diffusion

# --- ehd_film: compliant vs stiff plate under the same load ----------------
# The shipped science pair, again run unmodified and into one directory
# (load_relaxation_compliant_%04d.vti / load_relaxation_stiff_%04d.vti).
# Same Gaussian load (p0=0.5, a=8) held over 0 <= t < 60 then released; the
# only difference between the two is the bending stiffness B (100 vs 640).
#
# The figure is rendered from save index 6, i.e. t=60 -- the instant the load
# comes off, which is also where the diagnostics CSV records `h_center`'s
# minimum in both runs. Later saves show the dent healing, not the stiffness
# contrast at its clearest. Do not shorten `t1`: the 600-unit tail is what
# the chapter's spreading-radius claim (peaks under 22% of the domain
# half-width) is measured over.
run_case ehd_film ehd_film_nonlinear \
  "$REPO_ROOT/apps/ehd_film/inputs_json/load_relaxation_compliant.json" \
  ehd_film_load results/ehd_film_nonlinear
run_case ehd_film ehd_film_nonlinear \
  "$REPO_ROOT/apps/ehd_film/inputs_json/load_relaxation_stiff.json" \
  ehd_film_load results/ehd_film_nonlinear
# --- kawahara: a KdV solitary wave forced to radiate ----------------------
# The two shipped science-case-B inputs, unmodified: `gamma=0` (an exact KdV
# solitary wave, the control) and `gamma=1/90` (the full Kawahara equation).
# Nothing needs patching here -- both already write `.vti` every `saveat=2`
# to t1=100, and their filenames differ, so one output directory holds both.
# 1-D: `Ny = Nz = 1` means HeFFTe cannot decompose these across ranks.
run_case kawahara kawahara \
  "$REPO_ROOT/apps/kawahara/inputs_json/nonlinear_pulse_kdv_only.json" \
  kawahara_solitary results/kawahara
run_case kawahara kawahara \
  "$REPO_ROOT/apps/kawahara/inputs_json/nonlinear_pulse_kawahara.json" \
  kawahara_solitary results/kawahara

# --- wave2d: a pulse reflecting off a pressure-release and a rigid wall ----
# Same grid, same pulse, same steps; only `y_bc` changes. The figure is the
# A-vs-B comparison, so the two runs MUST agree in everything else.
#
# Parameter choices, none of them arbitrary:
#   192x96   the pulse (sigma = 0.12*min(Nx,Ny) = 11.5) needs room to reach
#            the y walls and come back without also wrapping in the periodic
#            x direction; a 2:1 slab gives 96 grid units of x headroom while
#            the wall is only 48 away.
#   fd_order 2, not higher. `wave2d_fd` advertises even orders 2..20, but the
#            halo width must fit in *every* owned dimension, and this is an
#            nz == 1 slab: order 4 (halo width 2) aborts in
#            pfc::halo::create_padded_face_types_6 before the first step.
#   dt=0.05  the drivers step with explicit Euler, which for this
#            (imaginary-eigenvalue) system is weakly unstable rather than
#            conditionally stable: the amplification per step is
#            sqrt(1 + (dt*omega)^2) for every mode. At dt=0.05 the growth
#            accumulated over 1400 steps is a couple of percent on the
#            physical pulse and stays in the last digits on the grid-scale
#            modes, which have no amplitude in a smooth Gaussian to begin
#            with. Raising dt makes the grid-scale noise visible.
#   1400 steps (t=70) with --vtk-every 200: the figure uses step 1000 (t=50),
#            after the reflection and before the two reflected fronts cross.
mkdir -p "$DATA_DIR/wave2d_walls/results/wave2d"
for bc in dirichlet neumann; do
  echo "==> wave2d_walls ($bc)"
  (cd "$DATA_DIR/wave2d_walls" && $RUNNER "$BUILD_DIR/apps/wave2d/wave2d_fd" \
    192 96 1400 0.05 2 "$bc" 0 \
    --vtk "results/wave2d/${bc}_u_%04d.vti" --vtk-every 200)
done

# --- allen_cahn: a favoured grain growing into the matrix -----------------
# This application has no output cadence: it writes at most two grayscale
# PNGs per run, the initial state and the final one, and nothing in between.
# A time series therefore means several runs at increasing `n_steps`. The
# initial condition is a deterministic Gaussian nucleus with no noise
# anywhere in the model, so these four runs sample one trajectory.
#
# 256^2 rather than the default 64^2: at 64^2 the grain has consumed the
# whole periodic box by 40000 steps (superlevel area saturates at 4096 =
# all cells), so the late panels would show the box, not a grain. Note that
# the program's own 5x-area exit criterion is grid-dependent for the same
# reason -- it passes at the default 64^2/5000 steps (6.1x) and fails at
# 256^2/5000 steps (2.5x); these runs go to 40000 steps, where 256^2
# reaches 16.7x.
#
# `|| true` below is deliberate and is the only place in this script that
# ignores an exit code. `allen_cahn` uses its exit status *as* the 5x-area
# criterion, so a run that stops early on purpose -- which is exactly what
# the first three frames of a time series are -- reports failure by design
# (10000 steps reaches 3.8x). The PNGs are written before the check, so
# they are complete regardless. Do not copy this pattern to the other runs:
# for them a nonzero status means the run actually failed.
mkdir -p "$DATA_DIR/allen_cahn_growth"
for n_steps in 10000 20000 30000 40000; do
  tag=$(printf "%05d" "$n_steps")
  echo "==> allen_cahn_growth ($n_steps steps)"
  (cd "$DATA_DIR/allen_cahn_growth" && $RUNNER "$BUILD_DIR/apps/allen_cahn/allen_cahn" \
    256 256 "$n_steps" 0.00009 8.0 0.19 10.0 phi_s00000.png "phi_s${tag}.png") || true
done

# --- kobayashi: an anisotropic dendrite growing into undercooled melt ------
# Positional arguments are `Nx Ny n_steps dt dx [output_dir]`; the PNG
# cadence is compiled in (`kNsave = 2000`), so 10000 steps gives frames at
# steps 0, 2000, ..., 10000 plus a final one.
#
# 512^2 rather than the default 256^2, and this one is worth being precise
# about. The box is a periodic torus with no heat sink, so latent heat
# accumulates and growth self-limits when <T> reaches T_eq -- that happens
# at a solid *fraction* of about 1/kappa = 0.56 regardless of box size. At
# 256^2 the dendrite reaches that fraction by t=0.8, but only by running its
# arms into the periodic boundary and merging with its own images: the last
# two frames show a lattice, not a dendrite. Quadrupling the melt volume
# (same dx, so the same physics and the same tip scale) keeps the six arms
# clear of the boundary for the whole run. Costs about six minutes on one
# login-node core.
mkdir -p "$DATA_DIR/kobayashi_dendrite/results/kobayashi"
echo "==> kobayashi_dendrite (512^2, 10000 steps)"
(cd "$DATA_DIR/kobayashi_dendrite" && $RUNNER \
  "$BUILD_DIR/apps/kobayashi/kobayashi_fd_manual" 512 512 10000 1.0e-4 0.03 \
  results/kobayashi)
# --- higher_order_pfc: which lattice does the kernel select? --------------
# Both inputs shipped, unmodified: same 128^2 box, same `seeded_noise` IC
# (seed 42, amplitude 0.01), same quench (eps=0.25, g=0.5, psi_bar=-0.15),
# same t1=400. The *only* difference is the correlation kernel --
# `single_mode_triangular.json` is n_modes=1 (one band, at |k|=1) and
# `two_mode_square.json` is n_modes=2 with q1=sqrt(2), r1=0.02 (a second band
# at |k|=sqrt(2)). That is exactly the matched-conditions comparison
# `apps/higher_order_pfc/README.md` tabulates, so do not "tidy up" either
# file's parameters: the point of the figure is that everything except the
# kernel is identical.
#
# Do NOT substitute the two `lattice_seed` presets. Those impose the target
# symmetry as their initial condition, so a figure of them shows only that
# the kernel does not destroy what it was handed; these two grow their
# lattice out of undifferentiated noise, which is the claim worth a picture.
#
# Single rank is mandatory here, not just convenient (see the header note).
run_case higher_order_pfc higher_order_pfc \
  "$REPO_ROOT/apps/higher_order_pfc/inputs_json/single_mode_triangular.json" \
  higher_order_pfc_single_mode results/higher_order_pfc
run_case higher_order_pfc higher_order_pfc \
  "$REPO_ROOT/apps/higher_order_pfc/inputs_json/two_mode_square.json" \
  higher_order_pfc_two_mode results/higher_order_pfc

# --- gradient_elasticity: what the internal length does to an inclusion ---
# `circular_inclusion.json` shipped as-is (256^2, R=32, ell=8, so R/ell=4),
# plus one patched copy with ell=0. This is a one-shot elliptic solve, not a
# time loop, so there is no t1/saveat to extend -- the only knob the figure
# needs is ell, and ell=0 is the classical Navier limit the Verification
# section already tests against. Holding R, the box, eps0 and the moduli
# fixed is what makes the pair a size-effect statement (R/ell = 4 against
# R/ell -> infinity) rather than two unrelated pictures.
#
# The ell=0 copy also drops `line_profile`, which would otherwise overwrite
# the shipped run's CSV in a shared directory; the two runs are kept in
# separate directories anyway because both write `inclusion_*_%04d.vti`.
run_case gradient_elasticity gradient_elasticity \
  "$REPO_ROOT/apps/gradient_elasticity/inputs_json/circular_inclusion.json" \
  gradient_elasticity_gradient results/gradient_elasticity
mkdir -p "$DATA_DIR/gradient_elasticity_classical/results/gradient_elasticity"
echo "==> gradient_elasticity (classical limit, ell=0)"
python3 -c '
import json
d = json.load(open("'"$REPO_ROOT"'/apps/gradient_elasticity/inputs_json/circular_inclusion.json"))
d["model"]["params"]["ell"] = 0.0
d.pop("line_profile", None)
json.dump(d, open("'"$DATA_DIR"'/gradient_elasticity_classical/classical.json", "w"), indent=2)
'
(cd "$DATA_DIR/gradient_elasticity_classical" \
  && $RUNNER "$BUILD_DIR/apps/gradient_elasticity/gradient_elasticity" classical.json)

# --- aluminumNew: is the FCC seed above or below the critical size? -------
# `inputs_json/fcc_seed_nucleus.json` (192^3, one FCC seed of radius 60) run
# twice: as shipped, and with the seed radius dropped to 30. Everything else
# -- box, mean density n0=-0.006, T_const=980, rseed, t1 -- is held fixed, so
# the pair isolates the seed radius.
#
# Why not `inputs_json/smoke.json`: it is a 16^3 *constant* field with no
# seed at all, so every frame it writes is a uniform grey square. And not
# `aluminumNew.json` either: 1024x2048x256 is half a billion cells, far past
# what this script is meant to run.
#
# Why 192^3 and not something cheaper: at 128^3 the same radius-60 nucleus
# leaves only ~70 reduced units between its own periodic images, and the
# melt around it fills with a visible interference pattern from that
# self-interaction. At 192^3 (261 reduced units per side) it does not:
# beyond 100 reduced units from the seed centre |psi - n0| peaks at 0.106
# against the crystal's 4.4, and averages 0.004. Do not shrink the box
# without re-checking that.
#
# Output is redirected from `.vti` to `.bin` deliberately: `pfc::VTKWriter`
# writes `Origin="0 0 0" Spacing="1 1 1"` regardless of the domain's real
# `origin`/`dx`, so a `.vti` can only be plotted in grid cells. Reading the
# raw brick with an explicit `field_io.GridSpec` is what lets the figure
# carry reduced PFC length units, in which the FCC lattice constant
# a = 2*pi*sqrt(3) = 10.88 is a number the reader can measure off the axes.
for al_case in supercritical:60 subcritical:30; do
  al_name="${al_case%%:*}"
  al_radius="${al_case##*:}"
  mkdir -p "$DATA_DIR/aluminum_$al_name/results/aluminum"
  echo "==> aluminum_$al_name (192^3, seed radius $al_radius, t1=200)"
  python3 -c '
import json, sys
radius = float(sys.argv[1])
d = json.load(open("'"$REPO_ROOT"'/apps/aluminumNew/inputs_json/fcc_seed_nucleus.json"))
d["initial_conditions"][1]["radius"] = radius
d["fields"] = [{"name": "psi", "data": "results/aluminum/psi_%04d.bin"}]
json.dump(d, open("'"$DATA_DIR"'/aluminum_'"$al_name"'/run_config.json", "w"), indent=2)
' "$al_radius"
  (cd "$DATA_DIR/aluminum_$al_name" \
    && $RUNNER "$BUILD_DIR/apps/aluminumNew/aluminumNew" run_config.json)
done

echo
echo "Done. Data written under: $DATA_DIR"
echo "Now run: FIELD_DATA_DIR=$DATA_DIR <venv-python> docs/report/figures/make_field_figures.py"
