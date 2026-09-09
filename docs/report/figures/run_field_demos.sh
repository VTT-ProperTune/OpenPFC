#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Regenerate the raw field data (.vti / .bin) used by make_field_figures.py.
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
#   RUNNER            Launcher command (default: "mpirun -n 1"). On LUMI,
#                     with the shared allocation described in AGENT_NOTES.md:
#                       SLURM_JOB_ID=<job> TMPDIR=<shared tmp> \
#                       RUNNER="srun --overlap -n 1" \
#                       docs/report/figures/run_field_demos.sh <build dir>
#                     (SLURM_JOB_ID/TMPDIR are read from the environment by
#                     srun itself; export them before calling this script.)
#
# All three apps here (cahn_hilliard, thin_film, tungsten) are single-field,
# single-rank spectral-ETD demos: `-n 1` is enough and keeps the run trivial
# to place in its own output directory.

set -Eeuo pipefail

if [ -z "${BASH_VERSION-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${1:-build}"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
DATA_DIR="${FIELD_DATA_DIR:-$(pwd)/_field_demo_data}"
RUNNER="${RUNNER:-mpirun -n 1}"

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

echo
echo "Done. Data written under: $DATA_DIR"
echo "Now run: FIELD_DATA_DIR=$DATA_DIR <venv-python> docs/report/figures/make_field_figures.py"
