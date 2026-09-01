#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Two-grain noise sanity check before enabling F0 on production runs.
#   Same small box as lumi_bicrystal_probe.sh (W0 = 20 nm, 2.0 x 1.60 um).
#   Sweeps F0 = 0, 1e-4, 1e-3, 1e-2 and reports whether the superheated
#   liquid ahead of the front nucleates spuriously.
#
#   sbatch --dependency=afterok:$BUILD_ID apps/alloy_pf_directional/scripts/lumi_noise_probe.sh
#SBATCH --job-name=alcu-noise
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1750
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/noise-%j.out
set -euo pipefail

# shellcheck source=lumi_paths.sh
source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh

module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export OPENPFC_ALCU_QUIET=1
export OPENPFC_ALCU_SKIP_VTK=1
export OPENPFC_ALCU_SKIP_PNG=0
export OPENPFC_ALCU_WINDOW=0
export OPENPFC_ALCU_STOP_FAR_C=0
export OPENPFC_ALCU_STOP_RIGHT=0
export OPENPFC_ALCU_PERIODIC_Y=1
export OPENPFC_ALCU_G=3.0e6
export OPENPFC_ALCU_VP=0.4
# Long enough in x that liquid survives ahead of the front, otherwise the
# "did noise nucleate anything ahead?" test has nothing to look at.
export OPENPFC_ALCU_LX=8.0e-6
export OPENPFC_ALCU_LY=1.60e-6
export OPENPFC_ALCU_TEND=55.0e-6
export OPENPFC_ALCU_SEED=0.20e-6
export OPENPFC_ALCU_THETA=30
export OPENPFC_ALCU_W0=20e-9
export OPENPFC_ALCU_DXW=1.0
export OPENPFC_ALCU_DT_OVER_TAU=0.2
export OPENPFC_ALCU_MAX_STEPS=5000
export OPENPFC_ALCU_NGRANS=2

run_one() {
  local f0="$1" tag="$2"
  local out="${LUMI_RUNS}/bicrystal/noise_${tag}"
  rm -rf "${out}"; mkdir -p "${out}"
  echo "=== F0 = ${f0} ==="
  OPENPFC_ALCU_NOISE="${f0}" srun --cpu-bind=cores \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${LUMI_BIN}" ds "${out}" "${SLURM_CPUS_PER_TASK}" \
    --save-every 1000 --log-every 100 || true
  [[ -f "${out}/abort.txt" ]] && { echo "# abort"; cat "${out}/abort.txt"; }
  python3 - "${out}" "${f0}" <<'PY' || true
import struct, sys, pathlib
root, f0 = pathlib.Path(sys.argv[1]), sys.argv[2]
meta = {}
for line in (root/"meta.txt").read_text().splitlines():
    a = line.split()
    if len(a) >= 2:
        try: meta[a[0]] = float(a[1])
        except ValueError: pass
Nx, Ny = int(meta["Nx"]), int(meta["Ny"])
n = Nx*Ny
def load(name):
    p = root/name
    return list(struct.unpack(f"{n}d", p.read_bytes()[:n*8])) if p.exists() else None
p1, p2, c = load("phi_final.raw"), load("phi2_final.raw"), load("c_final.raw")
if p1 is None or c is None:
    print("  missing raw"); sys.exit(0)
psi = [max(-1.0, min(1.0, p1[i] + (p2[i] if p2 else -1.0) + 1.0)) for i in range(n)]
# solid fraction per column, to locate the connected front
col = [sum(1 for j in range(Ny) if psi[j*Nx+i] > 0.0)/Ny for i in range(Nx)]
front = max((i for i in range(Nx) if col[i] > 0.5), default=0)
ahead = list(range(min(Nx, front + 20), Nx))    # >20 cells beyond the front
iso = sum(1 for i in ahead for j in range(Ny) if psi[j*Nx+i] > 0.0)
nahead = max(1, len(ahead)*Ny)
# how far the bulk liquid was disturbed: max phi1 well ahead of the front
far = list(range(min(Nx, front + 60), Nx))
far_max = max((p1[j*Nx+i] for i in far for j in range(Ny)), default=-1.0)
# interface roughness = spread of the phi=0 crossing per row (sidebranching)
xs = []
for j in range(Ny):
    row = [psi[j*Nx+i] for i in range(Nx)]
    cx = [i for i in range(Nx-1) if row[i] >= 0.0 > row[i+1]]
    if cx: xs.append(max(cx))
rough = (max(xs)-min(xs)) if xs else 0
finite = all(x == x and abs(x) < 1e6 for x in c)
print(f"  frac_solid={sum(1 for v in psi if v>0)/n:.4f}  front_col={front}/{Nx}  liquid_ahead={len(ahead)*Ny}  "
      f"solid_ahead={iso} ({iso/nahead:.1e})  max_phi1_far={far_max:+.6f}  "
      f"front_roughness={rough} cells  c_finite={finite}  c_max={max(c):.4g}")
PY
}

run_one 0     f0_0
run_one 1e-4  f0_1em4
run_one 1e-3  f0_1em3
run_one 1e-2  f0_1em2
echo "noise probe done"
