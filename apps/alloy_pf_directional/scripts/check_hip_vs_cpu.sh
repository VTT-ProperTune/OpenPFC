#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Step 0 GPU (LUMI-G only): 1 GCD alloy_pf_directional_hip vs a CPU ALCU_VERIFY log.
# 1 rank / GCD, MPICH_GPU_SUPPORT_ENABLED=1. Packed-halo fallback:
#   OPENPFC_HIP_FORCE_PACKED_HALO=1
#
#   sbatch apps/alloy_pf_directional/scripts/check_hip_vs_cpu.sh
#   CPU_LOG=.../openmp.log ./apps/alloy_pf_directional/scripts/check_hip_vs_cpu.sh   # inside a job
#
#SBATCH --job-name=alcu-s0-g
#SBATCH --account=project_462001519
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/step0-g-%j.out
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lumi_paths.sh
if [[ -f "${SCRIPT_DIR}/lumi_paths.sh" ]]; then
  source "${SCRIPT_DIR}/lumi_paths.sh"
else
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh
fi
# shellcheck source=alcu_2d_env.sh
source "${SCRIPT_DIR}/alcu_2d_env.sh" 2>/dev/null || true

GRID="${GRID:-1280x160}"
STEPS="${STEPS:-800}"
alcu_2d_apply_grid "${GRID}"
export OPENPFC_ALCU_MAX_STEPS="${STEPS}"
export OPENPFC_ALCU_TEND="${OPENPFC_ALCU_TEND:-1.0e-6}"
export OPENPFC_ALCU_WARMUP=0
unset OPENPFC_ALCU_TIMED_STEPS || true

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  module purge
  module load "${LUMI_STACK}" partition/G cpeGNU cray-fftw lumi-CrayPath
  lumi_gpu_runtime_env
  export MPICH_GPU_SUPPORT_ENABLED=1
fi

BIN="${LUMI_BIN_HIP}"
OUT="${LUMI_SCALE2D}/step0_gpu/${SLURM_JOB_ID:-manual}"
CPU_LOG="${CPU_LOG:-${LUMI_SCALE2D}/step0_cpu/openmp.log}"
mkdir -p "${OUT}" "${LUMI_LOGS}"
if [[ ! -x "${BIN}" ]]; then
  echo "missing HIP binary ${BIN}" >&2
  echo "build with apps/alloy_pf_directional/scripts/lumi_build_hip.sh or ./scripts/build.sh --machine=lumi --with-rocm" >&2
  exit 1
fi

echo "ALCU_SCALE mode=step0 backend=hip nproc=1 grid=${GRID} packed=${OPENPFC_HIP_FORCE_PACKED_HALO:-0}"
echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-unset}"
WRAP="${SCRIPT_DIR}/lumi_select_gpu.sh"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks=1 --gpus-per-node=1 --cpu-bind=map_cpu:49 \
    "${WRAP}" "${BIN}" ds "${OUT}" | tee "${OUT}/hip.log"
else
  echo "not in a Slurm job — run this via sbatch on LUMI-G" >&2
  exit 1
fi

python3 - "${OUT}/hip.log" "${CPU_LOG}" <<'PY'
import pathlib, re, sys
hip_p, cpu_p = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

def grab(p):
    t = p.read_text() if p.exists() else ""
    m = re.search(r"ALCU_VERIFY (.*)", t)
    if not m:
        return {}
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d

h, c = grab(hip_p), grab(cpu_p)
print("HIP", {k: h.get(k) for k in ("mass1", "x_tip", "sum_phi", "sum_c", "n_steps_done")})
if not h:
    print("FAIL: no HIP ALCU_VERIFY"); sys.exit(1)
if not c:
    print("CPU log missing — HIP-only. Compare later with analyze_2d_scaling.py.")
    print("PASS_HIP_ONLY")
    sys.exit(0)
print("CPU", {k: c.get(k) for k in ("mass1", "x_tip", "sum_phi", "sum_c")})
# HIP vs CPU: looser than OpenMP vs MPI (device reductions / order)
tol = {"mass1": 1e-6, "x_tip": 1e-9, "sum_phi": 1e-5, "sum_c": 1e-5}
ok = True
for k, t in tol.items():
    da = abs(float(h[k]) - float(c[k]))
    rel = da / max(abs(float(c[k])), 1.0)
    print(f"  {k}: hip={h[k]} cpu={c[k]} rel_diff={rel:.3e} (tol {t})")
    if rel > t and da > t:
        ok = False
print("PASS" if ok else "FAIL: HIP vs CPU disagree (try OPENPFC_HIP_FORCE_PACKED_HALO=1)")
sys.exit(0 if ok else 2)
PY
