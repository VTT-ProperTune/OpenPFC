#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Step 0 CPU: OpenMP vs MPI np=1 on the 2D DS brick (Nz=1, n_dim=2).
# Default is the laptop 1280×160 box (W0=5 nm). PNG/VTK off.
#
#   BUILD=builds/macos-cpu-release STEPS=800 ./apps/alloy_pf_directional/scripts/check_nz1_vs_2d.sh
#   GRID=w0_10nm STEPS=1000 ./apps/alloy_pf_directional/scripts/check_nz1_vs_2d.sh
#   GRID=smoke STEPS=40 ./apps/alloy_pf_directional/scripts/check_nz1_vs_2d.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=lumi_paths.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lumi_paths.sh"
BUILD="${BUILD:-${OPENPFC_BUILD_DIR:-${ROOT}/builds/macos-cpu-release}}"
OMP="${OPENMP_BIN:-${BUILD}/apps/alloy_pf_directional/alloy_pf_directional_openmp}"
MPI="${MPI_BIN:-${BUILD}/apps/alloy_pf_directional/alloy_pf_directional_mpi}"
OUT="${OUT:-${ROOT}/results/alloy_pf_directional_nz1_check}"
GRID="${GRID:-1280x160}"
STEPS="${STEPS:-800}"
NTHREADS="${NTHREADS:-${OMP_NUM_THREADS:-8}}"
mkdir -p "${OUT}/openmp" "${OUT}/mpi"

if [[ "${GRID}" == "smoke" ]]; then
  export OPENPFC_ALCU_NDIM=2 OPENPFC_ALCU_NZ=1 OPENPFC_ALCU_LZ=0
  export OPENPFC_ALCU_LX=4.00e-7 OPENPFC_ALCU_LY=2.40e-7
  export OPENPFC_ALCU_NGRANS=1 OPENPFC_ALCU_NOISE=0
  export OPENPFC_ALCU_STOP_RIGHT=0 OPENPFC_ALCU_PERIODIC_Y=1
  export OPENPFC_ALCU_SKIP_PNG=1 OPENPFC_ALCU_SKIP_VTK=1 OPENPFC_ALCU_QUIET=1
else
  alcu_2d_apply_grid "${GRID}"
fi
export OPENPFC_ALCU_MAX_STEPS="${STEPS}"
export OPENPFC_ALCU_TEND="${OPENPFC_ALCU_TEND:-1.0e-6}"
export OPENPFC_ALCU_WARMUP=0
unset OPENPFC_ALCU_TIMED_STEPS || true
export OMP_NUM_THREADS="${NTHREADS}"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"

if [[ ! -x "${OMP}" ]]; then
  echo "missing ${OMP} — build alloy_pf_directional_openmp" >&2
  exit 1
fi

echo "== OpenMP Nz=1 n_dim=2 GRID=${GRID} STEPS=${STEPS} threads=${NTHREADS} =="
set +e
"${OMP}" ds "${OUT}/openmp" "${NTHREADS}" | tee "${OUT}/openmp.log"
omp_rc=${PIPESTATUS[0]}
echo "== MPI np=1 Nz=1 n_dim=2 =="
if [[ -x "${MPI}" ]]; then
  if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 "${MPI}" ds "${OUT}/mpi" | tee "${OUT}/mpi.log"
    mpi_rc=${PIPESTATUS[0]}
  else
    # np=1: prefer mpirun, but Open MPI 5 on a Mac without usable ifaddrs
    # can refuse the launcher. The binary is a valid MPI singleton.
    set +e
    if command -v mpirun >/dev/null 2>&1; then
      mpirun --oversubscribe -n 1 \
        --mca btl self,vader \
        --mca pmix_ptl_tool_if_include lo0 \
        "${MPI}" ds "${OUT}/mpi" > "${OUT}/mpi.log" 2> "${OUT}/mpi.launch.err"
      mpi_rc=$?
      cat "${OUT}/mpi.log"
      if [[ "${mpi_rc}" -ne 0 ]]; then
        echo "mpirun failed (rc=${mpi_rc}); retry as MPI singleton" >&2
        cat "${OUT}/mpi.launch.err" >&2 || true
        "${MPI}" ds "${OUT}/mpi" | tee "${OUT}/mpi.log"
        mpi_rc=${PIPESTATUS[0]}
      fi
    else
      "${MPI}" ds "${OUT}/mpi" | tee "${OUT}/mpi.log"
      mpi_rc=${PIPESTATUS[0]}
    fi
    set -e
  fi
else
  echo "alloy_pf_directional_mpi not built; skip MPI compare" | tee "${OUT}/mpi.log"
  mpi_rc=0
fi
set -e
if [[ "${omp_rc}" -ne 0 ]]; then
  echo "OpenMP run failed (rc=${omp_rc})" >&2
  exit 1
fi
if [[ "${mpi_rc}" -ne 0 ]]; then
  echo "MPI run failed (rc=${mpi_rc})" >&2
  exit 1
fi

python3 - "${OUT}" <<'PY'
import pathlib, re, sys
out = pathlib.Path(sys.argv[1])

def grab_line(p, prefix):
    t = p.read_text() if p.exists() else ""
    m = re.search(rf"^{prefix} (.*)$", t, re.M)
    if not m:
        return {}
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d

def grab_verify(p):
    return grab_line(p, "ALCU_VERIFY")

a, b = grab_verify(out / "openmp.log"), grab_verify(out / "mpi.log")
pa, pb = grab_line(out / "openmp.log", "ALCU_PERF"), grab_line(out / "mpi.log", "ALCU_PERF")
if not a:
    print("FAIL: no OpenMP ALCU_VERIFY"); sys.exit(1)
keys = ("mass1", "x_tip", "sum_phi", "sum_c")
print("OpenMP", {k: a.get(k) for k in keys})
if pa:
    print("OpenMP ALCU_PERF", {k: pa.get(k) for k in
          ("time_per_step_s", "solute_pct", "ghost_pct", "halo_pct", "nthreads")})
if not b:
    print("MPI binary missing — OpenMP-only. Re-run after building alloy_pf_directional_mpi.")
    sys.exit(0)
print("MPI   ", {k: b.get(k) for k in keys})
if pb:
    print("MPI ALCU_PERF", {k: pb.get(k) for k in
          ("time_per_step_s", "halo_pct", "kernel_pct", "nproc")})
tol_mass, tol_tip, tol_sum = 1e-9, 1e-10, 1e-8

def f(d, k):
    return float(d[k])
ok = True
for k, tol in (("mass1", tol_mass), ("x_tip", tol_tip), ("sum_phi", tol_sum), ("sum_c", tol_sum)):
    da, db = abs(f(a, k) - f(b, k)), max(abs(f(a, k)), 1.0)
    rel = da / db
    print(f"  {k}: openmp={a[k]} mpi={b[k]} rel_diff={rel:.3e} (tol {tol})")
    if rel > tol and da > tol:
        ok = False
print("PASS" if ok else "FAIL: Nz=1 OpenMP vs MPI disagree beyond tolerance")
sys.exit(0 if ok else 2)
PY
