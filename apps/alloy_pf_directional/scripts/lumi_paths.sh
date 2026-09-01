# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Path and account layout for Al-Cu FTA on LUMI-C (CPU OpenMP).
# Source this file on LUMI, or from the laptop scripts that rsync/ssh.
#
# Layout (same idea as Puhti/Mahti):
#   /projappl/<project>/<user>   installs, source, CMake build trees
#   /scratch/<project>/<user>    job I/O (fields, VTK, logs)
# Do not use scripts/build.sh on LUMI for this app: that defaults to HIP/LUMI-G.

LUMI_PROJECT="${LUMI_PROJECT:-project_462001519}"
LUMI_USER="${LUMI_USER:-tpinomaa}"
LUMI_HOST="${LUMI_HOST:-lumi.csc.fi}"

LUMI_PROJAPPL="${LUMI_PROJAPPL:-/projappl/${LUMI_PROJECT}/${LUMI_USER}}"
LUMI_SCRATCH="${LUMI_SCRATCH:-/scratch/${LUMI_PROJECT}/${LUMI_USER}}"

LUMI_SRC="${LUMI_SRC:-${LUMI_PROJAPPL}/src/OpenPFC}"
LUMI_BUILD="${LUMI_BUILD:-${LUMI_PROJAPPL}/build/openpfc-cpu}"
LUMI_HEFFTE_PREFIX="${LUMI_HEFFTE_PREFIX:-${LUMI_PROJAPPL}/opt/heffte/2.4.1-cpu}"
LUMI_HEFFTE_SRC="${LUMI_HEFFTE_SRC:-${LUMI_PROJAPPL}/src/heffte-2.4.1}"
LUMI_HEFFTE_BUILD="${LUMI_HEFFTE_BUILD:-${LUMI_PROJAPPL}/build/heffte-2.4.1-cpu}"
LUMI_PREFIX="${LUMI_PREFIX:-${LUMI_PROJAPPL}/opt/openpfc-cpu}"
LUMI_BIN="${LUMI_BIN:-${LUMI_PREFIX}/bin/alloy_pf_directional_openmp}"
LUMI_BIN_MPI="${LUMI_BIN_MPI:-${LUMI_PREFIX}/bin/alloy_pf_directional_mpi}"
LUMI_PREFIX_HIP="${LUMI_PREFIX_HIP:-${LUMI_PROJAPPL}/opt/openpfc-hip}"
LUMI_BUILD_HIP="${LUMI_BUILD_HIP:-${LUMI_PROJAPPL}/build/openpfc-hip}"
LUMI_BIN_HIP="${LUMI_BIN_HIP:-${LUMI_PREFIX_HIP}/bin/alloy_pf_directional_hip}"
# Historical scratch name: in-flight jobs still write here (do not rename while
# 21665355 / 21665357 are running). New binaries install as alloy_pf_directional_*.
LUMI_RUNS="${LUMI_RUNS:-${LUMI_SCRATCH}/alcu_fta}"
LUMI_LOGS="${LUMI_LOGS:-${LUMI_RUNS}/logs}"
LUMI_SCALE2D="${LUMI_SCALE2D:-${LUMI_RUNS}/scale_2d}"

LUMI_STACK="${LUMI_STACK:-LUMI/25.09}"
HEFFTE_VERSION="${HEFFTE_VERSION:-2.4.1}"

# Shared-node OpenMP jobs: never --exclusive, never a full 128-core node
# unless a production case actually needs it (ask first).
LUMI_SMOKE_CPUS="${LUMI_SMOKE_CPUS:-8}"
LUMI_BUILD_CPUS="${LUMI_BUILD_CPUS:-8}"
LUMI_DS_CPUS="${LUMI_DS_CPUS:-16}"
LUMI_MEM_PER_CPU="${LUMI_MEM_PER_CPU:-1750}"

# Compute nodes do not see libfabric via the default linker path. After
# `module load ... lumi-CrayPath`, merge Cray PE libs and the OFI library.
lumi_cpu_runtime_env() {
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  local fab
  for fab in /opt/cray/libfabric/*/lib64; do
    if [[ -e "${fab}/libfabric.so.1" ]]; then
      export LD_LIBRARY_PATH="${fab}:${LD_LIBRARY_PATH}"
      break
    fi
  done
}

# LUMI-G: GPU-aware MPI + ROCm runtime search path.
lumi_gpu_runtime_env() {
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
  local fab
  for fab in /opt/cray/libfabric/*/lib64; do
    if [[ -e "${fab}/libfabric.so.1" ]]; then
      export LD_LIBRARY_PATH="${fab}:${LD_LIBRARY_PATH}"
      break
    fi
  done
}

# Canonical 2D DS family (Nz=1, n_dim=2). Laptop box is 1280×160 at W0=5 nm.
# GRID=1280x160|2560x320|5120x640|3600x1280|7200x2560|10240x1280|20480x2560|w0_10nm
alcu_2d_apply_grid() {
  export OPENPFC_ALCU_NDIM=2
  export OPENPFC_ALCU_NZ=1
  export OPENPFC_ALCU_LZ=0
  export OPENPFC_ALCU_DXW="${OPENPFC_ALCU_DXW:-1.0}"
  export OPENPFC_ALCU_NGRANS=1
  export OPENPFC_ALCU_NOISE=0
  export OPENPFC_ALCU_STOP_RIGHT=0
  export OPENPFC_ALCU_PERIODIC_Y=1
  export OPENPFC_ALCU_SKIP_PNG=1
  export OPENPFC_ALCU_SKIP_VTK=1
  export OPENPFC_ALCU_QUIET=1
  export OPENPFC_ALCU_ISO="${OPENPFC_ALCU_ISO:-1}"
  case "${1:-1280x160}" in
    1280x160|laptop)
      export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0:-5e-9}"
      export OPENPFC_ALCU_LX=6.40e-6
      export OPENPFC_ALCU_LY=0.80e-6
      ;;
    2560x320)
      export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0:-5e-9}"
      export OPENPFC_ALCU_LX=12.80e-6
      export OPENPFC_ALCU_LY=1.60e-6
      ;;
    5120x640)
      export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0:-5e-9}"
      export OPENPFC_ALCU_LX=25.60e-6
      export OPENPFC_ALCU_LY=3.20e-6
      ;;
    3600x1280)
      # Lx=18 μm, Ly=6.40 μm at W0=5 nm (queued Lx18 production box)
      export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0:-5e-9}"
      export OPENPFC_ALCU_LX=18.0e-6
      export OPENPFC_ALCU_LY=6.40e-6
      ;;
    7200x2560)
      export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0:-2.5e-9}"
      export OPENPFC_ALCU_LX=18.0e-6
      export OPENPFC_ALCU_LY=6.40e-6
      ;;
    10240x1280)
      export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0:-5e-9}"
      export OPENPFC_ALCU_LX=51.20e-6
      export OPENPFC_ALCU_LY=6.40e-6
      ;;
    20480x2560)
      export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0:-5e-9}"
      export OPENPFC_ALCU_LX=102.40e-6
      export OPENPFC_ALCU_LY=12.80e-6
      ;;
    w0_10nm)
      export OPENPFC_ALCU_W0=10e-9
      export OPENPFC_ALCU_LX=6.40e-6
      export OPENPFC_ALCU_LY=0.80e-6
      ;;
    *)
      echo "unknown GRID='$1' (expected 1280x160|2560x320|5120x640|3600x1280|7200x2560|10240x1280|20480x2560|w0_10nm)" >&2
      return 2
      ;;
  esac
  export OPENPFC_ALCU_W0="${OPENPFC_ALCU_W0}"
  export OPENPFC_ALCU_LY="${OPENPFC_ALCU_LY}"
  export OPENPFC_ALCU_LX="${OPENPFC_ALCU_LX}"
}

# Extra names used by some campaign scripts
LUMI_USER="${LUMI_USER:-${LUMI_USER}}"

