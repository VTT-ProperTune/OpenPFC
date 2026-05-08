# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# pick_nsys_artifact_base.bash — source from Kobayashi Nsight Systems Slurm drivers.
#
# Nsight stores reports as sqlite-backed `.nsys-rep` files; writing them on Lustre
# can be extremely slow. If `/WRKDIR/$USER/openpfc` exists (fast node NVMe on
# some clusters), use it as the parent directory for large nsys artifacts;
# otherwise fall back to `SLURM_SUBMIT_DIR`.
#
# On success sets:
#   KOBAYASHI_NSYS_ARTIFACT_BASE   — directory (scripts append job-tagged subdirs)
#   KOBAYASHI_NSYS_ARTIFACT_BASE_REASON — short log string
#
# Optional: set `KOBAYASHI_NSYS_ARTIFACT_BASE` before sourcing to force a path.

: "${SLURM_SUBMIT_DIR:=.}"

if [[ -n "${KOBAYASHI_NSYS_ARTIFACT_BASE:-}" ]]; then
  KOBAYASHI_NSYS_ARTIFACT_BASE_REASON="preset KOBAYASHI_NSYS_ARTIFACT_BASE"
elif [[ -d "/WRKDIR/${USER}/openpfc" ]]; then
  KOBAYASHI_NSYS_ARTIFACT_BASE="/WRKDIR/${USER}/openpfc"
  KOBAYASHI_NSYS_ARTIFACT_BASE_REASON="WRKDIR NVMe (/WRKDIR/${USER}/openpfc)"
else
  KOBAYASHI_NSYS_ARTIFACT_BASE="${SLURM_SUBMIT_DIR}"
  KOBAYASHI_NSYS_ARTIFACT_BASE_REASON="SLURM_SUBMIT_DIR (no /WRKDIR/${USER}/openpfc)"
fi
