#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Quick Slurm snapshot for the AMDGPU partition (single-node pools often show ALLOCATED
# while another user's job holds all CPUs/GPUs — pending jobs stay Reason=Resources).
#
# Usage: apps/kobayashi/slurm/inspect_amdgpu_resources.sh [partition]
#        PARTITION=amdgpu apps/kobayashi/slurm/inspect_amdgpu_resources.sh

set -u

PART="${1:-${PARTITION:-amdgpu}}"

echo "=== scontrol show partition ${PART} ==="
scontrol show partition "${PART}" 2>&1 || true

echo ""
echo "=== Per-node state: sinfo -p ${PART} -N -l ==="
sinfo -p "${PART}" -N -l 2>&1 || true

echo ""
echo "=== Compact: NODELIST STATE CPUS MEMORY GRES ==="
sinfo -p "${PART}" -h -o '%N %T %C %m %G' 2>&1 || true

echo ""
echo "=== Running/pending jobs on partition ${PART} ==="
squeue -p "${PART}" -o '%.10i %.12u %.28j %.2t %.12M %.10l %.12b %R' 2>&1 || true

echo ""
echo "=== Your jobs (all partitions) ==="
squeue -u "${USER}" -o '%.10i %.10P %.28j %.2t %.12M %R' 2>&1 || true

echo ""
echo "=== Pending jobs: estimated start (if available) ==="
for j in $(squeue -u "${USER}" -h -o '%i' -p "${PART}" 2>/dev/null); do
  squeue -j "${j}" --start 2>/dev/null || true
done

echo ""
echo "Tip: if STATE is alloc/mixed but OverSubscribe=NO and all CPUs are Alloc, new jobs wait."
echo "Tip: compare AllocTRES vs CfgTRES: scontrol show node <NODELIST>"
