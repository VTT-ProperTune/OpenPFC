#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Submit tungsten_performance-style jobs for 1, 2, and 4 nodes (CPU + GPU).
# Run from anywhere; logs and run dirs under scratch (see *.sbatch).
#
# Usage: ./submit_tungsten_performance.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for n in 1 2 4; do
  echo "sbatch CPU ${n} node(s) (1024³)..."
  sbatch --nodes="${n}" --job-name="tperf-cpu-${n}n-1024" "${SCRIPT_DIR}/tungsten_cpu.sbatch"
  echo "sbatch GPU ${n} node(s) (1024³)..."
  sbatch --nodes="${n}" --job-name="tperf-gpu-${n}n-1024" "${SCRIPT_DIR}/tungsten_gpu.sbatch"
done

echo "Submitted 6 jobs. Logs: /scratch/project_462001245/juaho/tungsten_perf_jobs/logs/"
