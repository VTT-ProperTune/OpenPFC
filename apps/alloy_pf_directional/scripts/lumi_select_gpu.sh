#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LUMI-G: pin this rank to one GCD (MI250X has 8 GCDs / node).
# srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 this-script <binary> ...
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:-0}"
exec "$@"
