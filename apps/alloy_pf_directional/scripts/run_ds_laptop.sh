#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
exec "$(cd "$(dirname "$0")" && pwd)/run_ds_convergence.sh" "$@"
