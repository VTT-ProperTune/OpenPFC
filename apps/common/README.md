<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Shared FD app helpers (`apps/common`)

Header-only INTERFACE library `openpfc_apps_common`. Per-app `RunConfig` and
usage strings stay in the app; this tree holds the duplicated mechanics:

| Header | Role |
|--------|------|
| `openpfc_apps/cli.hpp` | `parse_or_print_usage`, `--flag` tokens, even FD-order check |
| `openpfc_apps/mpi_report.hpp` | MPI SUM/MAX reduce, timing lines, step-timing report |
| `openpfc_apps/gather.hpp` | pack owned z=0, rank-0 XY gather, ordered field stats |
