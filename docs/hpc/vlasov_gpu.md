<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# The GPU path of `vlasov_maxwell` (LUMI-G, HIP)

What was ported to the device, what was deliberately left on the host, the
measurement that decided each, the CPU/GPU parity tolerances and where they
come from, and the limitations. Every number below is from a `standard-g`
compute-node run and carries its Slurm job id.

This page belongs to issue #84's "CPU / MPI / GPU path" requirement and its
acceptance criterion *"CPU/GPU parity on a small case, to a stated
tolerance"*. The application itself is documented by its own headers; start
at `apps/vlasov_maxwell/include/vlasov_maxwell/parameters.hpp`.

## PLACEHOLDER
