<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Held-out HeFFTe selector (issue #107) — blocked scaffold

This is software scaffolding only. Do **not** start held-out jobs. Do
**not** fit a selector. Do **not** change production reshape policy.

Issue: #107.
Depends on #106.

## Allowed now

- Feature schema (pre-execution quantities from the #106 descriptor
  replica, plus an empty topology slot).
- Regret `T(selected)/T(best) - 1`.
- Train vs candidate-held-out split records.
- Unit tests of regret and hit rate.

## Not allowed yet

- Timing any held-out family.
- Choosing held-out cases after seeing their protocol ranking.
- Fitting coefficients on #107 cases.
- Adding topology features until #106 shows they are observable and
  reproducible before execution.

## Candidate held-out families

`384`, `640`, `896` local-edge. These are a list, not a freeze.
Freeze only after #106 reports and before any of those families is
timed.
