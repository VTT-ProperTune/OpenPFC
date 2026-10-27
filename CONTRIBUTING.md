<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to OpenPFC

> Process rules come from the latest approved
> [OodiHandbook](https://github.com/ahojukka5/OodiHandbook) tag (AUTH-007).
> Start with its `guides/local-development.qmd` and `guides/contributing.qmd`.
> This file adds only this repository's facts (ENG-001, ENG-028).

Commits, history and review follow ENG-005, ENG-006, ENG-008 and ENG-018
(rebase-merge, never squash); branches follow ENG-026. Build, test and CI
details are in [`AGENTS.md`](AGENTS.md); guides start at
[`docs/README.md`](docs/README.md). Bugs and features go to
[GitHub Issues](https://github.com/ahojukka5/OpenPFC/issues).

**Commit scopes.** Use `type(scope): subject` with one of:
`kernel`, `data`, `field`, `domain`, `decomp`, `halo`, `fft`, `gpu`,
`simulation`, `stepper`, `frontend`, `io`, `apps`, `tungsten`,
`aluminum`, `kobayashi`, `cmake`, `ci`, `docs`, `tests`.
`build` covers CMake, HeFFTe, `scripts/build.sh` and dependencies.

**Hook.** `git config core.hooksPath .githooks` enables the pre-commit
check: clang-format 20 on staged C/C++ files, as in CI.

**Never commit:** build trees (`build/`, `builds/`), run output
(`results/`, `out/`, `*.log`, `*.out`), HeFFTe sources or tarballs, Julia
manifests, local preferences, or new `*.sbatch` campaign launchers. A user-
or developer-visible change adds a `CHANGELOG.md` `[Unreleased]` line in the
same commit.
