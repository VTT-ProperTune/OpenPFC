<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to OpenPFC

> Exception to ENG-028 (owner approval,
> [#767](https://github.com/ahojukka5/OpenPFC/issues/767)): OpenPFC is public,
> so this file is self-contained for contributors who cannot open the private
> OodiHandbook. Agents also follow [`AGENTS.md`](AGENTS.md) and the Handbook.

Thank you for helping. Bugs and feature ideas go to
[GitHub Issues](https://github.com/ahojukka5/OpenPFC/issues); start a change
from an issue so its purpose is agreed first.

**Workflow.** Branch from `master` as `<type>/<issue-number>-<slug>`, for
example `fix/412-halo-corner`. Keep one coherent change per PR. The maintainer
reviews every PR and integrates it by rebase-merge, so `master` stays a
straight line; please do not merge or squash yourself.

**Commits.** Subject `type(scope): imperative description`, at most 72
characters, then a blank line and a body wrapped at 80 columns that says why.
One logical change per commit, with its tests. No `Co-authored-by` or tool
attribution lines. Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`,
`build`, `ci`, `style`, `chore`. Scopes: `kernel`, `data`, `field`,
`domain`, `decomp`, `halo`, `fft`, `gpu`, `simulation`, `stepper`,
`frontend`, `io`, `apps`, `tungsten`, `aluminum`, `kobayashi`, `cmake`, `ci`,
`docs`, `tests`; `build` covers CMake, HeFFTe, `scripts/build.sh` and
dependencies.

**Tests.** Run the smallest test that exercises your change, and the full
`ctest` for wide changes (field type, halo, decomposition, FFT or a CMake
option). In the PR, name the commands you ran and what you could not run.
Build and test commands are in [`AGENTS.md`](AGENTS.md); guides start at
[`docs/README.md`](docs/README.md).

**Hook.** `git config core.hooksPath .githooks` enables the pre-commit check:
clang-format 20 on staged C/C++ files, as in CI.

**Never commit:** build trees (`build/`, `builds/`), run output (`results/`,
`out/`, `*.log`, `*.out`), HeFFTe sources or tarballs, Julia manifests, local
preferences, or new `*.sbatch` campaign launchers. A user- or developer-visible
change adds a `CHANGELOG.md` `[Unreleased]` line in the same commit.
