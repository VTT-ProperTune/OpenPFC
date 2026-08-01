<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to OpenPFC

## Documentation

- [`docs/README.md`](docs/README.md) — index of all guides (architecture, HPC, tutorials). **API reference (HTML)** vs prose: see the opening table there.
- [`docs/learning_paths.md`](docs/learning_paths.md) — run / extend / integrate tracks.
- [`docs/tutorials/README.md`](docs/tutorials/README.md) — step-by-step tutorials (VTK, HeFFTe, spectral examples, GPU, …).
- [`docs/personas.md`](docs/development/personas.md) — short entry points by role (cluster runner, model developer, integrator).
- [`docs/tutorials/add_catch2_test.md`](docs/tutorials/add_catch2_test.md) — minimal Catch2 / `ctest` pattern.
- [`docs/showcase.md`](docs/user_guide/showcase.md) — figures mapped to apps and examples.
- [`docs/testing.md`](docs/development/testing.md) — `ctest`, `openpfc-tests`, MPI test CMake options.
- [`docs/contributing-docs.md`](docs/development/contributing-docs.md) — link checks, SPDX headers, where to add cross-links in the doc index.
- Run from the repo root: `python3 scripts/check_doc_links.py`
- Style for code and headers: [`docs/styleguide.md`](docs/development/styleguide.md)
- Extending the library (models, `App`, validation): [`docs/extending_openpfc/README.md`](docs/extending_openpfc/README.md), [`docs/class_tour.md`](docs/reference/class_tour.md), [`docs/tutorials/custom_app_minimal.md`](docs/tutorials/custom_app_minimal.md), [`docs/parameter_validation.md`](docs/user_guide/parameter_validation.md)

## Build and test

Follow [`INSTALL.md`](INSTALL.md) for MPI, HeFFTe, and CMake. Run tests with your configured build (e.g. `ctest` or the project’s test targets) after `OpenPFC_BUILD_TESTS=ON`.

Verify locally before opening or merging a PR — run the build and full test suite yourself rather than treating CI as the first check. Run the **full** suite, not just tests for files you touched: a change to shared/global state (a module-level function, a class method, a bootstrap side effect) can silently break an existing test that never shows up in your diff. If CI is broken, misconfigured, or unavailable, that raises the bar rather than lowering it — run the full suite locally yourself before merging.

## CI (GitHub Actions)

Pull requests run workflows under [`.github/workflows/`](.github/workflows): **`ci.yml`** (main build/test matrix on Ubuntu 24.04), **`docs.yml`** (markdown link check via `scripts/check_doc_links.py`, Doxygen when enabled), **`coverage.yml`**, **`asan.yml`**, **`clang-tidy.yml`**. Doc-only edits under `docs/**` still trigger the **Documentation** workflow’s link job—run `python3 scripts/check_doc_links.py` locally before pushing.

## Commit scope

One commit = one logical change, nothing unrelated riding along.

- Prefer touching 1-5 files per commit. Only exceed that when the extra files are inseparable parts of the same logical entity (a function and the one test that proves it).
- Implementation, tests, and docs *may* share a commit when they describe one capability and are genuinely more useful reviewed together — but split them when the description would need "and also," when different file groups touch different subsystems, or when the tests/docs actually describe a different feature than the rest of the commit.
- If your change breaks or requires updating an existing test elsewhere, fold that fix into the same commit that caused it — don't leave it for a later "fix tests" commit, which hides which change actually caused the breakage.
- Conversely, squash commits that were only ever split by authorship mechanics, not logical independence: a chain of `fix:` commits correcting mistakes introduced earlier in the *same* PR (not a regression in already-merged code) was never independently revertable — squash each fix into the commit whose mistake it corrects, so history shows the feature working correctly on the first attempt, not the debugging trail.
- Before opening or updating a PR, reorder/fixup/split with `git rebase -i origin/master` so each commit tells one clear story. Don't squash unrelated logical changes together "to save time."
- When unsure whether something should be one commit or several: split further. Small and reviewable beats big and tidy.

## Commit messages

Every commit (human or agent-authored) follows [Conventional Commits](https://www.conventionalcommits.org/) for the subject, plus a structured body:

- **Subject:** `type(scope): short description`, imperative mood, **max 72 characters**, no trailing period. Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `build`, `chore`.
- **Body:** a blank line after the subject, then **1-3 sentences** summarizing what changed and why (not a restatement of the subject).
- If more detail is needed, follow the summary with a **bullet-pointed list** of specifics (files/functions touched, notable tradeoffs, what was deliberately left out).
- **Wrap the body at 80 characters** per line.

Example:

```
fix(mpi): delete copy/move on MPI_Worker to prevent double MPI_Finalize

MPI_Worker relied on the default copyable/movable special member
functions, so two copies of an owning worker could both believe they
were responsible for calling MPI_Finalize() -- undefined behavior per
the MPI standard.

- Delete copy/move constructor and assignment, matching the pattern
  already used by BinaryWriter/BinaryReader in the same MPI layer.
- Add compile-time tests asserting MPI_Worker is neither copyable nor
  movable.
```

A one-line subject with no body is fine for genuinely trivial changes (e.g. a
single typo fix), but is the exception, not the default.

## Pull requests and merging

- Branch from the latest `master`. A PR can and often should contain multiple
  commits, as long as each is a small, coherent, independently reviewable/
  revertable unit (see "Commit scope" above).
- Read the full diff yourself, not just the hunks you expect, before asking
  for review — a partial rewrite can silently drop unrelated trailing content
  (a license section, a doc link) that no test exercises.
- If your PR removes or replaces a code path (a fallback, a deprecated flag),
  grep the whole repository for other places that still assert or depend on
  the old behavior — docs, error messages, CI scripts — and fix them in the
  same PR.
- **Merge by rebase, not squash**, so the commit boundaries you settled on
  during review are preserved in `master`'s history. This is exactly why
  commit boundaries and messages have to be right *before* merge, not cleaned
  up after.
- If review finds a defect in a specific commit before merge, fix it by
  amending/rewriting that commit (`git rebase -i`), not by appending a
  correction on top — a commit that's only correct once a later commit "fixes"
  it was never self-contained, and `git bisect` lands on the broken
  intermediate state.
- Never push directly to `master` (or any branch others are also committing
  to). The one narrow exception already in use by Limina gate agents is
  documented in [`AGENTS.md`](AGENTS.md).
- After rewriting history, diff the full range against the original tip to
  confirm only commit boundaries moved, not content:
  `diff <(git diff old-tip...master) <(git diff new-tip...master)`.

## Changelog

User-visible and developer-facing changes are recorded in [`CHANGELOG.md`](CHANGELOG.md). Add a note under `[Unreleased]` when your change affects behavior, CMake options, or config file keys.

## Questions

Use [GitHub Issues](https://github.com/VTT-ProperTune/OpenPFC/issues) for bugs and feature discussion.
