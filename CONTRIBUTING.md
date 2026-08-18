<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to OpenPFC

Every commit is a small, self-contained, reviewable, revertable unit that
tells one clear story. That is the bar for humans and agents alike.

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

Use [`scripts/build.sh`](scripts/build.sh) — do not invoke `cmake` / `ctest` by
hand for routine work. The script loads the correct modules and toolchain.
See [`AGENTS.md`](AGENTS.md) for Tohtori vs LUMI. Manual toolchain notes live
in [`INSTALL.md`](INSTALL.md).

```bash
./scripts/build.sh --help
./scripts/build.sh                              # auto-detects Tohtori vs LUMI
./scripts/build.sh --machine=tohtori --with-cuda
./scripts/build.sh --machine=lumi --with-rocm   # HIP; compile+test on dev-g
```

Verify locally before opening or merging a PR. Run the **full** suite, not
just tests for files you touched: a change to shared state (a field type, a
halo exchanger, a CMake option) can break a test that never appears in your
diff. "My new tests pass" and "nothing broke" are different claims; only the
full suite proves the second. If CI is broken or unavailable, that raises
the bar — run the suite yourself before merging.

If a PR adds a new entry point (binary, script, CMake target), the suite
must invoke that entry point. Unit tests of helpers are not enough.

## Commit scope

One commit = one logical change. Nothing unrelated riding along.

- Prefer **1–5 files** per commit. Exceed that only when the extra files
  are inseparable parts of the same entity (a function and the one test
  that proves it; a header and its `.cpp`).
- Default to **one file per commit** when the files are not tightly coupled.
- Do **not** mix unrelated implementation, tests, docs, cleanup, generated
  files, or refactors in one commit.
- Implementation, tests, and docs *may* share a commit when they describe
  one capability and are more useful reviewed together.
- If a change could be reviewed, reverted, or explained independently of
  the rest, split it.
- If your change breaks or updates an existing test elsewhere, fold that
  fix into the **same** commit that caused it. Never leave a "fix tests"
  commit that hides which change broke the suite.
- Unrelated hunks in one file belong in different commits (`git add -p`).
- Cleanup or formatting that is not required by the functional change is
  its own commit — or, better, left out of the PR.

**Split** when the description would need "and also"; when file groups
touch different subsystems (e.g. halo vs FFT vs frontend); when tests
cover a different feature than the implementation; or when docs describe
behavior this commit did not change.

**Combine** commits that were only split by authorship mechanics, not
logical independence:

- a `test:` commit whose test exercises several preceding commits together
  means those commits were never independently revertable — squash them;
- a `docs:` commit that describes the combined effect of several preceding
  commits, rather than one commit's own change, is the same signal;
- a chain of `fix:` commits that correct mistakes introduced earlier in
  the *same* PR (not a regression in already-merged code) — squash each
  fix into the commit whose mistake it corrects so history shows the
  feature working on the first attempt, not the debugging trail.

When unsure, **split further**. Small and reviewable beats big and tidy.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) subject plus a
short body. Every commit, human or agent:

```
type(scope): short imperative subject

One to three sentences on what changed and why, not a restatement of the
subject and not a file list.

- optional one-line bullet with a decision, tradeoff, or coverage note
- another bullet only if it adds something the summary does not say
```

Rules:

- Subject: imperative mood (`add`, `fix`, `reject` — not `added`/`adding`),
  **max 72 characters**, no trailing period.
- One blank line between subject and body.
- Body lines wrap at **80 characters**.
- Body starts with 1–3 sentences of what and why.
- Extra detail goes in bullets **after** the summary, only if needed.
- The body is never just a list of edited files.
- **Each bullet is one line.** If it does not fit, shorten the wording —
  do not wrap the bullet onto a continuation line.

**Type** (what kind of change):

| Type | Use for |
|------|---------|
| `feat` | new user-facing capability |
| `fix` | bug fix |
| `refactor` | internal change with no behavior change |
| `perf` | performance improvement |
| `test` | test-only changes |
| `docs` | documentation-only changes |
| `build` | CMake, HeFFTe, `scripts/build.sh`, dependencies |
| `ci` | GitHub Actions / workflow changes |
| `chore` | maintenance that does not fit the types above |
| `revert` | deliberate reversal of an earlier commit |

**Scope** (where it lives). Type and scope are independent: write
`test(halo): …`, not a collapsed `test:` or `halo:` when both help a
skimmer. Scope is optional when the type alone is unambiguous. Prefer
these OpenPFC names and reuse them consistently:

`kernel`, `data`, `field`, `domain`, `decomp`, `halo`, `fft`, `gpu`,
`simulation`, `stepper`, `frontend`, `io`, `apps`, `tungsten`,
`aluminum`, `kobayashi`, `cmake`, `ci`, `docs`, `tests`

Example:

```
fix(halo): reject a zero direction in padded slab geometry

padded_send_slab treated {0,0,0} as a valid neighbour and produced an
empty slab instead of failing closed. Callers then posted a zero-byte
MPI message and silently skipped the exchange.

- throw std::invalid_argument from validate_padded_slab_args
- cover the zero-direction case in test_halo_geometry
```

A subject-only commit is fine for a genuine one-line typo fix. That is
the exception, not the default.

## Cleaning up a branch before review

If the branch already has mass commits, `wip`/`fix`/`more changes`
subjects, mixed unrelated files, or throwaway checkpoints, **rewrite
it before asking for review**. Reviewers should see the final history,
not how you got there.

```bash
git rebase -i origin/master
```

- Reorder, split (`edit` + `git reset HEAD^` + `git add -p`), reword,
  and `squash`/`fixup` into the units above.
- Drop throwaway checkpoint commits rather than leaving them for a
  reviewer to decode.
- After rewriting, confirm only boundaries moved:
  `diff <(git diff old-tip...origin/master) <(git diff HEAD...origin/master)`
- Force-push **only that feature branch** with
  `git push --force-with-lease`. Never force-push `master` / `main`.

If `git rebase -i` is unavailable, assemble the same history forward:
`git checkout -b rewritten origin/master`, then
`git cherry-pick -n <commit>`, `git reset -- <paths for later>`,
commit, repeat, folding fixes into the commit that should have had them.

To collapse an over-split branch that was never independently
revertable: `git reset --soft origin/master` and one commit that
follows the message rules.

## Pull requests and merging

- Branch from the latest `master`. A PR can and often should contain
  multiple commits, as long as each is a small, coherent, independently
  reviewable/revertable unit.
- Keep unrelated changes in separate PRs.
- Describe impact, behavior changes, and what you actually ran
  (`./scripts/build.sh …` on which machine). If you could not run the
  suite (no GPU allocation, no cluster access), say so in the PR
  description — an unverified PR should read as unverified.
- Read the **full** diff yourself, not just the hunks you expect. A
  rewrite can silently drop trailing content (a license block, a doc
  link) that no test exercises.
- If the PR removes or replaces a path (a fallback, a deprecated flag),
  grep the repo for leftovers — docs, error messages, CI scripts — and
  fix them in the same PR.
- Only mark a PR ready once every commit is reviewable on its own.
- **Merge by rebase, not squash**, so the commit boundaries you settled
  on during review are what lands on `master`. That is why messages and
  boundaries have to be right *before* merge.
- If review finds a defect in a specific commit, amend or rewrite that
  commit (`git rebase -i`). Do not append a correction on top: a commit
  that is only correct after a later "fix" was never self-contained, and
  `git bisect` lands on the broken intermediate.
- Never push directly to `master` (or any branch others commit to). The
  one narrow exception for Limina gate agents is in [`AGENTS.md`](AGENTS.md).

## CI (GitHub Actions)

Pull requests run workflows under [`.github/workflows/`](.github/workflows):
**`ci.yml`** (main build/test matrix on Ubuntu 24.04), **`docs.yml`**
(markdown link check via `scripts/check_doc_links.py`, Doxygen when
enabled), **`coverage.yml`**, **`asan.yml`**, **`clang-tidy.yml`**.
Doc-only edits under `docs/**` still trigger the Documentation
workflow's link job — run `python3 scripts/check_doc_links.py` locally
before pushing.

Passing checks make a PR mergeable, not right. Fix only lint/test
failures **this PR introduced**; fold those fixes into the commit that
caused them. Leave pre-existing baseline debt alone.

## Changelog

User-visible and developer-facing changes go in
[`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]` when they affect
behavior, CMake options, or config keys. Put that note in the same
commit as the change when the changelog line is about that one change;
do not bundle an unrelated Unreleased dump.

## Questions

Use [GitHub Issues](https://github.com/VTT-ProperTune/OpenPFC/issues)
for bugs and feature discussion.
