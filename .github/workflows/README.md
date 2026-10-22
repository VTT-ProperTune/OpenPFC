<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# GitHub Actions Workflows

This directory contains CI/CD workflows for the OpenPFC project.

## Workflows

## Tiers

| Event | What runs | Typical time |
| --- | --- | --- |
| Pull request | Code quality; GCC 13 Release with the full unit and 2-rank MPI tests | ~30 min |
| Push to master | The pull-request jobs plus the small matrix: GCC 13 Debug with the 3/4/6-rank multigrid transfers and the benchmark compile, the Clang compile check, packaging, CUDA and HIP compile-only | ~45 min |
| Monthly (and on demand) | `compatibility.yml`, then coverage, clang-tidy and ASan on the following days | a few hours |

Every job runs in `ahojukka5/OpenPFC` only. VTT-ProperTune/OpenPFC mirrors
`master` and runs only the documentation and website workflow.

### `ci.yml` - Pull request and master tiers

Runs on: pull requests to master, pushes to master, `workflow_dispatch`.

Jobs: `code-quality` → `build-and-test` on every event; `debug-and-mpi`,
`clang-compile`, `packaging-smoke`, `cuda-compile` and `hip-compile` on
pushes only. `CI Status` fails when a job it needs failed; on a pull request
the push-only jobs may be skipped. A newer pull-request push cancels the
running check; master runs are not cancelled.

1. Code Quality: clang-format 20 (advisory), REUSE compliance, kernel
   layering, GPU memcpy single-source, app self-containment and the
   accelerator build policy (`scripts/check_gpu_ci_policy.py`).
2. Common build (`build-and-test`): GCC 13 Release with HeFFTe, examples
   and apps, full `ctest` including 2-rank MPI
   (`OpenPFC_MPI_TEST_MAX_WORLD_SIZE=2`). Release is the `scripts/build.sh`
   default and the configuration installed as modules.
3. `debug-and-mpi`: GCC 13 Debug `ctest`, the multigrid plan on 3, 4 and 6
   oversubscribed ranks, and the benchmark compile.
4. `clang-compile`: clang-18 builds the library and the test TU that
   instantiates the host `FFT_Impl` (issue #373); compile only.
5. `packaging-smoke`: install and build `find_package(OpenPFC)` consumers.
6. `cuda-compile` / `hip-compile`: compile-only in the official CUDA and ROCm
   images, without HeFFTe; required on every master push.

Host jobs set up through `.github/actions/setup-build` (apt packages and the
cached HeFFTe build).

### `compatibility.yml` - Monthly compatibility matrix

Runs on: the 1st of each month 02:00 UTC, and `workflow_dispatch`.

Full unit and 2-rank MPI tests on GCC 11 (Ubuntu 22.04, Debug and Release),
GCC 12 (22.04 and 24.04), GCC 14 (Debug and Release), Clang 18, MPICH, and
the build without HeFFTe. `fail-fast` is off so every cell reports. A red
cell is a finding; open an issue for it rather than editing the matrix.

---

### `clang-tidy.yml` - Static analysis (non-blocking)

Runs on: the 3rd of each month 02:00 UTC, and `workflow_dispatch`. Not on pull requests or master.

Purpose: Run `scripts/run-clang-tidy.sh` with HeFFTe + `compile_commands.json`. A full-tree pass is ~45–60 min on GitHub runners and is `continue-on-error` until the WarningsAsErrors backlog is clear. Do not add this job as a required check.

Typical Duration: ~45–60 minutes.

---

### `docs.yml` - Documentation and website

Runs on:
- Push to master
- PRs to master (when documentation sources change)
- Manual trigger (workflow_dispatch)

Jobs:

1. Check documentation source: relative links, toolchain references, the
   examples catalog, the Allen-Cahn tutorial and fenced shell syntax.
2. Build unified documentation: Sphinx handbook with the Breathe API from
   Doxygen XML, themed from `docs/_static/openpfc/` (vendored from
   OpenPFCDesignSystem; `SOURCE` records the commit). Fails on new Doxygen
   or Sphinx diagnostics.
3. Render the website (VTT-ProperTune/OpenPFC, push to master only): checks
   out `ahojukka5/openpfc-pages` `master` with the `OPENPFC_PAGES_REPO`
   secret of the `OpenPFC` environment, renders it with pinned Quarto, puts
   the handbook at `docs/` (API at `docs/api/`), redirects the old `dev/` and
   `api/` URLs, and checks every link.
4. Deploy the website (same condition) to
   <https://vtt-propertune.github.io/OpenPFC/> with the Pages Actions API.
   Pages must be set to "GitHub Actions" in VTT-ProperTune/OpenPFC.

The fork builds and checks the documentation but does not deploy; the site
is published from the mirror push to VTT-ProperTune.

Typical Duration: 5-10 minutes

---

### `coverage.yml` - Code Coverage

Runs on: the 2nd of each month 02:00 UTC, and `workflow_dispatch`. Not on
pull requests or master.

### `asan.yml` - AddressSanitizer

Runs on: the 4th of each month 02:00 UTC, and `workflow_dispatch`. GCC 13
Debug with HeFFTe and OpenPFC built with `-fsanitize=address`.

Purpose: Measure and report test coverage (target: >90%).

Jobs:

1. Coverage Analysis
 - Builds with GCC 11 + coverage flags
 - Runs full test suite
 - Generates lcov coverage report
 - Uploads to Codecov
 - Comments coverage summary on PRs
 - Uploads HTML coverage report artifact

Coverage Targets:
- Line Coverage: >90%
- Function Coverage: >90%

Typical Duration: 15-20 minutes

Setup Required:
1. Create Codecov account (optional)
2. Add `CODECOV_TOKEN` secret to repository
3. Without token, artifact is still uploaded

---

## Secrets Configuration

Required secrets (configure in repository settings):

| Secret | Required | Purpose | Where to Get |
|--------|----------|---------|--------------|
| `CODECOV_TOKEN` | Optional | Upload coverage to Codecov | [codecov.io](https://codecov.io) |

---

## Caching Strategy

All workflows use GitHub Actions cache to speed up builds:

- HeFFTe builds: cached by `.github/actions/setup-build` per OS image,
  compiler, MPI implementation, build type and `scripts/install-heffte-ci.sh`
  content
- Cache retention: 7 days without use

Expected speedup: 2-3x faster on cache hits

---

## Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/VTT-ProperTune/OpenPFC/workflows/CI/badge.svg)](https://github.com/VTT-ProperTune/OpenPFC/actions/workflows/ci.yml)
[![Clang-Tidy](https://github.com/VTT-ProperTune/OpenPFC/workflows/Clang-Tidy/badge.svg)](https://github.com/VTT-ProperTune/OpenPFC/actions/workflows/clang-tidy.yml)
[![Documentation](https://github.com/VTT-ProperTune/OpenPFC/workflows/Documentation/badge.svg)](https://github.com/VTT-ProperTune/OpenPFC/actions/workflows/docs.yml)
[![Coverage](https://github.com/VTT-ProperTune/OpenPFC/workflows/Coverage/badge.svg)](https://github.com/VTT-ProperTune/OpenPFC/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/VTT-ProperTune/OpenPFC/branch/master/graph/badge.svg)](https://codecov.io/gh/VTT-ProperTune/OpenPFC)
```

---

## Troubleshooting

### Build Matrix Failures

Problem: One compiler/OS combination fails 
Solution: Check uploaded test logs in workflow artifacts

### Coverage Below Threshold

Problem: Coverage drops below 90% 
Solution: Add tests for uncovered code paths, view HTML coverage report

### Documentation Warnings

Problem: Doxygen warnings fail the build 
Solution: Fix warnings in source code documentation, or temporarily disable check

### Slow Builds

Problem: Workflows take >45 minutes 
Solution:
1. Ensure caching is working
2. Consider reducing matrix size

---

## Local Testing

Test workflows locally before pushing:

```bash
# Install act (GitHub Actions local runner)
# https://github.com/nektos/act

# Run CI workflow
act push

# Run the pull-request tier's common build
act pull_request -j build-and-test

# Run one cell of the compatibility matrix
act workflow_dispatch -W .github/workflows/compatibility.yml --matrix cc:gcc-14 --matrix build_type:Release
```

---

## Maintenance

### Adding New Compilers

Add a row to the `include:` matrix of `compatibility.yml`. Change the
pull-request or master tier in `ci.yml` only when the common setup changes.

### Updating Dependencies

When bumping HeFFTe (current release: v2.4.1), update the shared installer and cache keys:

- Tarball URL, source directory, and install prefix in `scripts/install-heffte-ci.sh`
- Cache keys using that prefix, e.g. `heffte-2.4.1-...` → `heffte-2.5.0-...` when moving to the next version

---

Last Updated: 2026-10-23
