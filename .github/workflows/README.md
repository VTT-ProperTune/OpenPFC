<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# GitHub Actions Workflows

This directory contains CI/CD workflows for the OpenPFC project.

## Workflows

### `ci.yml` - Main CI Pipeline

Runs on: Push to master/main/develop, PRs to master/main

Purpose: Primary continuous integration pipeline ensuring code quality and functionality.

Jobs: `code-quality` → `build-and-test` + `packaging-smoke`. If Code Quality fails, the build matrix is skipped.

1. Code Quality
 - clang-format 20 (advisory)
 - REUSE compliance (must pass)
 - kernel layering and GPU memcpy single-source checks

2. CMake Build Matrix
 - PRs: gcc-13 Debug + Release only
 - Pushes to master/develop: also gcc-11 Debug
 - Caches HeFFTe; full `ctest` including 2-rank MPI (`OpenPFC_MPI_TEST_MAX_WORLD_SIZE=2`)

3. Packaging smoke (`find_package` consumer)

4. CUDA/HIP compile-only: **push to master/develop only** (not PRs), `continue-on-error`

5. CI Status
 - Required for merging
 - Aggregates Code Quality, build-and-test, and packaging-smoke

Typical Duration: ~10–15 minutes on a PR (with HeFFTe cache)

---

### `clang-tidy.yml` - Static analysis (non-blocking)

Runs on: weekly Sunday 03:00 UTC, and `workflow_dispatch`. Not on pull requests.

Purpose: Run `scripts/run-clang-tidy.sh` with HeFFTe + `compile_commands.json`. A full-tree pass is ~45–60 min on GitHub runners and is `continue-on-error` until the WarningsAsErrors backlog is clear. Do not add this job as a required check.

Typical Duration: ~45–60 minutes.

---

### `docs.yml` - Documentation

Runs on: 
- Push to master/main
- PRs to master/main (when docs/, include/, or README.md changed)
- Manual trigger (workflow_dispatch)

Purpose: Build and deploy Doxygen documentation.

Jobs:

1. Build Documentation
 - Installs Doxygen, Graphviz, LaTeX
 - Generates HTML and PDF documentation
 - Checks for Doxygen warnings (fails on warnings)
 - Uploads documentation artifact

2. Deploy to GitHub Pages (master branch only)
 - Deploys to GitHub Pages
 - Comments deployment URL on commit
 - Requires Pages to be enabled in repository settings

Typical Duration: 10-15 minutes

Setup Required:
1. Enable GitHub Pages in repository settings
2. Set source to "GitHub Actions"

---

### `coverage.yml` - Code Coverage

Runs on:
- Push to master/main/develop
- Weekly schedule (Sunday 00:00 UTC)
- Manual trigger (workflow_dispatch)

Not on pull requests (duplicates a Debug `ctest` already in `ci.yml`).

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

- HeFFTe builds: Cached per OS/compiler/build-type
- Cache retention: 7 days

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

# Run specific job
act -j build-and-test

# Run with specific matrix combination
act -j build-and-test --matrix os:ubuntu-24.04 --matrix compiler:gcc-11
```

---

## Maintenance

### Adding New Compilers

Edit `ci.yml` matrix section:

```yaml
matrix:
 compiler: [gcc-11, gcc-13, gcc-14] # Add gcc-14
 include:
 - compiler: gcc-14
 cc: gcc-14
 cxx: g++-14
```

### Updating Dependencies

When bumping HeFFTe (current release: v2.4.1), update the shared installer and cache keys:

- Tarball URL, source directory, and install prefix in `scripts/install-heffte-ci.sh`
- Cache keys using that prefix, e.g. `heffte-2.4.1-...` → `heffte-2.5.0-...` when moving to the next version

---

Last Updated: 2026-03-30
