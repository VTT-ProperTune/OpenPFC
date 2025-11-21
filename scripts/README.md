<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Scripts Directory

This directory contains utility scripts for OpenPFC development and workflow automation.

## Development Scripts

### pre-commit-hook

**Purpose**: Automatically check code formatting before commits to prevent CI failures.

**Installation** (Required for all developers):

```bash
# From the project root directory
cp scripts/pre-commit-hook .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**What it does**:
- Runs automatically before each `git commit`
- Checks C++ files (`.cpp`, `.hpp`, `.h`, `.cc`, `.cxx`) for formatting issues
- Uses `clang-format` to verify code style compliance
- Blocks commits if formatting issues are found

**Requirements**:
- `clang-format` (version 17+ recommended, minimum version 9.0)
- On Rocky Linux 8: `dnf install clang` (provides clang-format 19+)

**Why this is mandatory**:
The CI/CD pipeline runs `clang-format` checks and **will fail** if code is not properly formatted. Installing this hook ensures you catch formatting issues locally before pushing, saving CI time and preventing failed builds.

**If formatting check fails**:

```bash
# Fix all staged files automatically
clang-format -i path/to/file.cpp

# Or let the hook tell you which files need fixing
# It will provide the exact command to run
```

**Testing the hook**:

```bash
# Stage a C++ file with formatting issues
git add tests/unit/fft/test_fft.cpp

# Try to commit (hook will check formatting)
git commit -m "test"

# If it passes, you're good!
# If it fails, run the suggested clang-format command
```

**Bypassing the hook** (not recommended):

```bash
# Only use this if you have a very good reason
git commit --no-verify -m "emergency fix"
```

## Other Scripts

### xdmfgen.py

Generate XDMF files for visualization with ParaView.

### pvrender.py

Render images from ParaView for batch visualization.

---

## Contributing

When adding new development scripts:

1. Add them to this directory
2. Make them executable: `chmod +x script_name`
3. Add SPDX license headers
4. Document them in this README
5. If they're part of the required workflow, mark them as such
