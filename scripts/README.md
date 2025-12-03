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

## Build Scripts

### build_cuda.sh

**Purpose**: Automated build script for OpenPFC with optional CUDA support.

**Usage**:
```bash
# Build with CUDA support (default)
./scripts/build_cuda.sh

# Build without CUDA
./scripts/build_cuda.sh --no-cuda

# Build Release with custom job count
./scripts/build_cuda.sh --build-type Release --jobs 16

# Build without cleaning first
./scripts/build_cuda.sh --no-clean
```

**What it does**:
- Automatically loads required modules (`cuda`, `openmpi/4.1.1`)
- Sets correct compiler paths (GCC 11.2.0)
- Selects appropriate HeFFTe version (2.4.1 for CPU, 2.4.1-cuda for GPU)
- Configures CMake with correct options
- Builds with specified number of parallel jobs
- Provides clear status messages and error handling

**Key Features**:
- ✅ Handles module loading automatically
- ✅ Correctly sets compiler paths (fixes CMake auto-detection issues)
- ✅ Selects HeFFTe version based on CUDA enablement
- ✅ Cleans build directory by default (use `--no-clean` to skip)
- ✅ Works on both AMD (no CUDA) and NVIDIA (with CUDA) systems

**Options**:
- `--build-type TYPE`: Debug or Release (default: Debug)
- `--cuda` / `--no-cuda`: Enable/disable CUDA (default: enabled)
- `--jobs N, -j N`: Number of parallel build jobs (default: 8)
- `--no-clean`: Don't clean build directory before building
- `--help, -h`: Show help message

**Examples**:
```bash
# Standard CUDA build
./scripts/build_cuda.sh

# CPU-only build
./scripts/build_cuda.sh --no-cuda

# Release build with 16 jobs
./scripts/build_cuda.sh --build-type Release -j 16
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
