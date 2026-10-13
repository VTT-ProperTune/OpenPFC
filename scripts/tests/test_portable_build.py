# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise the canonical entry point without HPC modules or toolchains."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "build.sh"


def test_portable_preserves_toolchain_environment(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    trace = tmp_path / "trace"
    cmake = tools / "cmake"
    cmake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$TRACE"\n')
    cmake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tools}:/usr/bin:/bin", TRACE=str(trace),
               CC="/usr/bin/cc", CXX="/usr/bin/c++")
    env.pop("MACHINE", None)
    result = subprocess.run([str(SCRIPT), "--machine=portable", "--cpu", "--no-test",
                             f"--build-dir={tmp_path / 'build'}", "--jobs=2",
                             "--cmake-arg=-DMPI_HOME=/chosen/mpi"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = trace.read_text()
    assert "-DMPI_HOME=/chosen/mpi" in commands
    assert "--build" in commands
    assert "CMAKE_TOOLCHAIN_FILE" not in commands
    assert "MPI_C_COMPILER" not in commands
    assert "caller-supplied environment" in result.stdout
