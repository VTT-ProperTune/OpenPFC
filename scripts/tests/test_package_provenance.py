# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check clean, dirty and archive source identity without compiling the library."""
import subprocess
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "cmake/SourceProvenance.cmake"


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def identity(source, script):
    script.write_text(
        f'set(PROJECT_SOURCE_DIR "{source.as_posix()}")\n'
        f'include("{MODULE.as_posix()}")\n'
        'message("IDENTITY=${OpenPFC_SOURCE_REVISION}:${OpenPFC_SOURCE_STATE}")\n'
    )
    output = subprocess.check_output(
        ["cmake", "-P", str(script)], stderr=subprocess.STDOUT, text=True
    )
    return output.split("IDENTITY=")[1].splitlines()[0]


def test_source_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run("git", "init", "-q", str(source))
    run("git", "-C", str(source), "config", "user.name", "Test")
    run("git", "-C", str(source), "config", "user.email", "test@example.invalid")
    (source / "code.cpp").write_text("original\n")
    run("git", "-C", str(source), "add", "code.cpp")
    run("git", "-C", str(source), "commit", "-qm", "initial")
    revision = run("git", "-C", str(source), "rev-parse", "HEAD")
    script = tmp_path / "check.cmake"
    assert identity(source, script) == f"{revision}:clean"
    (source / "code.cpp").write_text("changed\n")
    assert identity(source, script) == f"{revision}:dirty"
    (source / "code.cpp").write_text("original\n")
    (source / "extra.hpp").write_text("untracked\n")
    assert identity(source, script) == f"{revision}:dirty"
    archive = tmp_path / "archive"
    archive.mkdir()
    assert identity(archive, script) == "unknown:unknown"
    nested_archive = source / "archive"
    nested_archive.mkdir()
    assert identity(nested_archive, script) == "unknown:unknown"
