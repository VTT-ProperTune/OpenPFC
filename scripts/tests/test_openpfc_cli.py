# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


CLI = Path(__file__).resolve().parents[1] / "openpfc"
RANK_VARS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")


def call(*args, env=None, cli=CLI):
    clean = dict(os.environ)
    for name in (*RANK_VARS, "SLURM_JOB_ID"):
        clean.pop(name, None)
    clean.update(env or {})
    return subprocess.run([sys.executable, str(cli), *map(str, args)],
                          env=clean, text=True, capture_output=True)


@pytest.fixture
def case(tmp_path):
    path = tmp_path / "case with spaces"
    result = call("init", path)
    assert result.returncode == 0, result.stderr
    return path


def test_version_source_and_relocated_install(tmp_path):
    assert call("--version").stdout.startswith("OpenPFC 0.2.")
    binary = tmp_path / "prefix/bin/openpfc"
    binary.parent.mkdir(parents=True)
    shutil.copyfile(CLI, binary)
    version = tmp_path / "prefix/share/openpfc/version"
    version.parent.mkdir(parents=True)
    version.write_text("9.8.7-dev\n")
    assert call("--version", cli=binary).stdout == "OpenPFC 9.8.7-dev\n"
    assert call("init", tmp_path / "installed-case", cli=binary).returncode == 0


def test_init_refuses_overwrite(case):
    original = (case / "input.json").read_bytes()
    result = call("init", case)
    assert result.returncode != 0
    assert (case / "input.json").read_bytes() == original
    assert not (case / "results").exists()


@pytest.mark.parametrize("profile,flag,binary", [
    ("local", "--cpu", "tungsten"),
    ("tohtori", "--with-cuda", "tungsten_cuda"),
    ("lumi", "--with-rocm", "tungsten_hip"),
])
def test_compile_invokes_build_script_and_records_success(case, tmp_path, profile, flag, binary):
    source = tmp_path / "source with spaces"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    build = tmp_path / "build with spaces"
    # Stand in for the expensive site build; exercise the real CLI subprocess
    # boundary, profile flags, and success/failure publication of the manifest.
    (scripts / "build.sh").write_text(
        'printf "%s\\n" "$@" > arguments.txt\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in --build-dir=*) build="${arg#*=}";; esac\n'
        'done\n'
        'mkdir -p "$build/apps/tungsten"\n'
        'cp /bin/true "$build/apps/tungsten/' + binary + '"\n'
    )
    result = call("compile", case, "--profile", profile, "--source", source,
                  "--build-dir", build, "--jobs=2")
    assert result.returncode == 0, result.stderr
    arguments = (source / "arguments.txt").read_text().splitlines()
    assert flag in arguments
    assert "--test" in arguments
    if profile == "lumi":
        assert "--wait" in arguments
    manifest = json.loads((case / "openpfc.json").read_text())
    assert manifest["executable"] == str(build / "apps/tungsten" / binary)
    assert manifest["profile"] == profile
    before = (case / "openpfc.json").read_bytes()
    (scripts / "build.sh").write_text("exit 17\n")
    result = call("compile", case, "--source", source, "--build-dir", build)
    assert result.returncode == 17
    assert (case / "openpfc.json").read_bytes() == before


def test_dry_run_does_not_publish_build(case):
    before = (case / "openpfc.json").read_bytes()
    result = call("compile", case, "--source", CLI.parent.parent, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "scripts/build.sh" in result.stdout
    assert (case / "openpfc.json").read_bytes() == before


def test_build_tree_driver_finds_its_checkout(case, tmp_path):
    driver = tmp_path / "build/bin/openpfc"
    driver.parent.mkdir(parents=True)
    shutil.copyfile(CLI, driver)
    (driver.parent.parent / "CMakeCache.txt").write_text(
        "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(CLI.parent.parent) + "\n")
    result = call("compile", case, "--dry-run", cli=driver)
    assert result.returncode == 0, result.stderr
    assert str(CLI.parent / "build.sh") in result.stdout


@pytest.mark.parametrize("env,extra,expected", [
    ({}, ["--ranks=2"], "mpirun -n 2"),
    ({"SLURM_JOB_ID": "123"}, [], "srun "),
    ({"SLURM_JOB_ID": "123"}, ["--ranks=3"], "srun --ntasks 3"),
])
def test_launcher_selection(case, env, extra, expected):
    result = call("run", case, "--executable=/bin/true", "--dry-run", *extra, env=env)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert not (case / "results").exists()


@pytest.mark.parametrize("rank_var", RANK_VARS)
def test_outer_mpi_does_not_nest_launchers(case, rank_var):
    result = call("run", case, "--executable=/bin/true", "--dry-run", env={rank_var: "0"})
    assert result.returncode == 0, result.stderr
    assert "mpirun" not in result.stdout and "srun" not in result.stdout
    result = call("run", case, "--executable=/bin/true", "--ranks=2", env={rank_var: "0"})
    assert result.returncode != 0
    assert "already inside MPI" in result.stderr


def test_run_sets_case_directory_and_propagates_exit(case, tmp_path):
    app = tmp_path / "fake app"
    app.write_text('#!/bin/sh\npwd > cwd.txt\nprintf "%s\\n" "$@" > argv.txt\nexit 19\n')
    app.chmod(0o755)
    result = call("run", case, "--launcher=none", "--executable", app)
    assert result.returncode == 19
    assert (case / "cwd.txt").read_text().strip() == str(case)
    assert (case / "argv.txt").read_text().strip() == str(case / "input.json")


def test_invalid_ranks_and_missing_binary(case):
    assert call("run", case, "--ranks=0").returncode != 0
    result = call("run", case, "--executable=/does/not/exist")
    assert result.returncode != 0
    assert "compile the case" in result.stderr
