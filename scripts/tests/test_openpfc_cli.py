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


@pytest.mark.parametrize("app,preset", [
    ("aluminum", "smoke"), ("cahn_hilliard", "spinodal"),
    ("cahn_hilliard", "mode"), ("thin_film", "leveling"), ("thin_film", "dewetting"),
    ("surface_diffusion", "smoothing"), ("kawahara", "pulse"), ("ehd_film", "relaxation"),
    ("gradient_elasticity", "inclusion"), ("gradient_elasticity", "gaussian"),
])
def test_catalog_presets_and_installed_data(tmp_path, app, preset):
    case = tmp_path / "case"
    result = call("init", case, "--app", app, "--preset", preset)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((case / "openpfc.json").read_text())
    assert manifest["app"] == app and manifest["preset"] == preset
    settings = json.loads((case / "input.json").read_text())
    assert settings["model"]["name"] == app
    assert all(isinstance(settings["domain"][key], int) for key in ("Lx", "Ly", "Lz"))
    assert all(f["data"].startswith("results/") for f in settings["fields"])
    prefix = tmp_path / "prefix"
    driver = prefix / "bin/openpfc"
    driver.parent.mkdir(parents=True)
    shutil.copyfile(CLI, driver)
    directory = "aluminumNew" if app == "aluminum" else app
    shutil.copytree(CLI.parent.parent / "apps" / directory / "inputs_json",
                    prefix / "share/openpfc/cases" / directory / "inputs_json")
    relocated = tmp_path / "relocated-case"
    result = call("init", relocated, "--app", app, "--preset", preset, cli=driver)
    assert result.returncode == 0, result.stderr
    assert (relocated / "input.json").read_bytes() == (case / "input.json").read_bytes()


def test_catalog_and_unsupported_backend(tmp_path):
    result = call("apps")
    assert "cahn_hilliard" in result.stdout and "spinodal, mode" in result.stdout
    case = tmp_path / "case"
    assert call("init", case, "--app=cahn_hilliard", "--preset=bad").returncode != 0
    assert not case.exists()
    assert call("init", case, "--app=cahn_hilliard").returncode == 0
    before = (case / "openpfc.json").read_bytes()
    result = call("compile", case, "--profile=tohtori", "--dry-run")
    assert result.returncode != 0 and "no cuda target" in result.stderr
    assert (case / "openpfc.json").read_bytes() == before


def test_invalid_manifest_is_a_clear_error(case):
    manifest = json.loads((case / "openpfc.json").read_text())
    manifest["profile"] = "missing"
    (case / "openpfc.json").write_text(json.dumps(manifest))
    result = call("run", case)
    assert result.returncode == 2 and "unsupported profile" in result.stderr
    manifest.pop("profile")
    manifest["app"] = []
    (case / "openpfc.json").write_text(json.dumps(manifest))
    result = call("run", case)
    assert result.returncode == 2 and "unsupported openpfc.json" in result.stderr


@pytest.mark.parametrize("app,binary,directory,profile", [
    ("cahn_hilliard", "cahn_hilliard", "cahn_hilliard", "local"),
    ("cahn_hilliard", "cahn_hilliard_hip", "cahn_hilliard", "lumi"),
    ("aluminum", "aluminum_etd_cuda", "aluminumNew", "tohtori"),
])
def test_multi_app_build_selection(tmp_path, app, binary, directory, profile):
    case = tmp_path / "case"
    assert call("init", case, "--app", app).returncode == 0
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/build.sh").write_text("exit 0\n")
    build = tmp_path / "build"
    executable = build / "apps" / directory / binary
    executable.parent.mkdir(parents=True)
    shutil.copyfile("/bin/true", executable)
    executable.chmod(0o755)
    result = call("compile", case, "--source", source, "--build-dir", build, "--profile", profile)
    assert result.returncode == 0, result.stderr
    assert json.loads((case / "openpfc.json").read_text())["executable"] == str(executable)
    assert call("run", case, "--launcher=none").returncode == 0
