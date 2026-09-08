# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "bake_case.py"


@pytest.fixture
def bundle(tmp_path):
    case = tmp_path / "case with spaces"
    case.mkdir()
    build = tmp_path / "build"
    (build / "share/openpfc").mkdir(parents=True)
    (build / "share/openpfc/version").write_text("0.2.0\n")
    app = build / "tungsten"
    app.write_bytes(b"#!/bin/sh\nexit 0\n")
    app.chmod(0o755)
    host = tmp_path / "host-mpi"
    host.mkdir()
    mpi = host / "libmpi.so.40"
    mpi.write_bytes(b"MPI stays on host")
    physics = tmp_path / "libheffte.so.2"
    physics.write_bytes(b"packaged physics dependency")
    libc = tmp_path / "libc.so.6"
    libc.write_bytes(b"base supplies its own libc")
    fake_tools = tmp_path / "tools"
    fake_tools.mkdir()
    ldd = fake_tools / "ldd"
    lines = ["libmpi.so.40 => {} (0x123)".format(mpi),
             "libheffte.so.2 => {} (0x456)".format(physics),
             "libc.so.6 => {} (0x789)".format(libc)]
    ldd.write_text("#!/bin/sh\nprintf '%s\\n' " + " ".join(map(shlex.quote, lines)) + "\n")
    ldd.chmod(0o755)
    (case / "openpfc.json").write_text(json.dumps({
        "format_version": 1, "app": "tungsten", "input": "input.json", "profile": "local",
        "executable": str(app), "build_dir": str(build), "source": str(tmp_path),
    }))
    (case / "input.json").write_text(json.dumps({
        "initial_conditions": [{"type": "constant", "n0": -0.4}],
        "fields": [{"data": "results/psi_%d.vti"}],
    }))
    base = tmp_path / "base.sif"
    base.write_bytes(b"test base identity")
    output = tmp_path / "bundle with spaces"
    command = [sys.executable, str(SCRIPT), str(case), "--base", str(base),
               "--output", str(output), "--host-prefix", str(host), "--prepare-only"]
    env = dict(os.environ, PATH=str(fake_tools) + os.pathsep + os.environ["PATH"])
    return command, env, output, case, ldd


def test_prepare_cli_packages_case_but_not_mpi_or_glibc(bundle):
    command, env, output, case, _ = bundle
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = output / "payload"
    assert (payload / "case/input.json").read_bytes() == (case / "input.json").read_bytes()
    assert (payload / "lib/libheffte.so.2").exists()
    assert not (payload / "lib/libmpi.so.40").exists()
    assert not (payload / "lib/libc.so.6").exists()
    provenance = json.loads((payload / "provenance.json").read_text())
    assert "libmpi.so.40" in provenance["host_libraries"]
    assert len(provenance["binary_sha256"]) == 64
    assert len(provenance["base_sha256"]) == 64
    subprocess.run(["sh", "-n", str(output / "run-host.sh")], check=True)
    entrypoint = [sys.executable, str(payload / "bin/baked-case")]
    version = subprocess.run(entrypoint + ["--version"], text=True, capture_output=True, check=True)
    assert version.stdout == "OpenPFC 0.2.0\n"
    copied = output / "copied-case"
    subprocess.run(entrypoint + ["init", str(copied)], check=True)
    assert (copied / "input.json").read_bytes() == (case / "input.json").read_bytes()
    subprocess.run(entrypoint + ["run", str(copied)], check=True)
    assert (copied / "results").is_dir()
    before = (payload / "provenance.json").read_bytes()
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    assert (payload / "provenance.json").read_bytes() == before


def test_unresolved_libraries_fail_before_staging(bundle):
    command, env, output, _, ldd = bundle
    ldd.write_text("#!/bin/sh\necho 'libmpi.so.40 => not found'\n")
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "load the build's compiler/MPI modules" in result.stderr
    assert not output.exists()


def test_mpi_must_be_in_declared_host_prefix(bundle, tmp_path):
    command, env, output, _, _ = bundle
    command[command.index("--host-prefix") + 1] = str(tmp_path / "case with spaces")
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "needs --host-prefix" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("settings", [
    {"restart_from": "results/checkpoints/step_10"},
    {"initial_conditions": [{"type": "from_file", "filename": "missing.bin"}]},
    {"fields": [{"data": "/old/machine/psi.bin"}]},
    {"checkpoint": {"directory": "results/../../outside"}},
])
def test_nonportable_case_is_rejected(bundle, settings):
    command, env, output, case, _ = bundle
    (case / "input.json").write_text(json.dumps(settings))
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert not output.exists()


@pytest.fixture
def cray_bundle(tmp_path):
    """A LUMI-shaped case: Cray MPICH naming, and Slingshot libraries sharing a
    directory with ordinary system libraries."""
    case = tmp_path / "case"
    case.mkdir()
    build = tmp_path / "build"
    (build / "share/openpfc").mkdir(parents=True)
    (build / "share/openpfc/version").write_text("0.2.0\n")
    app = build / "tungsten"
    app.write_bytes(b"#!/bin/sh\nexit 0\n")
    app.chmod(0o755)
    host = tmp_path / "cray-pe"
    host.mkdir()
    for name in ("libmpi_gnu_123.so.12", "libfabric.so.1", "libpmi.so.0"):
        (host / name).write_bytes(b"cray host stack")
    # /usr/lib64 equivalent: Slingshot and ordinary system libraries together.
    sysdir = tmp_path / "usr-lib64"
    sysdir.mkdir()
    (sysdir / "libcxi.so.1").write_bytes(b"slingshot provider, host only")
    (sysdir / "libstdc++.so.6").write_bytes(b"ordinary system library, bundle it")
    physics = tmp_path / "libheffte.so.2"
    physics.write_bytes(b"packaged physics dependency")
    libc = tmp_path / "libc.so.6"
    libc.write_bytes(b"base supplies its own libc")
    tools = tmp_path / "tools"
    tools.mkdir()
    lines = ["libmpi_gnu_123.so.12 => {} (0x1)".format(host / "libmpi_gnu_123.so.12"),
             "libfabric.so.1 => {} (0x2)".format(host / "libfabric.so.1"),
             "libpmi.so.0 => {} (0x3)".format(host / "libpmi.so.0"),
             "libcxi.so.1 => {} (0x4)".format(sysdir / "libcxi.so.1"),
             "libstdc++.so.6 => {} (0x5)".format(sysdir / "libstdc++.so.6"),
             "libheffte.so.2 => {} (0x6)".format(physics),
             "libc.so.6 => {} (0x7)".format(libc)]
    ldd = tools / "ldd"
    ldd.write_text("#!/bin/sh\nprintf '%s\\n' " + " ".join(map(shlex.quote, lines)) + "\n")
    ldd.chmod(0o755)
    (case / "openpfc.json").write_text(json.dumps({
        "format_version": 1, "app": "tungsten", "input": "input.json", "profile": "lumi",
        "executable": str(app), "build_dir": str(build), "source": str(tmp_path),
    }))
    (case / "input.json").write_text(json.dumps({
        "initial_conditions": [{"type": "constant", "n0": -0.4}],
        "fields": [{"data": "results/psi_%d.vti"}],
    }))
    base = tmp_path / "base.sif"
    base.write_bytes(b"test base identity")
    output = tmp_path / "bundle"
    command = [sys.executable, str(SCRIPT), str(case), "--base", str(base),
               "--output", str(output), "--site", "cray",
               "--host-prefix", str(host), "--prepare-only"]
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"])
    return command, env, output, sysdir


def test_cray_slingshot_library_is_not_silently_bundled(cray_bundle):
    """libcxi shares /usr/lib64 with ordinary libraries, so a prefix rule cannot
    reach it. Baking it would put the interconnect inside the image."""
    command, env, output, _ = cray_bundle
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 2, result.stdout
    assert "libcxi.so.1" in result.stderr
    assert "--host-lib" in result.stderr
    assert not output.exists()


def test_cray_host_lib_splits_a_shared_system_directory(cray_bundle):
    command, env, output, sysdir = cray_bundle
    command = command + ["--host-lib", str(sysdir / "libcxi.so.1")]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = output / "payload"
    # Same directory, opposite sides of the split.
    assert not (payload / "lib/libcxi.so.1").exists()
    assert (payload / "lib/libstdc++.so.6").is_file()
    assert (payload / "lib/libheffte.so.2").is_file()
    provenance = json.loads((payload / "provenance.json").read_text())
    assert provenance["site"] == "cray"
    assert provenance["profile"] == "lumi"
    assert "libcxi.so.1" in provenance["host_libraries"]
    assert "libmpi_gnu_123.so.12" in provenance["host_libraries"]
    assert "libstdc++.so.6" in provenance["bundled_libraries"]
    # The wrapper binds the single file, never the shared directory.
    wrapper = (output / "run-host.sh").read_text()
    assert "libcxi.so.1:/opt/openpfc/hostlib/libcxi.so.1:ro" in wrapper
    assert str(sysdir) + ":" + str(sysdir) not in wrapper
    assert "/opt/openpfc/hostlib" in wrapper


def test_cray_mpi_naming_counts_as_an_mpi_application(cray_bundle):
    """Cray names its library libmpi_gnu_123.so.12, not libmpi.so.*."""
    command, env, output, sysdir = cray_bundle
    command = command + ["--host-lib", str(sysdir / "libcxi.so.1")]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "expected a dynamically linked MPI application" not in result.stderr


def test_openmpi_site_rejects_a_lumi_profile_case(cray_bundle):
    command, env, output, sysdir = cray_bundle
    command = [a for a in command if a not in ("cray",)]
    command = [a for a in command if a != "--site"]
    command += ["--host-lib", str(sysdir / "libcxi.so.1")]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "profile" in result.stderr


def test_cray_wrapper_binds_the_slurm_pmi_socket_directory(cray_bundle):
    """Cray PMI reaches the Slurm daemon through /var/spool/slurmd. Without it
    MPI_Init fails and every rank believes it is rank 0."""
    command, env, output, sysdir = cray_bundle
    command = command + ["--host-lib", str(sysdir / "libcxi.so.1")]
    assert subprocess.run(command, env=env, capture_output=True, text=True).returncode == 0
    assert "/var/spool/slurmd:/var/spool/slurmd" in (output / "run-host.sh").read_text()


def test_openmpi_wrapper_has_no_cray_site_binds(bundle):
    command, env, output, _, _ = bundle
    assert subprocess.run(command, env=env, capture_output=True, text=True).returncode == 0
    assert "/var/spool/slurmd" not in (output / "run-host.sh").read_text()
