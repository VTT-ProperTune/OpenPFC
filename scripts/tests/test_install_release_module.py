# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the release-module installer up to the build, without compiling."""
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "install_release_module.sh"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def release(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "CMakeLists.txt").write_text("project(OpenPFC VERSION 1.2.3)\n")
    git(repo, "add", "CMakeLists.txt")
    git(repo, "commit", "-qm", "release")
    git(repo, "-c", "tag.gpgSign=false", "tag", "-a", "-m", "release", "v1.2.3")
    root = tmp_path / "shared"
    for backend in ("cpu-ompi5", "cuda-ompi5"):
        (root / f"apps/heffte/2.4.1-{backend}/lib64/cmake/Heffte").mkdir(parents=True)
    mpicc = root / "apps/openmpi/5.0.10-cuda/bin/mpicc"
    mpicc.parent.mkdir(parents=True)
    mpicc.write_text("#!/bin/sh\n")
    mpicc.chmod(0o755)
    return repo, root, git(repo, "rev-parse", "HEAD")


def install(repo, root, *args):
    return subprocess.run(
        [str(SCRIPT), "--version=1.2.3", f"--root={root}", f"--repository={repo}", *args],
        capture_output=True,
        text=True,
    )


def test_prepare_cpu(release):
    repo, root, revision = release
    result = install(repo, root, "--prepare-only")
    assert result.returncode == 0, result.stderr
    stage = root / "build/openpfc-1.2.3-cpu"
    assert (stage / "src/CMakeLists.txt").is_file()
    module = (stage / "1.2.3-cpu.lua").read_text()
    assert revision in module
    assert 'family("openpfc")' in module
    assert 'depends_on("openmpi/5.0.10")' in module
    assert f'local root = "{root}/apps/openpfc/1.2.3-cpu"' in module
    assert f"{root}/apps/heffte/2.4.1-cpu-ompi5/lib64" in module
    assert not (root / "apps/openpfc").exists()
    assert not (root / "modulefiles").exists()


def test_prepare_cuda_uses_shared_mpi(release):
    repo, root, _ = release
    result = install(repo, root, "--backend=cuda", "--prepare-only")
    assert result.returncode == 0, result.stderr
    module = (root / "build/openpfc-1.2.3-cuda/1.2.3-cuda.lua").read_text()
    assert 'depends_on("cuda/13.1")' in module
    assert f'prepend_path("PATH", "{root}/apps/openmpi/5.0.10-cuda/bin")' in module
    assert "openmpi/5.0.10\")" not in module
    assert "CUDA architectures: 80;86;90" in module


@pytest.mark.parametrize(
    "args, message",
    [
        (["--root=/home/someone/shared"], "resolves under /home"),
        (["--heffte-prefix=/home/someone/heffte"], "resolves under /home"),
        (["--version=9.9.9"], "tag v9.9.9 not found"),
        (["--backend=rocm"], "--backend must be cpu or cuda"),
        (["--openmpi-root=/opt/mpi"], "applies to --backend=cuda only"),
        (["--cuda-architectures=sm_90"], "invalid --cuda-architectures"),
    ],
)
def test_refusals(release, args, message):
    repo, root, _ = release
    result = install(repo, root, "--prepare-only", *args)
    assert result.returncode != 0
    assert message in result.stderr


def test_existing_install_is_not_replaced(release):
    repo, root, _ = release
    (root / "apps/openpfc/1.2.3-cpu").mkdir(parents=True)
    result = install(repo, root, "--prepare-only")
    assert result.returncode != 0
    assert "is not replaced" in result.stderr
