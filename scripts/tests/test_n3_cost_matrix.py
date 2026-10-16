#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the OpenPFC #124 N^3 cost-matrix tooling."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "heat3d" / "scripts"))

import n3_cost_matrix as m  # noqa: E402

SUBMIT = ROOT / "docs" / "lumi_slurm" / "submit_heat3d_n3_cost.sh"
BATCH = ROOT / "docs" / "lumi_slurm" / "heat3d_n3_cost.sbatch"


def _run(args, env, cwd):
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        universal_newlines=True,
    )
    out, err = proc.communicate()
    proc.stdout_text = out
    proc.stderr_text = err
    return proc


def _clean_env():
    env = os.environ.copy()
    for key in (
        "ACCOUNT",
        "PARTITION",
        "DRY_RUN",
        "OPENPFC_SRC",
        "OPENPFC_SCALING_ROOT",
        "HEAT3D_SPECTRAL_HIP_BIN",
        "HEAT3D_HIP_BIN",
        "HEAT3D_USE_PENCILS",
        "SBATCH_ACCOUNT",
        "HEAT3D_REPEAT",
        "BUILD_DIR",
        "HEFFTE_MODULE",
        "HEFFTE_PREFIX",
    ):
        env.pop(key, None)
    return env


def test_frozen_matrix_has_24_cells():
    cells = list(m.cells())
    assert len(cells) == 24
    assert m.check() == 0


def test_submit_check(tmp_path):
    env = _clean_env()
    proc = _run(["bash", str(SUBMIT), "check"], env, ROOT)
    assert proc.returncode == 0, proc.stderr_text
    assert "24 cells" in proc.stdout_text


def test_submit_refuses_wrong_account():
    env = _clean_env()
    env["ACCOUNT"] = "project_462001245"
    proc = _run(["bash", str(SUBMIT), "check"], env, ROOT)
    assert proc.returncode == 2
    assert "refusing ACCOUNT" in proc.stderr_text


def test_submit_dry_run_three_repeats(tmp_path):
    env = _clean_env()
    env["DRY_RUN"] = "1"
    env["HEAT3D_SPECTRAL_HIP_BIN"] = str(tmp_path / "heat3d_spectral_hip")
    env["HEAT3D_HIP_BIN"] = str(tmp_path / "heat3d_fd_hip")
    (tmp_path / "heat3d_spectral_hip").write_text("x")
    (tmp_path / "heat3d_fd_hip").write_text("x")
    os.chmod(tmp_path / "heat3d_spectral_hip", 0o755)
    os.chmod(tmp_path / "heat3d_fd_hip", 0o755)
    proc = _run(["bash", str(SUBMIT), "submit"], env, ROOT)
    assert proc.returncode == 0, proc.stderr_text + proc.stdout_text
    assert "h3-n3-r1" in proc.stdout_text
    assert "h3-n3-r2" in proc.stdout_text
    assert "h3-n3-r3" in proc.stdout_text
    assert "/scratch/project_462001519/juaho/openpfc-scaling/heat3d-n3-cost-592" in proc.stdout_text
    assert "/scratch/project_462001519/juaho/logs" in proc.stdout_text
    assert "768" not in proc.stdout_text
    assert "fd4" not in proc.stdout_text.lower()


def test_build_dry_run_uses_production_heffte():
    env = _clean_env()
    env["DRY_RUN"] = "1"
    env["BUILD_DIR"] = "/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace-heat3d"
    env["HEFFTE_MODULE"] = "heffte-rocm-trace/2.4.1"
    proc = _run(["bash", str(SUBMIT), "build"], env, ROOT)
    assert proc.returncode == 0, proc.stderr_text
    assert "n3-cost-592" in proc.stdout_text
    assert "/scratch/project_462001519/juaho/logs" in proc.stdout_text
    assert "heffte-trace" not in proc.stdout_text


def test_sbatch_is_one_node_standard_g():
    text = BATCH.read_text()
    assert "--nodes=1" in text
    assert "--ntasks=8" in text
    assert "project_462001519" in text
    assert "use_pencils=0" in text
    assert "HEAT3D_WARMUP" in text
    assert "768" not in text
    assert "1536" not in text


def _write_cell(root: Path, tag: str, n: int, method: str, order: int, repeat: int,
                wall: float, **meta):
    run = root / "runs" / tag
    run.mkdir(parents=True)
    keys = {
        "job": "3001",
        "tag": tag,
        "repeat": str(repeat),
        "N": str(n),
        "method": method,
        "fd_order": str(order),
        "account": m.ACCOUNT,
        "partition": m.PARTITION,
        "nodes": "1",
        "ntasks": "8",
        "revision": "deadbeef",
        "dirty": "0",
        "steps": "30",
        "warmup": "5",
        "dt": "0.01",
        "bin_sha256": "abc",
        "gpu_aware": "1",
    }
    if method == "spectral":
        keys["use_pencils"] = "0"
    keys.update(meta)
    (run / "run_meta.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in keys.items()) + "\n"
    )
    frames = [{"scalars": [5 + i, 0, wall], "regions": {}} for i in range(25)]
    (run / "timing_profile.json").write_text(
        json.dumps(
            {
                "frame_metric_names": ["step", "mpi_rank", "wall_step"],
                "n_mpi_ranks": 8,
                "ranks": [{"mpi_rank": 0, "frames": frames}],
            }
        )
    )
    extra = "HEAT3D_SPECTRAL_HIP N=512 ranks=8 use_pencils=0 gpu_aware=1\n"
    if method == "fd":
        extra = f"HEAT3D_HIP_HALO_MODE=device gpu_aware=1 N={n} fd_order={order}\n"
    (run / "cell.log").write_text(extra)


def test_harvest_admits_matched_cell(tmp_path):
    _write_cell(tmp_path, "n512_spectral_r1", 512, "spectral", 0, 1, 0.05)
    rows = m.harvest_root(tmp_path)
    assert rows[0]["admitted"] == "yes"
    assert abs(rows[0]["wall_step_ms"] - 50.0) < 1e-9


def test_harvest_rejects_wrong_account(tmp_path):
    _write_cell(
        tmp_path,
        "n512_spectral_r1",
        512,
        "spectral",
        0,
        1,
        0.05,
        account="project_462001245",
    )
    rows = m.harvest_root(tmp_path)
    assert rows[0]["admitted"] == "no"
    assert "account" in rows[0]["reject_reason"]
