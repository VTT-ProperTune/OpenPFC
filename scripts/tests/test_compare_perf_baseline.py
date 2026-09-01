#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from compare_perf_baseline import (  # noqa: E402
    classify,
    load_profiling_json,
    main,
    mean_metric,
)


def _doc(wall_steps: list[float], *, version: int = 2) -> dict:
    frames = [{"scalars": [float(i), 0.0, w], "regions": {}} for i, w in enumerate(wall_steps)]
    payload = {
        "schema_version": version,
        "n_mpi_ranks": 1,
        "total_frames": len(wall_steps),
        "frame_metric_names": ["step", "mpi_rank", "wall_step"],
        "region_paths": [],
        "ranks": [{"mpi_rank": 0, "n_frames": len(wall_steps), "frames": frames}],
    }
    if version == 3:
        payload["run_id"] = "test"
        payload["metadata"] = {"host": "test"}
    return payload


def test_mean_and_classify_pass_warn_fail(tmp_path: Path) -> None:
    base = _doc([1.0, 1.0])
    assert mean_metric(base, "wall_step", warmup_frames=0) == pytest.approx(1.0)
    assert classify(1.0, 1.04) == "PASS"
    assert classify(1.0, 1.06) == "WARN"
    assert classify(1.0, 1.20) == "FAIL"
    assert classify(1.0, 0.50) == "PASS"


def test_warmup_skips_leading_frames() -> None:
    doc = _doc([10.0, 1.0, 1.0])
    assert mean_metric(doc, "wall_step", warmup_frames=1) == pytest.approx(1.0)


def test_schema_v3_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "v3.json"
    path.write_text(json.dumps(_doc([2.0], version=3)), encoding="utf-8")
    loaded = load_profiling_json(path)
    assert loaded["schema_version"] == 3
    assert mean_metric(loaded, "wall_step", warmup_frames=0) == pytest.approx(2.0)


def test_cli_fail_and_warn_exit_codes(tmp_path: Path) -> None:
    baseline = tmp_path / "base.json"
    current = tmp_path / "cur.json"
    baseline.write_text(json.dumps(_doc([1.0, 1.0])), encoding="utf-8")
    current.write_text(json.dumps(_doc([1.20, 1.20])), encoding="utf-8")
    assert main([str(baseline), str(current)]) == 1

    current.write_text(json.dumps(_doc([1.06, 1.06])), encoding="utf-8")
    assert main([str(baseline), str(current)]) == 0
    assert main([str(baseline), str(current), "--fail-on-warn"]) == 1

    current.write_text(json.dumps(_doc([0.90, 0.90])), encoding="utf-8")
    assert main([str(baseline), str(current)]) == 0
