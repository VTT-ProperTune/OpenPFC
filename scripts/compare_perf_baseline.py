#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare an OpenPFC profiling JSON export against a stored baseline.

Reads schema v2 or v3 files from ``ProfilingSession::finalize_and_export``.
The compared value is the mean of a named frame scalar (default ``wall_step``)
across all ranks and frames after an optional warmup skip.

Thresholds (regression = current slower than baseline):

* pass:  regression <= 5%
* warn:  5% < regression <= 15%  (exit 0 unless ``--fail-on-warn``)
* fail:  regression > 15%        (exit 1)

Speedups always pass. Use ``--strict`` to treat warn as fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WARN_RATIO = 0.05
FAIL_RATIO = 0.15
DEFAULT_METRIC = "wall_step"


def load_profiling_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    version = data.get("schema_version")
    if version not in (2, 3):
        raise ValueError(f"{path}: unsupported schema_version {version!r} (need 2 or 3)")
    if "ranks" not in data or "frame_metric_names" not in data:
        raise ValueError(f"{path}: missing ranks or frame_metric_names")
    return data


def metric_index(doc: Mapping[str, Any], name: str) -> int:
    names = doc["frame_metric_names"]
    try:
        return list(names).index(name)
    except ValueError as exc:
        raise ValueError(
            f"metric {name!r} not in frame_metric_names {list(names)}"
        ) from exc


def iter_scalar_values(
    doc: Mapping[str, Any], name: str, *, warmup_frames: int
) -> Iterable[float]:
    idx = metric_index(doc, name)
    for rank in doc["ranks"]:
        frames = rank.get("frames") or []
        for i, frame in enumerate(frames):
            if i < warmup_frames:
                continue
            scalars = frame.get("scalars") or []
            if idx >= len(scalars):
                raise ValueError(
                    f"rank {rank.get('mpi_rank')} frame {i}: scalars length "
                    f"{len(scalars)} < index {idx} for {name!r}"
                )
            yield float(scalars[idx])


def mean_metric(doc: Mapping[str, Any], name: str, *, warmup_frames: int) -> float:
    values = list(iter_scalar_values(doc, name, warmup_frames=warmup_frames))
    if not values:
        raise ValueError(f"no samples for metric {name!r} after warmup_frames={warmup_frames}")
    return sum(values) / len(values)


def classify(baseline: float, current: float) -> str:
    if baseline <= 0.0:
        raise ValueError(f"baseline mean must be positive, got {baseline}")
    regression = (current - baseline) / baseline
    if regression > FAIL_RATIO:
        return "FAIL"
    if regression > WARN_RATIO:
        return "WARN"
    return "PASS"


def format_report(
    *,
    baseline_path: Path,
    current_path: Path,
    metric: str,
    baseline: float,
    current: float,
    status: str,
    n_baseline: int,
    n_current: int,
) -> str:
    delta = current - baseline
    pct = 100.0 * delta / baseline
    return (
        f"{status}  metric={metric}\n"
        f"  baseline: {baseline:.6g} s  ({n_baseline} samples, {baseline_path})\n"
        f"  current:  {current:.6g} s  ({n_current} samples, {current_path})\n"
        f"  delta:    {delta:+.6g} s  ({pct:+.2f}%)  "
        f"[warn >{100 * WARN_RATIO:.0f}%, fail >{100 * FAIL_RATIO:.0f}% regression]"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path, help="Stored profiling JSON (schema v2/v3)")
    p.add_argument("current", type=Path, help="New profiling JSON to compare")
    p.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"frame_metric_names entry to compare (default: {DEFAULT_METRIC})",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Skip this many leading frames per rank (default: 0)",
    )
    p.add_argument(
        "--fail-on-warn",
        "--strict",
        dest="fail_on_warn",
        action="store_true",
        help="Exit 1 on WARN as well as FAIL",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_doc = load_profiling_json(args.baseline)
    current_doc = load_profiling_json(args.current)
    baseline_mean = mean_metric(
        baseline_doc, args.metric, warmup_frames=args.warmup_frames
    )
    current_mean = mean_metric(
        current_doc, args.metric, warmup_frames=args.warmup_frames
    )
    n_base = len(
        list(iter_scalar_values(baseline_doc, args.metric, warmup_frames=args.warmup_frames))
    )
    n_cur = len(
        list(iter_scalar_values(current_doc, args.metric, warmup_frames=args.warmup_frames))
    )
    status = classify(baseline_mean, current_mean)
    report = format_report(
        baseline_path=args.baseline,
        current_path=args.current,
        metric=args.metric,
        baseline=baseline_mean,
        current=current_mean,
        status=status,
        n_baseline=n_base,
        n_current=n_cur,
    )
    print(report)
    if status == "FAIL" or (status == "WARN" and args.fail_on_warn):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
