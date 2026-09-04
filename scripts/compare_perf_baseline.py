#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare an OpenPFC profiling JSON export against a stored baseline.

Reads schema v2/v3 full traces from ``ProfilingSession::finalize_and_export``,
or schema v4 in-tree summaries (no per-frame arrays). The compared value is
the mean of a named frame scalar (default ``wall_step``) after an optional
warmup skip. Schema v4 already stores that mean; ``--warmup-frames`` then
applies only to a v2/v3 *current* file.

Thresholds (regression = current slower than baseline):

* pass:  regression <= 5%
* warn:  5% < regression <= 15%  (exit 0 unless ``--fail-on-warn``)
* fail:  regression > 15%        (exit 1)

Speedups always pass. Use ``--strict`` to treat warn as fail.

Write a summary from a full export::

    python3 scripts/compare_perf_baseline.py --summarize full.json \\
        --warmup-frames=1 -o tests/baselines/perf/machine-tag.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WARN_RATIO = 0.05
FAIL_RATIO = 0.15
DEFAULT_METRIC = "wall_step"
SCHEMA_FULL = (2, 3)
SCHEMA_SUMMARY = 4


def is_summary(doc: Mapping[str, Any]) -> bool:
    return int(doc.get("schema_version", 0)) == SCHEMA_SUMMARY


def load_profiling_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    version = data.get("schema_version")
    if version in SCHEMA_FULL:
        if "ranks" not in data or "frame_metric_names" not in data:
            raise ValueError(f"{path}: missing ranks or frame_metric_names")
        return data
    if version == SCHEMA_SUMMARY:
        if "metrics" not in data or not isinstance(data["metrics"], dict):
            raise ValueError(f"{path}: schema 4 summary missing metrics")
        return data
    raise ValueError(
        f"{path}: unsupported schema_version {version!r} (need 2, 3, or 4)"
    )


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
    if is_summary(doc):
        raise ValueError("schema 4 summary has no per-frame scalars")
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


def _summary_metric(doc: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metrics = doc["metrics"]
    if name not in metrics:
        raise ValueError(
            f"metric {name!r} not in summary metrics {sorted(metrics)}"
        )
    entry = metrics[name]
    if not isinstance(entry, dict) or "mean" not in entry:
        raise ValueError(f"summary metrics[{name!r}] must contain mean")
    return entry


def mean_metric(doc: Mapping[str, Any], name: str, *, warmup_frames: int) -> float:
    if is_summary(doc):
        return float(_summary_metric(doc, name)["mean"])
    values = list(iter_scalar_values(doc, name, warmup_frames=warmup_frames))
    if not values:
        raise ValueError(
            f"no samples for metric {name!r} after warmup_frames={warmup_frames}"
        )
    return sum(values) / len(values)


def sample_count(doc: Mapping[str, Any], name: str, *, warmup_frames: int) -> int:
    if is_summary(doc):
        entry = _summary_metric(doc, name)
        if "n_samples" in entry:
            return int(entry["n_samples"])
        return int(doc.get("n_samples") or 0)
    return len(list(iter_scalar_values(doc, name, warmup_frames=warmup_frames)))


def _region_exclusive_means(
    doc: Mapping[str, Any], *, warmup_frames: int
) -> dict[str, float]:
    acc: dict[str, float] = {}
    n = 0
    for rank in doc["ranks"]:
        frames = rank.get("frames") or []
        for i, frame in enumerate(frames):
            if i < warmup_frames:
                continue
            n += 1
            regions = frame.get("regions") or {}
            for key, node in regions.items():
                if isinstance(node, dict) and "exclusive" in node:
                    acc[key] = acc.get(key, 0.0) + float(node["exclusive"])
    if n == 0:
        return {}
    return {k: acc[k] / n for k in sorted(acc)}


def summarize_doc(doc: Mapping[str, Any], *, warmup_frames: int) -> dict[str, Any]:
    """Collapse a v2/v3 trace to a schema-4 pin (no per-frame arrays)."""
    if is_summary(doc):
        raise ValueError("input is already a schema 4 summary")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be >= 0")
    names = list(doc["frame_metric_names"])
    metrics: dict[str, Any] = {}
    n_samples = 0
    for name in names:
        values = list(iter_scalar_values(doc, name, warmup_frames=warmup_frames))
        if not values:
            continue
        n_samples = len(values)
        metrics[name] = {
            "mean": sum(values) / len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    if not metrics:
        raise ValueError(
            f"no samples for any metric after warmup_frames={warmup_frames}"
        )
    frames_per_rank = [
        int(r.get("n_frames") or len(r.get("frames") or [])) for r in doc["ranks"]
    ]
    return {
        "schema_version": SCHEMA_SUMMARY,
        "kind": "perf_summary",
        "openpfc_version": doc.get("openpfc_version"),
        "n_mpi_ranks": doc["n_mpi_ranks"],
        "n_frames_per_rank": frames_per_rank[0] if frames_per_rank else 0,
        "total_frames": doc.get("total_frames", sum(frames_per_rank)),
        "warmup_frames": warmup_frames,
        "n_samples": n_samples,
        "metrics": metrics,
        "region_exclusive_mean": _region_exclusive_means(
            doc, warmup_frames=warmup_frames
        ),
    }


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
    p.add_argument(
        "baseline",
        type=Path,
        help="Stored pin (schema 4 summary or v2/v3 trace)",
    )
    p.add_argument(
        "current",
        nargs="?",
        type=Path,
        help="New profiling JSON to compare (omit with --summarize)",
    )
    p.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"frame metric to compare (default: {DEFAULT_METRIC})",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Skip this many leading frames per rank on v2/v3 files (default: 0)",
    )
    p.add_argument(
        "--summarize",
        action="store_true",
        help="Write a schema-4 summary of BASELINE (full v2/v3 trace) to --output",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination for --summarize (default: stdout)",
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
    if args.summarize:
        doc = load_profiling_json(args.baseline)
        summary = summarize_doc(doc, warmup_frames=args.warmup_frames)
        text = json.dumps(summary, indent=2) + "\n"
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    if args.current is None:
        print("error: current JSON is required unless --summarize", file=sys.stderr)
        return 2
    baseline_doc = load_profiling_json(args.baseline)
    current_doc = load_profiling_json(args.current)
    baseline_mean = mean_metric(
        baseline_doc, args.metric, warmup_frames=args.warmup_frames
    )
    current_mean = mean_metric(
        current_doc, args.metric, warmup_frames=args.warmup_frames
    )
    n_base = sample_count(
        baseline_doc, args.metric, warmup_frames=args.warmup_frames
    )
    n_cur = sample_count(
        current_doc, args.metric, warmup_frames=args.warmup_frames
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
