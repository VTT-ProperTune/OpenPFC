#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Step-3 gate: kernel-bound vs halo-bound from ALCU_PERF / ALCU_VERIFY logs.

Usage:
  python3 apps/alloy_pf_directional/scripts/analyze_2d_scaling.py [log_dir ...]
  python3 apps/alloy_pf_directional/scripts/analyze_2d_scaling.py results/alloy_pf_directional_nz1_check
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

KV = re.compile(r"(\w+)=(\S+)")
PERF = re.compile(r"ALCU_PERF\s+(.*)")
VERIFY = re.compile(r"ALCU_VERIFY\s+(.*)")
SCALE = re.compile(r"ALCU_SCALE\s+(.*)")


def parse_kv(s: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV.finditer(s)}


def fget(d: dict[str, str], *keys: str, default: float | None = None) -> float | None:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except ValueError:
                continue
    return default


def collect(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    files: list[Path] = []
    for p in paths:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(p.rglob("*.log"))
            files.extend(p.rglob("*.out"))
    seen: set[Path] = set()
    for fp in files:
        rp = fp.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        try:
            text = fp.read_text(errors="replace")
        except OSError:
            continue
        scale: dict[str, str] = {}
        for m in SCALE.finditer(text):
            scale = parse_kv(m.group(1))
        for m in PERF.finditer(text):
            d = parse_kv(m.group(1))
            d["_file"] = str(fp)
            d["_kind"] = "perf"
            for k, v in scale.items():
                d.setdefault(k, v)
            rows.append(d)
        if not any(r.get("_file") == str(fp) and r.get("_kind") == "perf" for r in rows):
            for m in VERIFY.finditer(text):
                d = parse_kv(m.group(1))
                d["_file"] = str(fp)
                d["_kind"] = "verify"
                for k, v in scale.items():
                    d.setdefault(k, v)
                rows.append(d)
    return rows


def resources(d: dict) -> int:
    be = (d.get("backend") or "").lower()
    if be in ("openmp", "omp"):
        v = fget(d, "nthreads", "nthreads")
        if v is not None:
            return int(v)
    for k in ("nproc", "nproc", "nthreads", "nthreads"):
        v = fget(d, k)
        if v is not None:
            return int(v)
    return 1


def halo_pct(d: dict) -> float | None:
    v = fget(d, "halo_pct", "ghost_pct")
    if v is not None:
        return v
    halo = fget(d, "halo_s", "ghost_s")
    kern = fget(d, "kernel_s")
    if halo is not None and kern is not None and halo + kern > 0:
        return 100.0 * halo / (halo + kern)
    return None


def solute_pct(d: dict) -> float | None:
    v = fget(d, "solute_pct")
    if v is not None and v >= 0:
        return v
    return None


def time_per_step_ms(d: dict) -> float | None:
    t = fget(d, "time_per_step_s")
    return None if t is None else 1000.0 * t


def backend_of(d: dict) -> str:
    b = d.get("backend") or d.get("mode") or ""
    path = d.get("_file", "").lower()
    if b:
        return b
    if "hip" in path or "gpu" in path:
        return "hip"
    if "mpi" in path:
        return "mpi"
    if "openmp" in path or "omp" in path:
        return "openmp"
    return d.get("_kind", "?")


def verdict(rows: list[dict]) -> str:
    gpu = [r for r in rows if backend_of(r) in ("hip", "gpu")]
    cpu = [r for r in rows if backend_of(r) in ("openmp", "mpi", "cpu")]
    lines = []
    if gpu:
        by_n = sorted(gpu, key=resources)
        h8 = [halo_pct(r) for r in by_n if resources(r) >= 8]
        h8 = [h for h in h8 if h is not None]
        h12 = [halo_pct(r) for r in by_n if 8 <= resources(r) <= 16]
        h12 = [h for h in h12 if h is not None]
        if h8 and max(h8) >= 40.0:
            lines.append(
                "VERDICT: halo-bound at ≥8 GCDs "
                f"(halo_pct up to {max(h8):.1f}%). Fix device halo before 3D."
            )
        elif h12 and max(h12) < 25.0:
            lines.append(
                "VERDICT: kernel-bound at 1–2 GPU nodes "
                f"(halo_pct ≤ {max(h12):.1f}%). Proceed to 3D GPU bricks."
            )
        elif h12:
            lines.append(
                "VERDICT: mixed at 8–16 GCDs "
                f"(halo_pct {min(h12):.1f}–{max(h12):.1f}%). "
                "Re-check 2-node GPU before 3D; do not grow 3D until halo < ~35%."
            )
        else:
            lines.append(
                "VERDICT: GPU logs present but halo/kernel split missing. "
                "Waiting on LUMI-G ALCU_PERF lines."
            )
    else:
        lines.append(
            "VERDICT: GPU pending (no HIP ALCU_PERF). "
            "Do not start 3D production bricks until LUMI-G 1–2 node numbers exist."
        )
        lines.append(
            "How to interpret later: halo_pct ≥ 40% at 8 GCDs → halo-bound "
            "(fix FullPaddedDeviceHalo / packed fallback). "
            "halo_pct < 25% at 8–16 GCDs → kernel-bound (proceed to 3D GPU)."
        )
    if cpu:
        hs = [halo_pct(r) for r in cpu]
        hs = [h for h in hs if h is not None]
        ss = [solute_pct(r) for r in cpu]
        ss = [s for s in ss if s is not None]
        extra = []
        if ss:
            extra.append(f"CPU solute_pct {min(ss):.1f}–{max(ss):.1f}%")
        if hs:
            extra.append(f"CPU halo/ghost_pct {min(hs):.1f}–{max(hs):.1f}%")
        if extra:
            lines.append("Local/CPU context: " + "; ".join(extra) + ".")
    return "\n".join(lines)


def table(rows: list[dict]) -> str:
    hdr = (
        f"{'backend':<10} {'n':>5} {'ms/step':>10} {'halo%':>8} {'eu%':>8} "
        f"{'grain%':>8} {'solute%':>8}  file"
    )
    out = [hdr, "-" * len(hdr)]
    for r in sorted(rows, key=lambda d: (backend_of(d), resources(d))):
        ms = time_per_step_ms(r)
        hp = halo_pct(r)
        sp = solute_pct(r)
        eu = fget(r, "eu_pct")
        gp = fget(r, "grain_pct")
        out.append(
            f"{backend_of(r):<10} {resources(r):5d} "
            f"{(f'{ms:.3f}' if ms is not None else '—'):>10} "
            f"{(f'{hp:.1f}' if hp is not None else '—'):>8} "
            f"{(f'{eu:.1f}' if eu is not None else '—'):>8} "
            f"{(f'{gp:.1f}' if gp is not None else '—'):>8} "
            f"{(f'{sp:.1f}' if sp is not None else '—'):>8}  "
            f"{r.get('_file','')}"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["results/alloy_pf_directional_nz1_check"])
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()
    roots = [Path(p) for p in args.paths]
    rows = collect(roots)
    body = []
    body.append("Al-Cu FTA 2D scaling — step 3 gate")
    body.append("")
    if not rows:
        body.append("No ALCU_PERF / ALCU_VERIFY lines found.")
        body.append(verdict([]))
    else:
        body.append(table(rows))
        body.append("")
        body.append(verdict(rows))
    text = "\n".join(body) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text)
    else:
        for r in roots:
            if r.is_dir():
                (r / "STEP3_VERDICT.md").write_text(text)
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
