#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Summarize kobayashi scaling logs and check deterministic global statistics."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


verify_line_re = re.compile(r"^KOBAYASHI_VERIFY\b(.*)$")
hex_line_re = re.compile(r"^KOBAYASHI_VERIFY_HEX\b(.*)$")


def parse_kv_blob(blob: str) -> Dict[str, str]:
    """Parse ' key=value key=value' where values may be scientific floats."""
    out: Dict[str, str] = {}
    tokens = blob.strip().split()
    for tok in tokens:
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        out[key] = val
    return out


@dataclass
class RunRow:
    path: str
    nproc: int
    wall: float
    kv: Dict[str, str]
    hex_kv: Dict[str, str]


def scan_directory(root: str) -> List[RunRow]:
    paths = sorted(glob.glob(os.path.join(root, "run_np_*.log")))
    rows: List[RunRow] = []
    for path in paths:
        base = os.path.basename(path)
        m = re.match(r"run_np_(\d+)\.log\Z", base)
        if not m:
            continue
        stated_np = int(m.group(1))
        verify_lines: List[str] = []
        hex_line: Optional[str] = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if verify_line_re.match(line):
                    verify_lines.append(line.strip())
                if hex_line_re.match(line):
                    hex_line = line.strip()

        kv_line: Optional[str] = None
        for cand in reversed(verify_lines):
            vm_c = verify_line_re.match(cand)
            if not vm_c:
                continue
            kv_c = parse_kv_blob(vm_c.group(1))
            try:
                lnproc = int(kv_c.get("nproc", "-1"))
            except ValueError:
                lnproc = -1
            if lnproc == stated_np:
                kv_line = cand
                break
        if kv_line is None and verify_lines:
            kv_line = verify_lines[-1]

        if not kv_line:
            print(f"warn: no KOBAYASHI_VERIFY in {path}", file=sys.stderr)
            continue
        vm = verify_line_re.match(kv_line)
        hm = hex_line_re.match(hex_line) if hex_line else None
        assert vm is not None
        kv = parse_kv_blob(vm.group(1))
        hex_kv = parse_kv_blob(hm.group(1)) if hm else {}
        wall = float(kv.get("wall_loop_max_s", "nan"))
        nproc = int(kv.get("nproc", "-1"))
        if nproc != stated_np:
            print(
                f"warn: filename np={stated_np} vs line nproc={nproc} ({path}) — "
                "MPI_COMM_WORLD size does not match srun -n; fix PMI or rebuild against the "
                "same libmpi as the loaded Open MPI module.",
                file=sys.stderr,
            )
        rows.append(RunRow(path=path, nproc=nproc, wall=wall, kv=kv, hex_kv=hex_kv))
    rows.sort(key=lambda r: r.nproc)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "directory",
        help="kobayashi_scaling_<jobid> directory containing run_np_*.log",
    )
    args = ap.parse_args()

    rows = scan_directory(args.directory)
    if not rows:
        print("no runs parsed", file=sys.stderr)
        return 1

    ref = rows[0]
    ok_hex = True

    print(
        "nproc\twall_loop_max_s\tspeedup\tefficiency_%\thex_match\t"
        "sum_phi(dec)\tl2_phi(dec)"
    )
    t1 = ref.wall
    for r in rows:
        spd = t1 / r.wall if r.wall > 0 else float("nan")
        eff = 100.0 * spd / r.nproc if r.nproc > 0 else float("nan")
        if ref.hex_kv and r.hex_kv:
            hex_ok = r.hex_kv == ref.hex_kv
            if not hex_ok:
                ok_hex = False
        else:
            hex_ok = "n/a"
        print(
            f"{r.nproc}\t{r.wall:.9g}\t{spd:.9g}\t{eff:.9g}\t"
            f"{hex_ok}\t{r.kv.get('sum_phi','')}\t{r.kv.get('l2_phi','')}"
        )

    print("\n--- bitwise reference (first run, typically nproc=1) ---")
    print(f"log: {ref.path}")
    for k in sorted(ref.hex_kv.keys()):
        print(f"  {k}={ref.hex_kv[k]}")

    print("\n--- mismatched HEX lines (should be empty if MPI decomposition is deterministic) ---")
    mismatches = 0
    for r in rows:
        if not ref.hex_kv or not r.hex_kv:
            continue
        if r.hex_kv != ref.hex_kv:
            mismatches += 1
            print(f"nproc={r.nproc} log={r.path}")
            for k in sorted(set(ref.hex_kv) | set(r.hex_kv)):
                a = ref.hex_kv.get(k, "<missing>")
                b = r.hex_kv.get(k, "<missing>")
                if a != b:
                    print(f"  {k}: ref {a} != run {b}")
    if mismatches == 0:
        print("(none)")

    return 0 if ok_hex else 2


if __name__ == "__main__":
    raise SystemExit(main())
