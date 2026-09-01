#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot instantaneous 1D k(V) from isothermal quenches vs k^PF(V)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgm_delta import KE, k_pf  # noqa: E402


def smooth_v(t: np.ndarray, x: np.ndarray, n: int = 5) -> np.ndarray:
    v = np.full_like(t, np.nan)
    for i in range(len(t)):
        i0 = max(0, i - n)
        i1 = min(len(t), i + n + 1)
        if i1 - i0 < 4:
            continue
        v[i] = np.polyfit(t[i0:i1], x[i0:i1], 1)[0]
    return v


def load_hist(path: Path) -> dict:
    h = np.loadtxt(path, comments="#")
    out = {"t": h[:, 0], "x": h[:, 2]}
    if h.shape[1] >= 10:
        out["k"] = h[:, 7]
        out["cs"] = h[:, 8]
        out["cl"] = h[:, 9]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="iso run directories with history.tsv")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Vs = np.linspace(0.0, 1.6, 400)
    ax.plot([k_pf(v) for v in Vs], Vs, "k--", lw=1.4, label=r"$k^{\mathrm{PF}}(V)$")
    ax.axvline(KE, color="0.5", lw=0.8, ls=":", label=rf"$k_e={KE:g}$")

    for d in args.dirs:
        p = Path(d)
        hist = p / "history.tsv"
        if not hist.exists():
            print(f"skip missing {hist}", file=sys.stderr)
            continue
        data = load_hist(hist)
        if "k" not in data:
            print(f"skip {hist}: no k_part column", file=sys.stderr)
            continue
        v = smooth_v(data["t"], data["x"])
        k = data["k"]
        good = np.isfinite(k) & np.isfinite(v) & (v > 0.005) & (k > 0.05) & (k < 0.99)
        ax.plot(k[good], v[good], lw=1.5, label=p.name)

    ax.set_xlabel(r"$k=c_s/c_l^{\mathrm{spike}}$")
    ax.set_ylabel(r"$V$ (m/s)")
    ax.set_xlim(0.12, 0.55)
    ax.set_ylim(0.0, 0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("1D isothermal instantaneous partition vs trapping law")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
