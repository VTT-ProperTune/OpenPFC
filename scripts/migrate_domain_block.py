#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rewrite a legacy ``domain`` block as ``size`` / ``lower`` / ``upper``.

OpenPFC inputs used to give the grid as cell counts ``Lx, Ly, Lz``, spacings
``dx, dy, dz`` and ``"origin": "center" | "corner"`` (legacy ``origo``). The
domain is now the box itself::

    "domain": {"size": [256, 256, 256],
               "lower": [-142.2, -142.2, -142.2],
               "upper": [142.2, 142.2, 142.2]}

and the spacing is derived, ``(upper - lower) / size`` on a periodic axis.

The rewrite is exact whenever floating point allows it: ``lower`` is the old
origin (``-0.5 * dx * Lx`` for "center", 0 for "corner"), and ``upper`` is
chosen so that the derived spacing equals the old ``dx`` bit for bit, which
leaves the simulation, its checksums and its checkpoints unchanged. For a few
``(origin, Lx, dx)`` no such ``upper`` exists; the script then takes the
closest spacing and prints a warning.

Only the domain block is touched; the rest of the file keeps its formatting.
A block with only ``Lx/Ly/Lz/dx`` (the old thin-film style) uses ``dx`` on
every axis and a corner origin, as those drivers did.

Usage::

    scripts/migrate_domain_block.py FILE [FILE ...]      # rewrite in place
    scripts/migrate_domain_block.py --dry-run FILE ...   # print, do not write
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from pathlib import Path

LEGACY_KEYS = ("Lx", "Ly", "Lz", "dx", "dy", "dz", "origin", "origo")


class MigrationError(ValueError):
    """The block cannot be rewritten without guessing."""


def exact_limits(n: int, d: float, lower: float) -> tuple[float, float]:
    """Return ``(lower, upper)`` with ``(upper - lower) / n == d`` exactly."""
    candidates = []
    if lower < 0.0:
        candidates.append(-lower)  # a symmetric box reads best
    candidates.append(lower + n * d)
    for start in list(candidates):
        lo = hi = start
        for _ in range(64):
            candidates.extend((lo, hi))
            lo, hi = math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)
    for upper in candidates:
        if upper > lower and (upper - lower) / n == d:
            return lower, upper
    # Some (lower, n, d) have no exact solution: the division skips d.
    # Take the closest spacing and say so; it differs from d by round-off.
    upper = min((u for u in candidates if u > lower),
                key=lambda u: abs((u - lower) / n - d))
    print(f"warning: spacing {d!r} with {n} cells from {lower!r} is reproduced "
          f"only to {(upper - lower) / n!r}", file=sys.stderr)
    return lower, upper


def convert(old: dict) -> dict:
    """Map legacy domain keys to ``size``, ``lower`` and ``upper``."""
    if "size" in old or "lower" in old or "upper" in old:
        raise MigrationError("block already has size/lower/upper")
    try:
        size = [int(old["Lx"]), int(old.get("Ly", 1)), int(old.get("Lz", 1))]
        dx = float(old["dx"])
    except KeyError as err:
        raise MigrationError(f"missing {err.args[0]}") from None
    spacing = [dx, float(old.get("dy", dx)), float(old.get("dz", dx))]
    origin = old.get("origin", old.get("origo", "corner"))
    if origin not in ("center", "corner"):
        raise MigrationError(f"unknown origin {origin!r}")
    lower, upper = [], []
    for n, d in zip(size, spacing):
        # Same expression the old parser evaluated: -0.5 * dx * Lx.
        x0 = -0.5 * d * n if origin == "center" else 0.0
        lo, hi = exact_limits(n, d, x0)
        lower.append(lo)
        upper.append(hi)
    return {"size": size, "lower": lower, "upper": upper}


def _fmt(value) -> str:
    return json.dumps(value)


def _object_span(text: str, open_brace: int) -> int:
    """Index one past the brace that closes the object opened at ``open_brace``."""
    depth, in_str, esc = 0, False, False
    for i in range(open_brace, len(text)):
        c = text[i]
        if in_str:
            esc = c == "\\" and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise MigrationError("unbalanced braces in domain block")


def migrate_json(text: str) -> str:
    data = json.loads(text)
    if not isinstance(data.get("domain"), dict):
        if any(k in data for k in LEGACY_KEYS):
            raise MigrationError("flat domain keys at the root; move them under 'domain'")
        raise MigrationError("no domain block")
    old = data["domain"]
    new = convert(old)
    rest = {k: v for k, v in old.items() if k not in LEGACY_KEYS}

    m = re.search(r'"domain"\s*:\s*\{', text)
    start = m.end() - 1
    end = _object_span(text, start)
    line_start = text.rfind("\n", 0, m.start()) + 1
    outer = text[line_start : m.start()]
    inner_m = re.search(r"\n([ \t]+)\"", text[start:end])
    inner = inner_m.group(1) if inner_m else outer + "    "
    items = [f'{inner}"{k}": {_fmt(v)}' for k, v in {**new, **rest}.items()]
    block = "{\n" + ",\n".join(items) + "\n" + outer + "}"
    out = text[:start] + block + text[end:]
    if json.loads(out)["domain"] != {**new, **rest}:
        raise MigrationError("rewritten JSON does not read back")
    return out


def migrate_toml(text: str) -> str:
    data = tomllib.loads(text)
    if not isinstance(data.get("domain"), dict):
        raise MigrationError("no [domain] table")
    old = data["domain"]
    new = convert(old)
    lines = text.splitlines(keepends=True)
    try:
        head = next(i for i, l in enumerate(lines) if re.match(r"\s*\[domain\]\s*(#.*)?$", l))
    except StopIteration:
        raise MigrationError("domain is not a [domain] table") from None
    key_re = re.compile(r"\s*(" + "|".join(LEGACY_KEYS) + r")\s*=")
    out, inserted = lines[: head + 1], False
    i = head + 1
    while i < len(lines) and not re.match(r"\s*\[", lines[i]):
        if key_re.match(lines[i]):
            if not inserted:
                out += [f"{k} = {_fmt(v)}\n" for k, v in new.items()]
                inserted = True
        else:
            out.append(lines[i])
        i += 1
    out += lines[i:]
    result = "".join(out)
    if tomllib.loads(result)["domain"] != {
        **new,
        **{k: v for k, v in old.items() if k not in LEGACY_KEYS},
    }:
        raise MigrationError("rewritten TOML does not read back")
    return result


def migrate(path: Path) -> str:
    text = path.read_text()
    if path.suffix == ".toml":
        return migrate_toml(text)
    return migrate_json(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)
    status = 0
    for path in args.files:
        try:
            out = migrate(path)
        except (MigrationError, json.JSONDecodeError, tomllib.TOMLDecodeError) as err:
            print(f"{path}: {err}", file=sys.stderr)
            status = 1
            continue
        if args.dry_run:
            sys.stdout.write(out)
        else:
            path.write_text(out)
            print(f"{path}: migrated")
    return status


if __name__ == "__main__":
    sys.exit(main())
