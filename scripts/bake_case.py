#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bake a compiled CPU tungsten case into an Apptainer image using host MPI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
GLIBC = re.compile(r"lib(?:c|m|dl|pthread|rt|util|resolv|anl|nss_[\w]+)\.so(?:\.|$)")
MPI_STACK = re.compile(r"lib(?:mpi[^/]*|open-pal|pmix|ucp|uct|ucs|ucm)\.so(?:\.|$)")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def under(path, prefix):
    return path == prefix or prefix in path.parents


def parse_ldd(output):
    if "not found" in output:
        raise ValueError("unresolved runtime library; load the build's compiler/MPI modules first")
    libraries = {}
    for line in output.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(.+?)\s+\(0x[0-9a-f]+\)", line)
        if match:
            name, path = match.groups()
            if "/" in name or not path.startswith("/"):
                raise ValueError("unsupported ldd entry: " + line)
            libraries[name] = Path(path).resolve()
    if not any(name.startswith("libmpi.so") for name in libraries):
        raise ValueError("expected a dynamically linked MPI application")
    return libraries


def validate_input(settings):
    if "restart_from" in settings:
        raise ValueError("baking restart bundles is not supported yet; use a self-contained initial case")
    for initial in settings.get("initial_conditions", []):
        if initial.get("type") not in ("constant", "single_seed", "seed_grid"):
            raise ValueError("bake supports constant, single_seed, and seed_grid initial conditions only")
    paths = [field["data"] for field in settings.get("fields", [])]
    if settings.get("checkpoint", {}).get("directory"):
        paths.append(settings["checkpoint"]["directory"])
    for value in paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "results":
            raise ValueError("baked output paths must be relative paths under results/")


def prepare(args):
    case = Path(args.case).resolve()
    manifest = json.loads((case / "openpfc.json").read_text())
    if (manifest.get("format_version") != 1 or manifest.get("app") != "tungsten"
            or manifest.get("profile") != "local" or manifest.get("input") != "input.json"):
        raise ValueError("first runtime recipe requires a compiled local CPU tungsten case")
    validate_input(json.loads((case / "input.json").read_text()))
    executable = Path(manifest["executable"]).resolve()
    if not executable.is_file():
        raise ValueError("compiled executable is missing")
    base = Path(args.base).resolve()
    if not base.is_file():
        raise ValueError("--base must be a local Python 3.8+ Apptainer SIF image")
    prefixes = [Path(p).resolve() for p in args.host_prefix]
    # Site MPI may embed its original symlink spelling in plugin/help paths.
    bindings = sorted(set(prefixes + [Path(p).absolute() for p in args.host_prefix]))
    for prefix in prefixes:
        if not prefix.is_dir() or prefix == Path("/") or any(c in str(prefix) for c in ",:\n"):
            raise ValueError("host prefix must be a directory without commas, colons, or newlines")
    libraries = parse_ldd(subprocess.check_output(["ldd", str(executable)], text=True))
    bundled, external, system = {}, {}, {}
    for name, path in libraries.items():
        if any(under(path, prefix) for prefix in prefixes):
            external[name] = path
        elif MPI_STACK.match(name):
            raise ValueError("MPI/PMIx/UCX library needs --host-prefix: " + str(path))
        elif GLIBC.match(name):
            system[name] = path
        else:
            bundled[name] = path
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    for directory in ("bin", "lib", "case", "share/openpfc"):
        (payload / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(executable, payload / "bin/tungsten")
    shutil.copy2(ROOT / "scripts/openpfc", payload / "bin/openpfc")
    shutil.copy2(ROOT / "containers/runtime/baked_case.py", payload / "bin/baked-case")
    (payload / "bin/baked-case").chmod(0o755)
    build = Path(manifest["build_dir"])
    shutil.copy2(build / "share/openpfc/version", payload / "share/openpfc/version")
    shutil.copy2(case / "input.json", payload / "case/input.json")
    (payload / "case/openpfc.json").write_text(json.dumps({
        "format_version": 1, "app": "tungsten", "input": "input.json",
        "profile": "local", "executable": "/opt/openpfc/bin/tungsten",
    }, indent=2) + "\n")
    (payload / "case/README.md").write_text(
        "# Baked tungsten case\n\n"
        "This input is copied from the baked image. Run with the supplied host-MPI\n"
        "wrapper; do not compile. Output paths are relative to this case directory.\n"
        "Use a fresh case/output directory for each run.\n")
    for name, path in bundled.items():
        shutil.copy2(path, payload / "lib" / name)
    def records(items):
        return {name: {"path": str(path), "sha256": digest(path)}
                for name, path in sorted(items.items())}
    source = Path(manifest.get("source", ROOT))
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              text=True, capture_output=True)
    status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"],
                            text=True, capture_output=True)
    provenance = {
        "format_version": 1, "profile": "local", "mpi_mode": "host-bind",
        "base_sha256": digest(base), "binary_sha256": digest(executable),
        "input_sha256": digest(case / "input.json"),
        "openpfc_version": (payload / "share/openpfc/version").read_text().strip(),
        "packaging_checkout_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "packaging_checkout_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "source_identity_note": "Checkout at packaging time; not proof of binary build provenance.",
        "bundled_libraries": records(bundled), "host_libraries": records(external),
        "base_provided_glibc": records(system),
        "host_prefixes": [str(p) for p in prefixes],
        "host_bindings": [str(p) for p in bindings],
    }
    (payload / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Relative %files input avoids shell/definition interpolation of case paths.
    # Copy the base reference to a whitespace-free relative symlink as well.
    (output / "base.sif").symlink_to(base)
    (output / "runtime.def").write_text(
        "Bootstrap: localimage\nFrom: base.sif\n\n"
        "%files\n    payload /opt/openpfc\n\n"
        "%environment\n    export PATH=/opt/openpfc/bin:$PATH\n"
        "    export LD_LIBRARY_PATH=/opt/openpfc/lib:${LD_LIBRARY_PATH:-}\n\n"
        "%runscript\n    exec /opt/openpfc/bin/baked-case \"$@\"\n")
    library_path = ":".join(["/opt/openpfc/lib", *sorted({str(p.parent) for p in external.values()})])
    options = []
    for prefix in bindings:
        options += ["--bind", str(prefix) + ":" + str(prefix) + ":ro"]
    options += ["--env", "LD_LIBRARY_PATH=" + library_path]
    (output / "run-host.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        'bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        "exec apptainer run " + " ".join(shlex.quote(arg) for arg in options) +
        ' "$bundle_dir/case.sif" "$@"\n')
    (output / "run-host.sh").chmod(0o755)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--base", required=True, help="local Python runtime SIF")
    parser.add_argument("--output", required=True, help="new bundle directory under builds/")
    parser.add_argument("--host-prefix", action="append", required=True,
                        help="MPI/PMIx/UCX install prefix to mount read-only; repeat as needed")
    parser.add_argument("--prepare-only", action="store_true", help="stage recipe without building SIF")
    args = parser.parse_args(argv)
    try:
        output = prepare(args)
        if not args.prepare_only:
            subprocess.run(["apptainer", "build", "--disable-cache", "case.sif", "runtime.def"],
                           cwd=str(output), check=True)
        print("Runtime bundle: " + str(output))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("bake_case: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
