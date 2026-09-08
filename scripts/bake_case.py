#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bake a compiled CPU tungsten case into a container image using host MPI."""

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
GLIBC = re.compile(r"lib(?:c|m|mvec|dl|pthread|rt|util|resolv|anl|nss_[\w]+)\.so(?:\.|$)")

# Libraries that must come from the host, per site MPI stack. Baking any of
# these would put the site's interconnect inside the image, which defeats the
# host-MPI model and does not survive a move to another machine.
MPI_STACK = {
    # Open MPI / UCX (Tohtori).
    "openmpi": re.compile(r"lib(?:mpi[^/]*|open-pal|pmix|ucp|uct|ucs|ucm)\.so(?:\.|$)"),
    # Cray MPICH / libfabric / Slingshot (LUMI). libcxi and libxpmem live in
    # /usr/lib64 next to ordinary system libraries, so a prefix rule cannot
    # separate them: they need --host-lib.
    "cray": re.compile(
        r"lib(?:mpi[^/]*|fabric|cxi|pmi|pmi2|pals|xpmem|dsmml)\.so(?:\.|$)"),
}
MPI_MARKER = re.compile(r"libmpi[^/]*\.so(?:\.|$)")
SITE_PROFILES = {"openmpi": {"local"}, "cray": {"local", "lumi"}}
# Runtime state the site's process manager needs inside the container. Cray PMI
# reaches the Slurm daemon through this socket directory; without it MPI_Init
# fails with "job id unknown" and every rank believes it is rank 0.
SITE_BINDS = {"openmpi": (), "cray": ("/var/spool/slurmd",)}


def glibc_requirement(paths):
    """Highest GLIBC_x.y symbol version the host-provided libraries need.

    The bind model loads these host libraries inside the image, so the base
    image's glibc must be at least this new. Getting this wrong fails at run
    time with an unhelpful symbol error, so it is recorded in provenance and
    checked against the base image when a container runtime is available.
    """
    best = None
    for path in paths:
        try:
            out = subprocess.run(["objdump", "-T", str(path)], text=True,
                                 capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        for match in re.finditer(r"GLIBC_(\d+)\.(\d+)", out):
            version = (int(match.group(1)), int(match.group(2)))
            if best is None or version > best:
                best = version
    return best


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
    if not any(MPI_MARKER.match(name) for name in libraries):
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
    profile = manifest.get("profile")
    if (manifest.get("format_version") != 1 or manifest.get("app") != "tungsten"
            or profile not in SITE_PROFILES[args.site] or manifest.get("input") != "input.json"):
        raise ValueError("runtime recipe requires a compiled tungsten case with profile "
                         + "/".join(sorted(SITE_PROFILES[args.site])))
    validate_input(json.loads((case / "input.json").read_text()))
    executable = Path(manifest["executable"]).resolve()
    if not executable.is_file():
        raise ValueError("compiled executable is missing")
    base = Path(args.base).resolve()
    if not base.is_file():
        raise ValueError("--base must be a local Python 3.8+ Apptainer SIF image")
    host_libs = {}
    for value in args.host_lib:
        given = Path(value)
        path = given.resolve()
        if not path.is_file():
            raise ValueError("--host-lib is not a file: " + value)
        if any(c in str(path) for c in ",:\n"):
            raise ValueError("--host-lib path must not contain commas, colons, or newlines")
        # Key on the spelling given, which is the soname ldd reports
        # (libxpmem.so.0), not the versioned file it resolves to
        # (libxpmem.so.0.0.0).
        host_libs[given.name] = path
    prefixes = [Path(p).resolve() for p in args.host_prefix]
    # Site MPI may embed its original symlink spelling in plugin/help paths.
    bindings = sorted(set(prefixes + [Path(p).absolute() for p in args.host_prefix]))
    for prefix in prefixes:
        if not prefix.is_dir() or prefix == Path("/") or any(c in str(prefix) for c in ",:\n"):
            raise ValueError("host prefix must be a directory without commas, colons, or newlines")
    libraries = parse_ldd(subprocess.check_output(["ldd", str(executable)], text=True))
    mpi_stack = MPI_STACK[args.site]
    bundled, external, host_files, system = {}, {}, {}, {}
    for name, path in libraries.items():
        if name in host_libs:
            if host_libs[name] != path:
                raise ValueError("--host-lib does not match the resolved library: " + name)
            host_files[name] = path
        elif any(under(path, prefix) for prefix in prefixes):
            external[name] = path
        elif mpi_stack.match(name):
            raise ValueError("host MPI stack library needs --host-prefix or --host-lib: "
                             + str(path))
        elif GLIBC.match(name):
            system[name] = path
        else:
            bundled[name] = path
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    for directory in ("bin", "lib", "hostlib", "case", "share/openpfc"):
        (payload / directory).mkdir(parents=True, exist_ok=True)
    # Bind destinations must already exist in the image.
    (payload / "hostlib/.keep").write_text("")
    shutil.copy2(executable, payload / "bin/tungsten")
    shutil.copy2(ROOT / "scripts/openpfc", payload / "bin/openpfc")
    shutil.copy2(ROOT / "containers/runtime/baked_case.py", payload / "bin/baked-case")
    (payload / "bin/baked-case").chmod(0o755)
    build = Path(manifest["build_dir"])
    shutil.copy2(build / "share/openpfc/version", payload / "share/openpfc/version")
    shutil.copy2(case / "input.json", payload / "case/input.json")
    (payload / "case/openpfc.json").write_text(json.dumps({
        "format_version": 1, "app": "tungsten", "input": "input.json",
        "profile": profile, "executable": "/opt/openpfc/bin/tungsten",
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
    glibc_floor = glibc_requirement(sorted(set(external.values()) | set(host_files.values())))
    provenance = {
        "format_version": 1, "profile": profile, "mpi_mode": "host-bind",
        "base_sha256": digest(base), "binary_sha256": digest(executable),
        "input_sha256": digest(case / "input.json"),
        "openpfc_version": (payload / "share/openpfc/version").read_text().strip(),
        "packaging_checkout_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "packaging_checkout_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "source_identity_note": "Checkout at packaging time; not proof of binary build provenance.",
        "bundled_libraries": records(bundled),
        "host_libraries": records({**external, **host_files}),
        "host_glibc_requirement": (".".join(map(str, glibc_floor)) if glibc_floor else None),
        "site": args.site,
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
    hostlib_dir = "/opt/openpfc/hostlib"
    library_path = ":".join(["/opt/openpfc/lib",
                             *([hostlib_dir] if host_files else []),
                             *sorted({str(p.parent) for p in external.values()})])
    options = []
    for prefix in bindings:
        options += ["--bind", str(prefix) + ":" + str(prefix) + ":ro"]
    # Site process-manager state; present on the target, not necessarily here.
    for path in SITE_BINDS[args.site]:
        options += ["--bind", path + ":" + path]
    # Individually bound host libraries: their directory also holds ordinary
    # system libraries, so the whole directory must not be mounted over.
    for name, path in sorted(host_files.items()):
        options += ["--bind", str(path) + ":" + hostlib_dir + "/" + name + ":ro"]
    options += ["--env", "LD_LIBRARY_PATH=" + library_path]
    (output / "run-host.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        'bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        "exec " + shlex.quote(args.runtime) + " run " +
        " ".join(shlex.quote(arg) for arg in options) +
        ' "$bundle_dir/case.sif" "$@"\n')
    (output / "run-host.sh").chmod(0o755)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--base", required=True, help="local Python runtime SIF")
    parser.add_argument("--output", required=True, help="new bundle directory under builds/")
    parser.add_argument("--host-prefix", action="append", required=True,
                        help="host MPI stack install prefix to mount read-only; repeat as needed")
    parser.add_argument("--host-lib", action="append", default=[],
                        help="single host library file to bind (for stack libraries that "
                             "share a directory with ordinary system libraries, such as "
                             "LUMI's /usr/lib64/libcxi.so.1); repeat as needed")
    parser.add_argument("--site", choices=sorted(MPI_STACK), default="openmpi",
                        help="host MPI stack: openmpi (Tohtori) or cray (LUMI)")
    parser.add_argument("--runtime", choices=("apptainer", "singularity"), default="apptainer",
                        help="container runtime to build with and to call from run-host.sh; "
                             "LUMI ships singularity-ce, Tohtori ships apptainer")
    parser.add_argument("--prepare-only", action="store_true", help="stage recipe without building SIF")
    args = parser.parse_args(argv)
    try:
        output = prepare(args)
        if not args.prepare_only:
            subprocess.run([args.runtime, "build", "--disable-cache", "case.sif", "runtime.def"],
                           cwd=str(output), check=True)
        print("Runtime bundle: " + str(output))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("bake_case: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
