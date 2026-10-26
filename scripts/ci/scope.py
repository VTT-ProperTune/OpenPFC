#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Select PR components; reserve the complete routine suite for integration."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


def classify(paths, event):
    plan = dict(full=event != "pull_request", shared=False, cli=False,
                examples=False, apps=[], python=False, reasons=[])
    apps = set()
    for name in paths:
        path = PurePosixPath(name)
        parts = path.parts
        if (name.startswith("docs/") or path.suffix in {".md", ".rst", ".qmd"}
                or name.startswith(("LICENSE", "REUSE", ".reuse/"))
                or path.suffix == ".license"):
            continue
        if name.startswith(("tools/openpfc/", "tests/cli/")):
            plan["cli"] = True
        elif len(parts) > 2 and parts[0] == "apps":
            apps.add(parts[1])
        elif name.startswith("examples/"):
            plan["examples"] = True
        elif name.startswith("scripts/ci/") or name == "scripts/tests/test_ci_scope.py":
            plan["shared"] = True
            plan["reasons"].append("CI selection/control: " + name)
        elif name.startswith("scripts/") and path.suffix == ".py":
            plan["python"] = True
            # Python packaging tests exercise the installed native CLI too.
            if "bake_case" in name:
                plan["cli"] = True
        else:
            plan["shared"] = True
            plan["reasons"].append("shared or unclassified: " + name)
    plan["apps"] = sorted(apps)
    if plan["cli"] and (apps or plan["examples"]):
        # Installed-CLI checks install the configured applications/examples.
        plan["shared"] = True
        plan["reasons"].append("installed CLI with application/example changes")
    if event == "pull_request" and not paths:
        plan["shared"] = True
        plan["reasons"].append("empty change list: conservative fallback")
    plan["cpp"] = bool(plan["full"] or plan["shared"] or plan["cli"]
                       or plan["examples"] or apps)
    plan["python"] |= plan["full"] or plan["shared"]
    return plan


def required_jobs(plan):
    jobs = {"scope", "code-quality"}
    if plan["cpp"]:
        jobs.add("build-and-test")
    if plan["python"]:
        jobs.add("python-tests")
    if plan["full"]:
        jobs.update({"debug-and-mpi", "clang-compile", "packaging-smoke",
                     "cuda-compile", "hip-compile"})
    return jobs


def check_status(plan, results):
    required = required_jobs(plan)
    failures = []
    for job in required | set(results):
        result = results.get(job, {}).get("result", "missing")
        if result == "success" or (job not in required and result == "skipped"):
            continue
        failures.append(f"{job}: {result}")
    if failures:
        raise RuntimeError("CI failed:\n  " + "\n  ".join(sorted(failures)))


def trace_files(test, graph):
    """Include caller frames: app tests may use shared registration helpers."""
    node = test.get("backtrace")
    files = []
    while node is not None:
        frame = graph["nodes"][node]
        files.append(graph["files"][frame["file"]])
        node = frame.get("parent")
    return files


def owner(file, source):
    try:
        path = Path(file)
        if not path.is_absolute():
            path = source / path
        relative = path.resolve().relative_to(source.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) >= 2 and parts[0] in {"apps", "examples"}:
        return "/".join(parts[:2])
    if parts and parts[0] in {"tests", "cmake"}:
        return parts[0]
    return None


def select_tests(plan, listing, source):
    selected = set()
    consumer_smokes = {}
    for test in listing["tests"]:
        name = test["name"]
        if "benchmark" in name.lower():
            continue
        owners = {owner(f, source) for f in trace_files(test, listing["backtraceGraph"])}
        if plan["shared"]:
            # Library unit/integration tests plus SDK/native CLI contracts.
            if owners & {"tests", "cmake"}:
                selected.add(name)
            # At least one executable check per configured consumer. The full
            # app/example regression suites remain in the master tier.
            for component in sorted(o for o in owners if o and o.startswith(("apps/", "examples/"))):
                consumer_smokes.setdefault(component, name)
        if plan["cli"] and name.startswith("NativeCLI_"):
            selected.add(name)
        if plan["examples"] and any(o and o.startswith("examples/") for o in owners):
            selected.add(name)
        if owners & {"apps/" + app for app in plan["apps"]}:
            selected.add(name)
    selected.update(consumer_smokes.values())
    return sorted(selected)


def build_targets(plan, codemodel, reply):
    if plan["shared"]:
        return ["all"]
    targets = set()
    if plan["cli"]:
        targets.update({"openpfc", "openpfc_cli"})
    found_apps = set()
    for entry in codemodel["configurations"][0]["targets"]:
        target = json.loads((reply / entry["jsonFile"]).read_text())
        path = PurePosixPath(target["paths"]["source"])
        parts = path.parts
        if parts and parts[0] == "examples" and plan["examples"]:
            targets.add(target["name"])
        if len(parts) > 1 and parts[0] == "apps" and parts[1] in plan["apps"]:
            targets.add(target["name"])
            found_apps.add(parts[1])
    missing = set(plan["apps"]) - found_apps
    if missing:
        raise RuntimeError("No configured targets for changed apps: " + ", ".join(sorted(missing)))
    if not targets:
        raise RuntimeError("C++ selection produced no build targets")
    return sorted(targets)


def run_pr(plan, build):
    def run(command):
        print("+", command, flush=True)
        subprocess.run(command, check=True)

    source = Path.cwd().resolve()
    build = Path(build).resolve()
    query = build / ".cmake/api/v1/query"
    query.mkdir(parents=True, exist_ok=True)
    (query / "codemodel-v2").touch()
    apps = plan["shared"] or bool(plan["apps"])
    examples = plan["shared"] or plan["examples"]
    run(["cmake", "-S", str(source), "-B", str(build), "-GNinja",
         "-DCMAKE_BUILD_TYPE=Release", "-DOpenPFC_BUILD_TESTS=ON",
         "-DOpenPFC_BUILD_DOCUMENTATION=OFF", "-DOpenPFC_RUN_MPI_SUITES=ON",
         "-DOpenPFC_MPI_TEST_MAX_WORLD_SIZE=2",
         "-DOpenPFC_BUILD_APPS=" + ("ON" if apps else "OFF"),
         "-DOpenPFC_BUILD_EXAMPLES=" + ("ON" if examples else "OFF")])
    reply = build / ".cmake/api/v1/reply"
    index = json.loads(max(reply.glob("index-*.json"), key=lambda p: p.stat().st_mtime).read_text())
    codemodel = json.loads((reply / index["reply"]["codemodel-v2"]["jsonFile"]).read_text())
    targets = build_targets(plan, codemodel, reply)
    run(["cmake", "--build", str(build), "--parallel", "2", "--target", *targets])
    listing = json.loads(subprocess.check_output(
        ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True))
    tests = select_tests(plan, listing, source)
    if not tests:
        raise RuntimeError("C++ selection produced no tests; add the missing component mapping")
    (build / "pr-selection.json").write_text(json.dumps(dict(plan=plan, targets=targets, tests=tests), indent=2))
    print("Selected tests:", *tests, sep="\n  ", flush=True)
    expression = "^(" + "|".join(re.escape(test) for test in tests) + ")$"
    run(["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error",
         "--parallel", "2", "--timeout", "300", "--tests-regex", expression])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["select", "run-pr", "status"])
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "pull_request"))
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--build", default="build")
    args = parser.parse_args()
    if args.mode == "select":
        paths = []
        if args.event == "pull_request":
            if not args.base:
                parser.error("PR selection requires --base")
            # Include both sides of renames/deletions and paths containing spaces.
            raw = subprocess.check_output(["git", "diff", "--name-only", "--no-renames",
                                           "-z", args.base + "..." + args.head])
            paths = [p.decode("utf-8") for p in raw.split(b"\0") if p]
        plan = classify(paths, args.event)
        print(json.dumps(plan, indent=2))
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write("plan=" + json.dumps(plan) + "\n")
                for key in ("cpp", "python", "full"):
                    output.write(key + "=" + str(plan[key]).lower() + "\n")
    elif args.mode == "run-pr":
        run_pr(json.loads(os.environ["CI_PLAN"]), args.build)
    else:
        check_status(json.loads(os.environ["CI_PLAN"]), json.loads(os.environ["RESULTS"]))
        print("All selected CI checks passed")


if __name__ == "__main__":
    main()
