# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regressions for CI omissions, conservative fallback and required statuses."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci/scope.py"
spec = importlib.util.spec_from_file_location("ci_scope", SCRIPT)
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


def test_documentation_has_no_code_jobs():
    plan = scope.classify(["docs/tutorial.md", "README.md", "apps/heat3d/README.md"], "pull_request")
    assert not plan["cpp"]
    assert not plan["python"]
    assert scope.required_jobs(plan) == {"scope", "code-quality"}


def test_application_is_selected_without_other_apps():
    plan = scope.classify(["apps/heat3d/src/cpu/main.cpp"], "pull_request")
    assert plan["apps"] == ["heat3d"]
    assert not plan["shared"]
    assert not plan["examples"]


@pytest.mark.parametrize("path,component", [
    ("tools/openpfc/bundle.cpp", "cli"), ("tests/cli/check_models.cmake", "cli"),
    ("examples/diffusion.cpp", "examples")])
def test_cli_and_examples_select_their_own_components(path, component):
    plan = scope.classify([path], "pull_request")
    assert plan[component]
    assert not plan["shared"]


@pytest.mark.parametrize("paths", [
    ["include/openpfc/kernel/field.hpp"], ["apps/CMakeLists.txt"], ["CMakeLists.txt"],
    ["future-component/file.xyz"], [], [".github/workflows/ci.yml"], ["scripts/ci/scope.py"]])
def test_shared_unknown_and_empty_changes_cannot_skip_code(paths):
    plan = scope.classify(paths, "pull_request")
    assert plan["shared"] and plan["cpp"] and plan["reasons"]
    assert not plan["full"]


def test_python_packager_also_checks_native_cli():
    plan = scope.classify(["scripts/tests/test_bake_case.py"], "pull_request")
    assert plan["python"] and plan["cli"]


def test_cli_install_with_app_changes_cannot_omit_install_targets():
    plan = scope.classify(["tools/openpfc/main.cpp", "apps/heat3d/src/main.cpp"], "pull_request")
    assert plan["shared"]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_integration_always_runs_full_suite(event):
    plan = scope.classify(["README.md"], event)
    assert plan["full"]
    assert "hip-compile" in scope.required_jobs(plan)


def test_real_selection_entrypoint_includes_deleted_and_renamed_paths(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "apps/heat3d").mkdir(parents=True)
    (tmp_path / "apps/heat3d/model.cpp").write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("mv", "apps/heat3d/model.cpp", "README.md")
    git("commit", "-qam", "move")
    result = subprocess.check_output(
        [sys.executable, str(SCRIPT), "select", "--event", "pull_request", "--base", base],
        cwd=tmp_path, text=True,
        env={key: value for key, value in os.environ.items() if key != "GITHUB_OUTPUT"})
    assert json.loads(result)["apps"] == ["heat3d"]


@pytest.fixture
def listing():
    return {
        "backtraceGraph": {
            "files": ["/repo/tests/CMakeLists.txt", "/repo/apps/heat3d/CMakeLists.txt",
                      "/repo/apps/wave2d/CMakeLists.txt", "/repo/cmake/shared.cmake"],
            "nodes": [{"file": 0}, {"file": 1}, {"file": 2}, {"file": 3, "parent": 1}]},
        "tests": [{"name": "openpfc-all-tests", "backtrace": 0},
                  {"name": "heat-first", "backtrace": 1},
                  {"name": "heat-other", "backtrace": 1},
                  {"name": "wave-first", "backtrace": 2},
                  {"name": "heat-helper", "backtrace": 3}]}


def test_app_uses_its_tests_including_shared_helper_callers(listing):
    plan = scope.classify(["apps/heat3d/src/main.cpp"], "pull_request")
    assert scope.select_tests(plan, listing, Path("/repo")) == ["heat-first", "heat-helper", "heat-other"]


def test_shared_runs_library_tests_and_consumer_smokes(listing):
    plan = scope.classify(["CMakeLists.txt"], "pull_request")
    names = scope.select_tests(plan, listing, Path("/repo"))
    assert {"openpfc-all-tests", "heat-first", "wave-first"} <= set(names)
    assert "heat-other" not in names


def test_target_selection_does_not_build_unrelated_apps(tmp_path):
    entries = []
    for name in ["heat3d", "wave2d"]:
        file = name + ".json"
        (tmp_path / file).write_text(json.dumps({"name": name, "paths": {"source": "apps/" + name}}))
        entries.append({"jsonFile": file})
    model = {"configurations": [{"targets": entries}]}
    plan = scope.classify(["apps/heat3d/model.cpp"], "pull_request")
    assert scope.build_targets(plan, model, tmp_path) == ["heat3d"]
    plan["apps"] = ["deleted-or-unmapped"]
    with pytest.raises(RuntimeError, match="No configured targets"):
        scope.build_targets(plan, model, tmp_path)


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_only_intentionally_unselected_jobs_may_be_skipped(result):
    plan = scope.classify(["README.md"], "pull_request")
    results = {"scope": {"result": "success"}, "code-quality": {"result": "success"},
               "build-and-test": {"result": "skipped"}}
    scope.check_status(plan, results)
    results["scope"]["result"] = result
    with pytest.raises(RuntimeError):
        scope.check_status(plan, results)


@pytest.mark.parametrize("result", ["skipped", "failure", "cancelled", "missing"])
def test_selected_build_skips_or_missing_jobs_fail(result):
    plan = scope.classify(["tools/openpfc/main.cpp"], "pull_request")
    results = {job: {"result": "success"} for job in scope.required_jobs(plan)}
    if result == "missing":
        del results["build-and-test"]
    else:
        results["build-and-test"]["result"] = result
    with pytest.raises(RuntimeError):
        scope.check_status(plan, results)


def test_registration_paths_may_use_a_checkout_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    assert scope.owner(str(alias / "apps/heat3d/CMakeLists.txt"), source) == "apps/heat3d"


@pytest.mark.parametrize("component", ["examples/lesson", "apps/heat3d"])
def test_mixed_python_scope_builds_the_uploaded_cli(tmp_path, component):
    (tmp_path / "component.json").write_text(json.dumps({
        "name": "changed-model", "paths": {"source": component}}))
    model = {"configurations": [{"targets": [{"jsonFile": "component.json"}]}]}
    plan = scope.classify([component + "/model.cpp", "scripts/check_doc_links.py"],
                          "pull_request")
    assert plan["cpp"] and plan["python"] and not plan["cli"]
    assert not plan["shared"] and not plan["full"]
    assert scope.build_targets(plan, model, tmp_path) == [
        "changed-model", "openpfc", "openpfc_cli"]


def test_examples_without_python_do_not_build_cli(tmp_path):
    (tmp_path / "example.json").write_text(json.dumps({
        "name": "example", "paths": {"source": "examples/lesson"}}))
    model = {"configurations": [{"targets": [{"jsonFile": "example.json"}]}]}
    plan = scope.classify(["examples/lesson/model.cpp"], "pull_request")
    assert scope.build_targets(plan, model, tmp_path) == ["example"]
