# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A downloaded model keeps its dependencies, inputs and source identity."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "package_example_lessons", ROOT / "scripts/package_example_lessons.py"
)
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def fixture(tmp_path):
    root = tmp_path / "source"
    source = root / "examples"
    chapter = source / "heat/decay"
    chapter.mkdir(parents=True)
    (chapter / "index.qmd").write_text(
        '---\ntitle: "Heat decay"\n---\nRead [input](input.json).\n'
    )
    (chapter / "model.cpp").write_text(
        '#include "../shared/equation.hpp"\nint main() { return answer; }\n'
    )
    (chapter / "input.json").write_text('{"steps": 2}\n')
    headers = source / "heat/shared"
    headers.mkdir()
    (headers / "equation.hpp").write_text('#include "value.hpp"\n')
    (headers / "value.hpp").write_text("inline constexpr int answer = 0;\n")
    (chapter / "private.env").write_text("Not a lesson resource")
    (chapter / "unmentioned.json").write_text("Not a lesson input")
    (source / "downloads.toml").write_text("")
    (root / "LICENSES").mkdir()
    (root / "LICENSES/AGPL-3.0-or-later.txt").write_text("Fixture license\n")
    return source, chapter


def test_archive_keeps_transitive_headers_and_omits_unrelated_files(tmp_path):
    source, chapter = fixture(tmp_path)
    output = tmp_path / "downloads"
    [path] = packager.package(source, output, "reviewed-source")
    first = path.read_bytes()
    assert packager.package(source, output, "reviewed-source")[0].read_bytes() == first
    with zipfile.ZipFile(path) as archive:
        prefix = "openpfc-heat-decay/"
        names = set(archive.namelist())
        assert prefix + "shared/value.hpp" in names
        assert not any("private" in name or "unmentioned" in name for name in names)
        record = json.loads(archive.read(prefix + "lesson.json"))
        assert record["commands"] == [
            [
                "openpfc",
                "run",
                "decay/model.cpp",
                "decay/input.json",
                "--source-root",
                ".",
                "--launcher=none",
            ]
        ]
        for relative, digest in record["sources"].items():
            data = archive.read(prefix + relative)
            assert data == (source / "heat" / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
        archive.extractall(tmp_path / "extracted")
    model = tmp_path / "extracted" / prefix / "decay/model.cpp"
    include = model.parent / "../shared/equation.hpp"
    assert include.is_file() and (include.parent / "value.hpp").is_file()


@pytest.mark.parametrize("include", ["../missing.hpp", "../../outside.hpp"])
def test_missing_or_escaping_dependencies_fail_before_packaging(tmp_path, include):
    source, chapter = fixture(tmp_path)
    (source / "outside.hpp").write_text("Not in the lesson's theme\n")
    (chapter / "model.cpp").write_text(f'#include "{include}"\n')
    with pytest.raises(ValueError, match="Missing or out-of-theme"):
        packager.package(source, tmp_path / "downloads", "revision")
    assert not list((tmp_path / "downloads").rglob("*.zip"))


def test_unowned_archive_is_preserved(tmp_path):
    source, chapter = fixture(tmp_path)
    target = tmp_path / "downloads/heat/decay/lesson.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"A user's existing file")
    with pytest.raises(ValueError, match="unowned lesson archive"):
        packager.package(source, tmp_path / "downloads", "revision")
    assert target.read_bytes() == b"A user's existing file"


def test_real_packaging_entrypoint_covers_published_models_and_flag_cases(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/package_example_lessons.py"),
            "--output",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )
    for chapter in (ROOT / "examples").glob("*/*/index.qmd"):
        with zipfile.ZipFile(
            tmp_path / chapter.parent.relative_to(ROOT / "examples") / "lesson.zip"
        ) as archive:
            record_name = next(
                name for name in archive.namelist() if name.endswith("/lesson.json")
            )
            prefix = record_name.removesuffix("lesson.json")
            record = json.loads(archive.read(record_name))
            assert (
                archive.read(prefix + record["entrypoint"])
                == (chapter.parent / "model.cpp").read_bytes()
            )
            assert record["commands"]
            if not (chapter.parent / "input.json").exists():
                assert any(
                    command[:2] == ["openpfc", "compile"]
                    for command in record["commands"]
                )
                assert record["commands"][-1][-1] != "--help"
