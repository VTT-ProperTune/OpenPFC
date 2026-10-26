# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exercise the actual staging command and its source/website boundary."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
COMMAND = [sys.executable, str(ROOT / "scripts/stage_example_book.py")]


def test_stage_preserves_site_and_imports_runnable_sources(tmp_path):
    configuration = "project:\n  type: website\n  render: [index.qmd]\n"
    (tmp_path / "_quarto.yml").write_text(configuration)
    for _ in range(2):  # Regeneration should be repeatable.
        subprocess.run([*COMMAND, str(tmp_path)], check=True, capture_output=True)
    assert (tmp_path / "_quarto.yml").read_text() == configuration
    expected = {path.relative_to(ROOT / "examples") for path in
                (ROOT / "examples").glob("*/*/model.cpp")
                if (path.parent / "index.qmd").is_file()}
    actual = {path.relative_to(tmp_path / "learn") for path in
              (tmp_path / "learn").glob("*/*/model.cpp")}
    assert actual == expected
    assert not (tmp_path / "learn/check_lessons.py").exists()
    assert not (tmp_path / "learn/_quarto.yml").exists()
    assert "learn/*/*/index.qmd" in (tmp_path / "_quarto-examples.yml").read_text()
    assert "openpfc-include-code-files" in (tmp_path / "learn/_metadata.yml").read_text()
    assert (tmp_path / "learn/coverage.json").read_bytes() == (ROOT / "examples/coverage.json").read_bytes()
    assert "learn/coverage.qmd" in (tmp_path / "_quarto-examples.yml").read_text()
    for chapter in (ROOT / "examples").glob("*/*/index.qmd"):
        downloads = [*chapter.parent.glob("*.json"), *chapter.parent.glob("*.toml"),
                     *chapter.parent.glob("*.hpp"), *chapter.parent.parent.glob("*.hpp"),
                     *chapter.parent.glob("*.py"), *chapter.parent.glob("requirements.txt")]
        for source in downloads:
            assert (tmp_path / "learn" / source.relative_to(ROOT / "examples")).read_bytes() == source.read_bytes()
    assert "learn/**/*.py" in (tmp_path / "_quarto-examples.yml").read_text()
    assert "learn/**/requirements.txt" in (tmp_path / "_quarto-examples.yml").read_text()
    assert "learn/**/*.toml" in (tmp_path / "_quarto-examples.yml").read_text()


def test_inventory_matches_current_targets_and_chapter_declarations():
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_example_book.py")],
                   check=True, capture_output=True)


def test_stage_refuses_unowned_lesson_directory(tmp_path):
    (tmp_path / "_quarto.yml").write_text("project:\n  type: website\n")
    (tmp_path / "learn").mkdir()
    owned = tmp_path / "learn/index.qmd"
    owned.write_text("An existing tutorial")
    result = subprocess.run([*COMMAND, str(tmp_path)], capture_output=True)
    assert result.returncode != 0
    assert owned.read_text() == "An existing tutorial"
