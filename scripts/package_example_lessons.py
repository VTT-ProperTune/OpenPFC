#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build complete source downloads for the existing Quarto lessons.

The archive keeps a theme's relative include layout and the exact published
source bytes. Only the model, its quoted dependencies and resources named in
the lesson are included. No numerical programs run during packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)
RESOURCE = re.compile(
    r"(?<![\w.-])([\w.-]+\.(?:json|toml|py|cpp|hpp|h)|requirements\.txt)(?![\w.-])"
)


def checked_source(path: Path, theme: Path) -> Path:
    """Require a real local source inside the declared theme boundary."""
    path = path.resolve()
    if not path.is_relative_to(theme.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-theme lesson source: {path.name}")
    return path


def lesson_sources(chapter: Path) -> list[Path]:
    """Find the transitive quoted includes and explicitly mentioned resources."""
    theme = chapter.parent.parent.resolve()
    files = {checked_source(chapter.parent / "model.cpp", theme)}
    if (chapter.parent / "input.json").is_file():
        files.add(checked_source(chapter.parent / "input.json", theme))
    for name in RESOURCE.findall(chapter.read_text(encoding="utf-8")):
        candidate = chapter.parent / name
        if candidate.exists():
            files.add(checked_source(candidate, theme))
    pending = list(files)
    while pending:
        source = pending.pop()
        if source.suffix not in {".cpp", ".hpp", ".h"}:
            continue
        for name in INCLUDE.findall(source.read_text(encoding="utf-8")):
            dependency = checked_source(source.parent / name, theme)
            if dependency not in files:
                files.add(dependency)
                pending.append(dependency)
    return sorted(files)


def run_commands(chapter: Path, options: dict) -> list[list[str]]:
    """Provide a small runnable case, preserving flag-driven model contracts."""
    entry = f"{chapter.parent.name}/model.cpp"
    data = chapter.parent / "input.json"
    if data.is_file():
        return [
            [
                "openpfc",
                "run",
                entry,
                f"{chapter.parent.name}/input.json",
                "--source-root",
                ".",
                "--launcher=none",
            ]
        ]
    if not options:
        raise ValueError(f"Declare a small command-line case for {chapter.parent.name}")
    command = ["./results/model", *options["args"]]
    if options.get("ranks", 1) > 1:
        command = ["mpirun", "-n", str(options["ranks"]), *command]
    return [
        ["mkdir", "-p", "results"],
        ["openpfc", "compile", entry, "--source-root", ".", "-o", "results/model"],
        command,
    ]


def archive_bytes(
    chapter: Path, revision: str, options: dict, license_text: bytes
) -> bytes:
    """Return a reproducible ZIP with public source identity and run instructions."""
    chapter = chapter.resolve()
    theme = chapter.parent.parent
    slug = f"{theme.name}/{chapter.parent.name}"
    prefix = f"openpfc-{theme.name}-{chapter.parent.name}"
    files = {
        path.relative_to(theme).as_posix(): path.read_bytes()
        for path in lesson_sources(chapter)
    }
    commands = run_commands(chapter, options)
    title_match = re.search(r"^title:\s*(.+)$", chapter.read_text(), re.MULTILINE)
    title = title_match[1].strip().strip("\"'") if title_match else slug
    url = f"https://vtt-propertune.github.io/OpenPFC/learn/{slug}/"
    manifest = {
        "schema": 1,
        "lesson": slug,
        "source_revision": revision,
        "entrypoint": f"{chapter.parent.name}/model.cpp",
        "commands": commands,
        "sources": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(files.items())
        },
        "license": "AGPL-3.0-or-later",
        "lesson_url": url,
    }
    instructions = "\n".join(shlex.join(command) for command in commands)
    readme = f"""<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# {title}

Read the equations, expected result and experiments in [the lesson]({url}).

## Run this download

Extract the archive and open a terminal in `{prefix}`. Use a compatible
installed OpenPFC SDK and native CLI, with its compiler/MPI environment loaded.
The [installation guide](https://vtt-propertune.github.io/OpenPFC/docs/project/install/)
explains that setup. Run `openpfc doctor` once to check it.

From this directory:

```sh
{instructions}
```

The model is the same C++ shown in the lesson. Shared headers keep their
relative paths; `--source-root .` includes them in the build snapshot.
Edit the model or its input, predict what will change, and repeat the command.
Recorded `openpfc run` executions print their attempt directory. Open its
`stdout.log` for the program's terminal output and `results/` for data files.
Direct executable runs write to the paths in the command
or the model's `results/` directory.

Additional inputs named in the lesson are included where provided. Follow
the lesson for their assumptions, numerical checks and any MPI restrictions.
The optional Python helpers require their documented dependencies; ordinary
C++ execution uses the installed SDK.

Checkout revision: `{revision}`. `lesson.json` identifies the bundled files and their
checksums. Source attribution is preserved; the license text is in `LICENSES/`.
"""
    files["README.md"] = readme.encode()
    files["lesson.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    files["LICENSES/AGPL-3.0-or-later.txt"] = license_text
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{prefix}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return output.getvalue()


def package(source: Path, destination: Path, revision: str) -> list[Path]:
    """Write adjacent lesson.zip files under source or its staged website copy."""
    source = source.resolve()
    destination = destination.resolve()
    config = tomllib.loads((source / "downloads.toml").read_text())
    license_text = (source.parent / "LICENSES/AGPL-3.0-or-later.txt").read_bytes()
    outputs = []
    for chapter in sorted(source.glob("*/*/index.qmd")):
        slug = chapter.parent.relative_to(source).as_posix()
        data = archive_bytes(
            chapter, revision, config.get("cli", {}).get(slug, {}), license_text
        )
        target = destination / chapter.parent.relative_to(source) / "lesson.zip"
        if target.exists():
            try:
                with zipfile.ZipFile(target) as old:
                    manifests = [
                        name for name in old.namelist() if name.endswith("/lesson.json")
                    ]
                    record = (
                        json.loads(old.read(manifests[0]))
                        if len(manifests) == 1
                        else {}
                    )
                    owned = record.get("schema") == 1 and record.get("lesson") == slug
            except (zipfile.BadZipFile, ValueError):
                owned = False
            if not owned:
                raise ValueError(
                    f"Refusing to replace an unowned lesson archive: {target.name}"
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
        outputs.append(target)
    return outputs


def source_revision(root: Path) -> str:
    """Use a checkout identity when available; file hashes also cover source exports."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return "unversioned"
    lines = result.stdout.splitlines()
    if (
        result.returncode == 0
        and len(lines) == 2
        and Path(lines[0]).resolve() == root.resolve()
    ):
        return lines[1]
    return "unversioned"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples")
    args = parser.parse_args()
    outputs = package(ROOT / "examples", args.output.resolve(), source_revision(ROOT))
    print(f"Packaged {len(outputs)} runnable lesson downloads")


if __name__ == "__main__":
    main()
