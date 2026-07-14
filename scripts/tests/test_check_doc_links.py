#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for fenced-code handling in ``check_doc_links.py``."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from check_doc_links import strip_fenced_code  # noqa: E402


def test_strip_fenced_code_removes_non_indented_blocks() -> None:
    markdown = """Text before.

```cpp
int x = 5;
```

Text after.
"""

    result = strip_fenced_code(markdown)

    assert "int x = 5;" not in result
    assert "Text before." in result
    assert "Text after." in result


def test_strip_fenced_code_removes_indented_blocks() -> None:
    markdown = """- Item with an example:

  ```cpp
  int y = 10;
  ```

- Following item
"""

    result = strip_fenced_code(markdown)

    assert "int y = 10;" not in result
    assert "- Item with an example:" in result
    assert "- Following item" in result


def test_strip_fenced_code_handles_multiple_blocks() -> None:
    markdown = """First.

```cpp
int x = 5;
```

Middle.

  ```python
  value = 10
  ```

Last.
"""

    result = strip_fenced_code(markdown)

    assert "int x = 5;" not in result
    assert "value = 10" not in result
    assert "First." in result
    assert "Middle." in result
    assert "Last." in result


def test_links_in_code_blocks_not_reported() -> None:
    markdown = """[Real link](docs/README.md)

```text
[Not a link](missing.md)
```
"""

    result = strip_fenced_code(markdown)

    assert "[Real link](docs/README.md)" in result
    assert "[Not a link](missing.md)" not in result


def test_cpp_lambdas_not_reported_as_links() -> None:
    markdown = """- Example:

  ```cpp
  auto function = [const auto& idx, int i, int j, int k]() {};
  ```
"""

    result = strip_fenced_code(markdown)

    assert "[const auto& idx, int i, int j, int k]" not in result


def test_real_links_still_detected() -> None:
    markdown = """[Installation guide](INSTALL.md)

  ```cpp
  [Not a link](missing.md)
  ```

[Reference](docs/reference.md)
"""

    result = strip_fenced_code(markdown)

    assert "[Installation guide](INSTALL.md)" in result
    assert "[Reference](docs/reference.md)" in result
    assert "[Not a link](missing.md)" not in result
