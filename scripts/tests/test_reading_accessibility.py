# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static checklists expose their authored text without changing their state."""

import importlib.util
from html.parser import HTMLParser
from pathlib import Path

import pytest

nodes = pytest.importorskip("docutils.nodes")
spec = importlib.util.spec_from_file_location(
    "reading_accessibility",
    Path(__file__).resolve().parents[2] / "docs/_ext/reading_accessibility.py",
)
accessibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accessibility)


class Input(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "input":
            self.attrs = dict(attrs)


def test_static_checklist_has_escaped_name_and_retains_state():
    raw = nodes.raw(
        "",
        '<input class="task-list-item-checkbox" disabled="disabled" checked="checked" type="checkbox">',
        format="html",
    )
    paragraph = nodes.paragraph()
    paragraph.extend(
        [raw, nodes.Text('Check "inputs" and '), nodes.literal("", "x < n")]
    )
    root = nodes.section("", nodes.bullet_list("", nodes.list_item("", paragraph)))
    accessibility.label_checklists(None, root, "example")
    accessibility.label_checklists(None, root, "example")
    parser = Input()
    parser.feed(next(root.findall(nodes.raw)).astext())
    assert parser.attrs["aria-label"] == 'Check "inputs" and x < n'
    assert parser.attrs["disabled"] == "disabled"
    assert parser.attrs["checked"] == "checked"


def test_existing_labels_and_other_html_remain_unchanged():
    html = '<input class="task-list-item-checkbox" aria-label="Authored name" disabled="disabled" type="checkbox">'
    paragraph = nodes.paragraph()
    paragraph.extend([nodes.raw("", html, format="html"), nodes.Text("Text")])
    root = nodes.section("", paragraph)
    accessibility.label_checklists(None, root, "example")
    assert next(root.findall(nodes.raw)).astext() == html
