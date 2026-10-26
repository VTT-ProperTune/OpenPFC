# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Give static Markdown checklist states the same accessible name as their text."""

from html import escape

from docutils import nodes


def label_checklists(app, doctree, docname):
    for raw in list(doctree.findall(nodes.raw)):
        html = raw.astext()
        if "task-list-item-checkbox" not in html or "aria-label=" in html:
            continue
        container = raw.parent
        while container is not None and not isinstance(
            container, (nodes.paragraph, nodes.list_item)
        ):
            container = container.parent
        if container is None:
            continue
        text = container.deepcopy()
        for markup in list(text.findall(nodes.raw)):
            markup.parent.remove(markup)
        label = " ".join(text.astext().split())
        if label:
            html = html.replace(
                "<input ", f'<input aria-label="{escape(label, quote=True)}" ', 1
            )
            raw.replace_self(nodes.raw(raw.rawsource, html, format="html"))


def setup(app):
    app.connect("doctree-resolved", label_checklists)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
