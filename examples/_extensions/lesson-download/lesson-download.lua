-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Add a common entry point without copying download markup into every lesson.
function Pandoc(doc)
  if not doc.meta.targets then return doc end
  local link = pandoc.Link("Download runnable lesson (.zip)", "lesson.zip")
  local note = pandoc.Div({
    pandoc.Para({link}),
    pandoc.Para({pandoc.Str("Includes the model, inputs, required headers and a short run guide.")})
  }, pandoc.Attr("", {"lesson-download"}))
  for _, block in ipairs(doc.blocks) do
    if block.t == "Div" and block.classes:includes("article-body") then
      table.insert(block.content, 2, note)
      return doc
    end
  end
  table.insert(doc.blocks, 1, note)
  return doc
end
