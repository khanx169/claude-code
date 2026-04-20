#!/usr/bin/env bash
# PostToolUse hook — auto-format and lint-fix any .py file Claude writes or edits.
#
# Claude Code passes a JSON object on stdin describing the tool call:
#   { "tool_name": "Edit", "tool_input": { "file_path": "..." }, ... }
#
# For Write/Edit the path lives in .tool_input.file_path; for MultiEdit the
# response surfaces it under .tool_response.filePath — we check both.

set -euo pipefail

# --- 1. Extract the file path from stdin JSON ---
# Capture stdin first; piping directly to python3 -c would lose it to the heredoc.
stdin=$(cat)
f=$(echo "$stdin" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('tool_input', {}).get('file_path') or d.get('tool_response', {}).get('filePath', ''))
")

# --- 2. Skip anything that isn't a Python file ---
[[ "$f" == *.py ]] || exit 0

# --- 3. Run ruff (silently; errors don't block the edit) ---
.venv/bin/ruff format "$f" --quiet 2>/dev/null || true
.venv/bin/ruff check --fix "$f" --quiet 2>/dev/null || true
