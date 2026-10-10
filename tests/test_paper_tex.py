"""Generated paper LaTeX (scripts/exp10_report.py) stays structurally valid: no compiler is assumed."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_phase2_tex_structure():
    s = (ROOT / "docs" / "paper_phase2.tex").read_text(encoding="utf-8")
    body = "\n".join(line for line in s.splitlines() if not line.lstrip().startswith("%"))
    depth = 0
    for ch in body.replace("\\{", "").replace("\\}", ""):
        depth += (ch == "{") - (ch == "}")
        assert depth >= 0
    assert depth == 0
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([A-Za-z*]+)\}", body):
        if kind == "begin":
            stack.append(name)
        else:
            assert stack and stack.pop() == name, name
    assert not stack
    for spec, block in re.findall(r"\\begin\{tabular\}\{((?:[^{}]|\{\})*)\}(.*?)\\end\{tabular\}", body, re.S):
        ncol = len(re.sub(r"@\{\}", "", spec))
        assert all(r.count("&") == ncol - 1 for r in block.split("\\\\") if "&" in r), spec
    labels = re.findall(r"\\label\{([^}]+)\}", body)
    assert len(labels) == len(set(labels)) and set(re.findall(r"\\ref\{([^}]+)\}", body)) <= set(labels)
    assert "nan" not in body and "None" not in body
