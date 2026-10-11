"""Generated paper LaTeX (scripts/exp10_report.py, scripts/paper_appendix.py) stays structurally valid."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["paper_phase2.tex", "paper_appendix_ab.tex"])
def test_generated_tex_structure(name):
    s = (ROOT / "docs" / name).read_text(encoding="utf-8")
    body = "\n".join(line for line in s.splitlines() if not line.lstrip().startswith("%"))
    depth = 0
    for ch in body.replace("\\{", "").replace("\\}", ""):
        depth += (ch == "{") - (ch == "}")
        assert depth >= 0
    assert depth == 0
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([A-Za-z*]+)\}", body):
        if kind == "begin":
            stack.append(env)
        else:
            assert stack and stack.pop() == env, env
    assert not stack
    for spec, block in re.findall(r"\\begin\{tabular\}\{((?:[^{}]|\{\})*)\}(.*?)\\end\{tabular\}", body, re.S):
        ncol = len(re.findall(r"[lcrpX]", re.sub(r"\{[^}]*\}", "", spec)))   # column types only (not | or @{})
        assert all(r.count("&") == ncol - 1 for r in block.split("\\\\")
                   if "&" in r and "\\multicolumn" not in r), spec            # spanned header rows are exempt
    labels = re.findall(r"\\label\{([^}]+)\}", body)
    assert len(labels) == len(set(labels))     # refs may point into main.tex; the real compile checks those
    assert "nan" not in body and "None" not in body
