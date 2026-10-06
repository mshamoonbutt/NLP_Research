"""Simplified Exp 1: per-language metrics and the declared macro-F1 choice."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("exp1_compare_judges",
                                                  ROOT / "scripts" / "exp1_compare_judges.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp1_compare_judges"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_metrics_and_choice():
    cj = load()
    m = cj.metrics([(True, True)] * 8 + [(True, False)] * 2 + [(False, True)] * 2 + [(False, False)] * 88)
    assert (m["precision"], m["recall"], m["f1"]) == (0.8, 0.8, 0.8)
    assert abs(m["kappa"] - (0.96 - 0.82) / (1 - 0.82)) < 1e-9          # po .96, pe .82
    items = [{"item_id": f"{c}{n}", "condition": c} for c in cj.CONDS for n in range(20)]
    labels = {it["item_id"]: it["item_id"].endswith(("0", "1")) for it in items}   # 2 harmful each
    perfect = dict(labels)
    noisy = {i: (not v) if i.endswith("9") else v for i, v in labels.items()}      # 1 FP per language
    res = cj.compare(items, labels, {"noisy": noisy, "perfect": perfect})
    assert res["best"] == "perfect" and res["table"]["perfect"]["meets_090_everywhere"]
    assert abs(res["table"]["noisy"]["per_language"]["EN"]["precision"] - 2 / 3) < 1e-9
