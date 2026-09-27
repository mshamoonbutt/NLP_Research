"""phase2_feasibility: C vs matched counts, rater disagreement -> unresolved."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def w(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        d = csv.DictWriter(f, fieldnames=list(rows[0]))
        d.writeheader()
        d.writerows(rows)


def test_counts(tmp_path):
    spec = importlib.util.spec_from_file_location("feas", ROOT / "scripts" / "phase2_feasibility.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["feas"] = m
    spec.loader.exec_module(m)
    # f1: CS+EN unsafe (matched), f2: CS only, f3: neither, f4: raters disagree on CS
    lab = {("f1", "CS"): "true", ("f1", "EN"): "true", ("f2", "CS"): "true", ("f2", "EN"): "false",
           ("f3", "CS"): "false", ("f3", "EN"): "false", ("f4", "CS"): "true", ("f4", "EN"): "false"}
    items = [{"item_id": f"x:{f}:{c}", "model": "phi3", "base_id": f, "condition": c,
              "domain_id": "D1"} for (f, c) in lab]
    w(tmp_path / "items.csv", items)
    w(tmp_path / "rater1.csv", [{"item_id": f"x:{f}:{c}", "unsafe": v} for (f, c), v in lab.items()])
    r2 = [{"item_id": f"x:{f}:{c}", "unsafe": ""} for (f, c) in lab]
    r2[6]["unsafe"] = "false"   # disagree on f4/CS
    w(tmp_path / "rater2.csv", r2)
    rep = m.report(tmp_path, pool_size=100)
    p = rep["models"]["phi3"]
    assert p["C_cs_pairs"]["pilot_count"] == 2 and p["C_cs_pairs"]["reviewed_families"] == 3
    assert p["matched_B_C_D"]["pilot_count"] == 1
    assert rep["n_rater_disagreements"] == 1
    assert p["C_cs_pairs"]["projected_train_pool"][1] == round(2 / 3 * 100)
