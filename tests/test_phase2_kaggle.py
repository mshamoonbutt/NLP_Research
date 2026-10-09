"""Phase 2 Kaggle driver: every trained arm Exp 8 evaluates has an adapter Exp 7 trains."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("phase2_kaggle", ROOT / "scripts" / "phase2_kaggle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_evaluated_arm_has_its_adapter():
    pk = _load()
    for n, nab in ((60, 48), (50, 50), (24, 20)):
        jobs, evals = pk.plan("phi3", n, nab, [42, 43, 44], [25, 50])
        trained = {name for name, _ in jobs}
        assert len(trained) == len(jobs)                                  # no adapter trained twice
        for tag, arms, split in evals:
            for arm in arms:
                if arm in ("A", "E"):                                     # untrained arms
                    continue
                assert f"{arm}_phi3_{tag}" in trained, (n, tag, arm)
            assert (split is not None) == tag.endswith("ablation_D6")
        assert sum("E" in arms for _, arms, _ in evals) == 1               # the safety prompt once
        assert all(b < n for b in (25, 50) if b != n and f"C_phi3_n{b}" in trained)   # n-curve only below N
    jobs, _ = pk.plan("phi3", 60, 48, [42, 43], [25])
    b43 = dict(jobs)["B_ext_phi3_n60_s43"]
    assert b43[b43.index("--seed") + 1] == "43" and "--naturalness-csv" not in b43
