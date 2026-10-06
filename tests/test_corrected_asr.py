"""ASR corrected for the judge's measured per-language error (Rogan-Gladen)."""
from __future__ import annotations

import json

from tests.conftest import result

from csjail.aggregate import judge_error_counts, summarize
from csjail.asr import corrected_asr, rogan_gladen


def test_rogan_gladen():
    assert abs(rogan_gladen(0.26, 0.8, 0.9) - 0.16 / 0.7) < 1e-12
    assert abs(rogan_gladen(0.3, 1.0, 1.0) - 0.3) < 1e-12    # a perfect judge changes nothing
    assert rogan_gladen(0.02, 0.8, 0.9) == 0.0                # below the false-alarm floor: clipped
    assert rogan_gladen(0.5, 0.5, 0.5) is None                # judge no better than chance


def test_corrected_asr_interval_and_aggregate_wiring(tmp_path):
    outcomes = [i % 4 == 0 for i in range(200)]               # observed 0.25
    fams = [f"f{i // 2}" for i in range(200)]
    counts = {"tp": 80, "fn": 20, "tn": 270, "fp": 30}        # sens 0.8, spec 0.9
    r = corrected_asr(outcomes, fams, counts, bootstrap_n=500)
    assert abs(r["asr_corrected"] - 0.15 / 0.7) < 1e-12
    assert r["asr_corrected_ci_lo"] < r["asr_corrected"] < r["asr_corrected_ci_hi"]
    recs = [result("m", f"f{i}", "EN", i % 4 == 0) for i in range(40)]
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"judge_fingerprint": {"fingerprint_id": "fp-1"},
                               "result": {"per_language": {"EN": dict(counts, precision=0.7)}}}))
    got = judge_error_counts(man, recs)
    assert got == {"EN": counts}
    row = next(x for x in summarize(recs, bootstrap_n=200, error_counts=got) if x["domain"] == "ALL")
    assert abs(row["asr_corrected"] - 0.15 / 0.7) < 1e-12
    man.write_text(json.dumps({"judge_fingerprint": {"fingerprint_id": "other"},
                               "result": {"per_language": {"EN": counts}}}))
    assert judge_error_counts(man, recs) is None              # another judge's error never applies
