"""LoRA target resolution, chat formatting, exemplars, comprehension, Exp 7 budgets."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from csjail.chosen_gen import GeneratorConfig, assert_distinct_from_judge, load_exemplars
from csjail.comprehension import build_intent_probe, conditioned_asr, parse_comprehension, scorer_agreement
from csjail.judge import load_judge_config
from csjail.train_dpo import LoraTargetError, resolve_lora_targets, to_conversational

ROOT = Path(__file__).resolve().parent.parent

PHI3 = {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"}
LLAMA = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}
ALL = ["q", "k", "v", "o", "gate", "up", "down"]


def test_lora_targets_resolve_fused_and_split():
    phi = resolve_lora_targets(PHI3, ALL)
    assert phi["target_modules"] == ["down_proj", "gate_up_proj", "o_proj", "qkv_proj"]
    assert set(phi["fused"]) == {"q", "k", "v", "gate", "up"}
    lla = resolve_lora_targets(LLAMA, ALL)
    assert lla["target_modules"] == sorted(LLAMA - {"lm_head"}) and lla["fused"] == []
    with pytest.raises(LoraTargetError):
        resolve_lora_targets({"o_proj", "down_proj"}, ALL)   # the old silent-partial case


def test_conversational_format():
    c = to_conversational({"prompt": "p", "chosen": "c", "rejected": "r"})
    assert c["prompt"] == [{"role": "user", "content": "p"}]
    assert c["chosen"][0]["role"] == "assistant" and c["rejected"][0]["content"] == "r"


def test_generator_distinct_from_judge():
    judge = load_judge_config()
    assert_distinct_from_judge(GeneratorConfig("anthropic", "claude-sonnet-4-5"), judge)
    with pytest.raises(ValueError):
        assert_distinct_from_judge(GeneratorConfig(judge.provider, judge.model), judge)


def test_exemplars_domain_tagged_and_excludable():
    cs = load_exemplars("CS")
    assert all(e["domain_id"] for e in cs) and all(e["in_taxonomy"] for e in cs)
    assert len(cs) == 7                                        # out-of-taxonomy one skipped
    assert all(e["domain_id"] != "D2" for e in load_exemplars("CS", exclude_domains=["D2"]))
    assert {e["domain_id"] for e in load_exemplars("EN")} == {e["domain_id"] for e in cs}


def test_comprehension_buckets():
    assert parse_comprehension('{"comprehension": "refused_probe"}').label == "refused_probe"
    assert parse_comprehension("understood").label is None
    s = conditioned_asr([True, False, None, False, True],
                        ["understood", "understood", "understood", "refused_probe", None])
    assert s["understood"] == {"n": 3, "n_scored": 2, "unsafe_rate": 0.5}
    assert s["refused_probe"]["n"] == 1 and s["scorer_missing"]["n"] == 1
    assert s["understood_share_of_decided"] == 1.0
    assert "{message}" not in build_intent_probe("hello")
    assert scorer_agreement([{"human_comprehension": "", "comprehension": "x"}])["status"] == "NOT_REVIEWED"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_exp7_budget_prefix_never_pads(tmp_path):
    exp7 = _load_script("exp7_train_arms")
    pairs = [{"base_id": f"f{i}", "order_rank": i} for i in range(10)]
    assert [p["base_id"] for p in exp7.take_budget(list(reversed(pairs)), "4")] == ["f0", "f1", "f2", "f3"]
    with pytest.raises(Exception, match="not padding"):
        exp7.take_budget(pairs, "50")
    csv = tmp_path / "nat.csv"
    csv.write_text("naturalness_1to5,clean_refusal_yes_no\n5,yes\n4,yes\n", encoding="utf-8")
    assert exp7.naturalness_gate(str(csv))["status"] == "PASS"
    csv.write_text("naturalness_1to5,clean_refusal_yes_no\n5,yes\n3,no\n", encoding="utf-8")
    assert exp7.naturalness_gate(str(csv))["status"] == "FAIL"
    assert exp7.naturalness_gate(None)["status"] == "MISSING"


def test_exp8_flags_na():
    exp8 = _load_script("exp8_posteval")
    assert exp8.flag(None, lambda v: v > 0) == "NA"
    assert exp8.flag(0.3, lambda v: v >= 0.5) is False


def _jsonl(path, rows):
    import json as _j
    path.write_text("".join(_j.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_exp7_primary_arms_equal_budget(tmp_path):
    exp7 = _load_script("exp7_train_arms")
    pd = tmp_path / "pairs"
    pd.mkdir()
    cs = [{"base_id": f"f{i}", "order_rank": i, "prompt": "p", "chosen": "c", "rejected": "r"}
          for i in range(6)]
    _jsonl(pd / "pairs_cs_all.jsonl", cs)
    _jsonl(pd / "pairs_cs_matched.jsonl", cs[:2])
    _jsonl(pd / "pairs_en_matched.jsonl", [dict(p) for p in cs[:2]])
    ext = tmp_path / "ext.jsonl"
    _jsonl(ext, [{"prompt": f"e{i}", "chosen": "c", "rejected": "r"} for i in range(4)])
    cfg = {"prefdata": {"external_english_pairs": str(ext)}, "d_budget": "matched"}
    c = exp7.select_pairs("C", pd, "3", cfg)
    b = exp7.select_pairs("B_ext", pd, "3", cfg)
    assert [p["base_id"] for p in c] == ["f0", "f1", "f2"] and len(b) == 3     # equal N
    assert all(p["source"] == "external" for p in b)
    d = exp7.select_pairs("D", pd, "3", cfg)
    assert len(d) == 3 and sum(p.get("source") == "external" for p in d) == 2
    assert len(exp7.select_pairs("B_matched", pd, "all", cfg)) == 2
    with pytest.raises(Exception, match="external English pairs"):
        exp7.select_pairs("B_ext", pd, "all", cfg)                          # 6 > 4 available


def test_external_english_pair_rule():
    ext = _load_script("prepare_external_english_pairs")
    rows = [{"prompt": "a", "response_0": "safe", "response_1": "bad",
             "is_response_0_safe": True, "is_response_1_safe": False},
            {"prompt": "b", "response_0": "x", "response_1": "y",
             "is_response_0_safe": True, "is_response_1_safe": True},        # both safe: skip
            {"prompt": "c", "response_0": "bad", "response_1": "safe",
             "is_response_0_safe": False, "is_response_1_safe": True}]
    pairs = ext.to_pairs(rows, source="t")
    assert [(p["prompt"], p["chosen"], p["rejected"]) for p in pairs] == [
        ("a", "safe", "bad"), ("c", "safe", "bad")]
