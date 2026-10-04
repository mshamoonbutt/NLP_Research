"""End-to-end wiring of the GPU/API entry points with a FAKE model runner and
FAKE judge/generator (no GPU, no network). Proves the orchestration, not model
behaviour: Exp 1 sampling -> Exp 2 sweep (resume) -> aggregate -> Exp 3 ->
robustness -> Exp 4b -> Exp 6 -> Exp 8 preflight."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import csjail.judge as judge_mod
import csjail.models as models_mod
from csjail.judge import load_judge_config, parse_benign, parse_harm, rule_judgment
from csjail.judge_validation import PASS, write_manifest
from csjail.models import GenOutput
from tests.conftest import FIXTURE_CSV

ROOT = Path(__file__).resolve().parent.parent


def _h(*p) -> int:
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest(), 16)


class FakeRunner:
    calls = 0

    def __init__(self, spec, *, adapter_path=None, **kw):
        if adapter_path and not Path(adapter_path).exists():
            raise FileNotFoundError(adapter_path)
        self.spec, self.adapter_path = spec, adapter_path
        self.chat_template_sha256, self.template_probe = "tmpl", "<user>"

    def provenance(self, system=None):
        return {"model_key": self.spec.key, "hf_id": self.spec.hf_id,
                "revision": self.spec.revision, "chat_template_sha256": "tmpl",
                "adapter_sha256": None, "system_prompt_sha256": None, "template_probe": "<user>"}

    def generate(self, prompts, sampling, *, system=None, show_progress=True):
        FakeRunner.calls += len(prompts)
        out = []
        for p in prompts:
            draws = []
            for i in range(sampling.n):
                v = _h(self.spec.key, p, i) % 10
                t = ("UNSAFE fake" if v < 3 else "Main madad nahi kar sakta, maafi chahta hoon.")
                draws.append(GenOutput(t, i, "stop", 5, 5))
            out.append(draws)
        return out

    def generate_text(self, prompts, sampling=None, *, system=None):
        return [f"The message asks about {p[:10]}" for p in prompts]

    def shutdown(self):
        pass


SAFE = ('{"unsafe": false, "response_kind": "refusal", "compliance_degree": "none", '
        '"contains_refusal": true, "output_language_mismatch": false, "rationale": "x"}')
UNSAFE = ('{"unsafe": true, "response_kind": "harmful_compliance", "compliance_degree": "full", '
          '"contains_refusal": false, "output_language_mismatch": false, "rationale": "x"}')


class FakeJudge:
    def __init__(self, cfg=None, *, kind="harm"):
        self.cfg, self.kind = cfg or load_judge_config(), kind

    @property
    def fingerprint(self):
        return self.cfg.fingerprint(self.kind)

    def score_sync(self, pairs, show_progress=True):
        out = []
        for _p, r in pairs:
            ruled = rule_judgment(r, self.kind)
            if ruled:
                out.append(ruled)
            elif self.kind == "benign":
                out.append(parse_benign('{"refused": false, "response_kind": "safe_helpful"}'))
            else:
                out.append(parse_harm(UNSAFE if r.startswith("UNSAFE") else SAFE))
        return out


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(models_mod, "SLMRunner", FakeRunner)
    monkeypatch.setattr(judge_mod, "Judge", FakeJudge)
    exp0 = load_script("exp0_finalize_data")
    assert exp0.main(["--source-csv", str(FIXTURE_CSV), "--eval-size", "4", "--out-root",
                      str(tmp_path / "exp0"), "--no-latest"]) == 0
    exp0_dir = next((tmp_path / "exp0").glob("final-*"))
    man = tmp_path / "judge_manifest.json"
    write_manifest(man, fingerprint=load_judge_config().fingerprint("harm"),
                   result={"status": PASS}, sample_manifest={}, gold_sha256={}, gold_report={})
    bman = tmp_path / "judge_manifest_benign.json"
    write_manifest(bman, fingerprint=load_judge_config().fingerprint("benign"),
                   result={"status": PASS}, sample_manifest={}, gold_sha256={}, gold_report={})
    return {"tmp": tmp_path, "exp0": exp0_dir, "man": man, "bman": bman}


def test_full_wiring(env, monkeypatch):
    tmp, exp0_dir = env["tmp"], str(env["exp0"])
    from csjail import run_eval

    # Exp 1 sampler: 2 models x 4 conditions x 3 families from train_pool
    exp1 = load_script("exp1_sample_for_annotation")
    sd = tmp / "exp1_val"
    assert exp1.main(["--exp0-dir", exp0_dir, "--role", "validation", "--models", "qwen25", "phi3",
                      "--n-per-model-condition", "3", "--out-dir", str(sd)]) == 0
    sman = json.loads((sd / "sample_manifest.json").read_text(encoding="utf-8"))
    assert sman["n_items"] == 24 and (sd / "rater1.csv").exists()
    # resume: same plan -> zero new generations; different plan -> refused
    FakeRunner.calls = 0
    assert exp1.main(["--exp0-dir", exp0_dir, "--role", "validation", "--models", "qwen25", "phi3",
                      "--n-per-model-condition", "3", "--out-dir", str(sd)]) == 0
    assert FakeRunner.calls == 0
    assert exp1.main(["--exp0-dir", exp0_dir, "--role", "validation", "--models", "qwen25", "phi3",
                      "--n-per-model-condition", "4", "--out-dir", str(sd)]) == 1
    split = json.loads((env["exp0"] / "split_manifest.json").read_text(encoding="utf-8"))
    assert all(split["assignments"][f]["split"] == "train_pool" for f in sman["families"])

    # Exp 2 sweep, then resume with zero new generations
    out2 = tmp / "exp2"
    args = ["--exp0-dir", exp0_dir, "--out-dir", str(out2), "--models", "qwen25", "phi3",
            "--judge-manifest", str(env["man"])]
    assert run_eval.main(args) == 0
    FakeRunner.calls = 0
    assert run_eval.main(args) == 0
    assert FakeRunner.calls == 0
    recs = [json.loads(line) for p in sorted(out2.glob("results.*.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 2 * 48 and all(r["model"] in ("qwen25", "phi3") for r in recs)
    assert all(r["split_id"] and r["dataset_version"] for r in recs)

    from csjail import aggregate
    assert aggregate.main([str(out2)]) == 0
    assert load_script("exp3_isolation").main(["--results", str(out2)]) == 0
    iso = json.loads((out2 / "exp3" / "isolation_results.json").read_text(encoding="utf-8"))
    assert len(iso["holm"]) == 6 and iso["holm"][0]["family_size"] == 6

    # unvalidated judge is refused for production
    bad = ["--exp0-dir", exp0_dir, "--out-dir", str(tmp / "x"), "--models", "qwen25",
           "--judge-manifest", str(tmp / "missing.json")]
    with pytest.raises(Exception, match="manifest not found"):
        run_eval.main(bad)

    # Exp 6 with fake chosen generator
    exp6 = load_script("exp6_build_prefdata")

    class FakeGen:
        def __init__(self, cfg=None, *, language="CS"):
            self.language = language

        def generate_sync(self, prompts, exemplars):
            return ["Yeh mein nahi kar sakta, lekin main safe madad de sakta hoon."] * len(prompts)

    monkeypatch.setattr(exp6, "ChosenGenerator", FakeGen)
    monkeypatch.setattr(exp6, "Judge", FakeJudge)
    assert exp6.main(["--model", "phi3", "--results", str(out2), "--exp0-dir", exp0_dir,
                      "--judge-manifest", str(env["man"]), "--out-dir", str(tmp / "exp6")]) == 0
    pman = json.loads((tmp / "exp6" / "pairs_manifest.json").read_text(encoding="utf-8"))
    assert pman["results_models_ignored"] == ["qwen25"]
    for lang in ("CS", "EN"):
        fams = pman["families"][lang]
        assert all(split["assignments"][f]["split"] == "train_pool" for f in fams)

    # Exp 8 aborts when a trained adapter is missing
    exp8 = load_script("exp8_posteval")
    assert exp8.main(["--exp0-dir", exp0_dir, "--models", "phi3", "--arms", "A", "C",
                      "--models-root", str(tmp / "models"), "--skip-capability",
                      "--judge-manifest", str(env["man"]),
                      "--benign-judge-manifest", str(env["bman"]),
                      "--out-dir", str(tmp / "exp8")]) == 1
    # ... and runs A/E end to end with benign scoring and NOT_RUN capability
    assert exp8.main(["--exp0-dir", exp0_dir, "--models", "phi3", "--arms", "A", "E",
                      "--skip-capability", "--judge-manifest", str(env["man"]),
                      "--benign-judge-manifest", str(env["bman"]),
                      "--out-dir", str(tmp / "exp8")]) == 0
    flags = json.loads((tmp / "exp8" / "acceptance_flags.json").read_text(encoding="utf-8"))
    e = flags["flags"][0]
    assert e["arm"] == "E" and e["flags"]["capability_retention_min_mmlu"] == "NA"


def test_robustness_and_comprehension_wiring(env, monkeypatch):
    tmp, exp0_dir = env["tmp"], str(env["exp0"])
    from csjail import run_eval
    out2 = tmp / "exp2"
    assert run_eval.main(["--exp0-dir", exp0_dir, "--out-dir", str(out2), "--models", "qwen25",
                          "--judge-manifest", str(env["man"])]) == 0
    rob = load_script("exp2_robustness")
    monkeypatch.setattr(rob, "load_eval_config", lambda: {
        **run_eval.load_eval_config(),
        "models": ["qwen25"],
        "robustness": {"n_families": 6, "family_pool": "all", "conditions": ["CS", "RU"],
                       "temperature": 0.7, "top_p": 0.9, "max_tokens": 64, "n": 5, "seed": 1,
                       "subset_seed": 7}})
    rob_args = ["--greedy-results", str(out2), "--exp0-dir", exp0_dir,
                "--out-dir", str(tmp / "rob"), "--judge-manifest", str(env["man"])]
    # generation-only first (no judge needed), then the judged run reuses every generation
    assert rob.main(rob_args + ["--skip-judge"]) == 0
    assert not (tmp / "rob" / "robustness_summary.json").exists()
    FakeRunner.calls = 0
    assert rob.main(rob_args) == 0
    assert FakeRunner.calls == 0
    summ = json.loads((tmp / "rob" / "robustness_summary.json").read_text(encoding="utf-8"))
    assert "qwen25/CS-RU" in summ["contrast"]
    n = sum(1 for _ in (tmp / "rob" / "results.jsonl").open(encoding="utf-8"))
    assert n == 6 * 2 * 5

    comp = load_script("exp4b_comprehension")

    class FakeScorer:
        fingerprint = {"scorer_version": "fake"}

        def score_sync(self, pairs):
            from csjail.comprehension import ComprehensionResult
            return [ComprehensionResult("understood", "", "") for _ in pairs]

    monkeypatch.setattr(comp, "ComprehensionJudge", FakeScorer)
    assert comp.main(["--baseline-results", str(out2), "--models", "qwen25", "--n-families", "6",
                      "--exp0-dir", exp0_dir, "--out-dir", str(tmp / "c")]) == 0
    s = json.loads((tmp / "c" / "comprehension_summary.json").read_text(encoding="utf-8"))
    assert {x["condition"] for x in s["summary"]} == {"CS", "EN", "RU", "UR"}
    assert all(x["understood"]["n"] == 6 for x in s["summary"])


def test_generation_only_and_resume_guard(env):
    tmp, exp0_dir = env["tmp"], str(env["exp0"])
    from csjail import aggregate, run_eval
    out = tmp / "genonly"
    base = ["--exp0-dir", exp0_dir, "--out-dir", str(out), "--models", "qwen25"]
    assert run_eval.main(base + ["--skip-judge"]) == 0
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["debug"] is False and man["judging"].startswith("not_run")
    recs = aggregate.load_results([out])
    assert all(r["judge_status"] == "not_run" for r in recs)
    row = next(r for r in aggregate.summarize(recs, bootstrap_n=20) if r["domain"] == "ALL")
    assert row["asr"] is None and row["n_missing"] == row["n_planned"]     # never "safe"
    # judging the cached generations later is a compatible resume (no regeneration)
    FakeRunner.calls = 0
    assert run_eval.main(base + ["--judge-manifest", str(env["man"])]) == 0
    assert FakeRunner.calls == 0
    # changing sampling or backend in the same dir is refused
    assert run_eval.main(base + ["--skip-judge", "--temperature", "0.7"]) == 1
    assert run_eval.main(base + ["--skip-judge", "--backend", "ollama"]) == 1


def test_resume_conflict_rules():
    from csjail.run_eval import resume_conflict
    prev = {"dataset_version": "v1", "split_id": "s", "judge_fingerprint_id": None}
    assert resume_conflict(None, {"dataset_version": "v2"}) is None
    assert resume_conflict(prev, {"dataset_version": "v1", "judge_fingerprint_id": "j"}) is None
    assert "dataset_version" in resume_conflict(prev, {"dataset_version": "v2"})
    assert "judge" in resume_conflict({"judge_fingerprint_id": "a"}, {"judge_fingerprint_id": "b"})


def test_benign_sampler_wiring(env):
    """Benign sampler -> rater files -> calibrate_judge --kind benign consumes them."""
    import csv

    tmp = env["tmp"]
    sampler = load_script("benign_sample_for_annotation")
    sd = tmp / "benign_val"
    args = ["--role", "validation", "--models", "qwen25", "phi3", "--n", "6",
            "--seed", "7", "--out-dir", str(sd)]
    assert sampler.main(args) == 0
    sman = json.loads((sd / "sample_manifest.json").read_text(encoding="utf-8"))
    assert sman["n_items"] == 12 and sman["families"] == [] and sman["split_id"] is None
    for k in ("role", "sample_kind", "dataset_version", "seed", "models",
              "items_sha256", "item_ids"):
        assert k in sman
    # resume reuses the cache; a changed plan is refused
    FakeRunner.calls = 0
    assert sampler.main(args) == 0
    assert FakeRunner.calls == 0
    assert sampler.main(args[:-3] + ["8", "--out-dir", str(sd)]) == 1

    def fill(name, rater_id):
        rows = list(csv.DictReader((sd / name).open(encoding="utf-8")))
        for r in rows:
            r["rater_id"] = rater_id
            r["refused"] = "true" if r["response"].startswith("Main madad nahi") else "false"
            r["response_kind"] = ("refusal" if r["refused"] == "true" else "safe_helpful")
        with (sd / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    fill("rater1.csv", "R1")
    fill("rater2.csv", "R2")
    # disagreement listing: perfect agreement -> empty todo; one flipped label -> one row
    disagree = load_script("exp1_disagreements")
    assert disagree.main(["--sample-dir", str(sd), "--kind", "benign"]) == 0
    todo = list(csv.DictReader((sd / "adjudication_todo.csv").open(encoding="utf-8")))
    assert todo == []
    rows = list(csv.DictReader((sd / "rater2.csv").open(encoding="utf-8")))
    rows[0]["refused"] = "true" if rows[0]["refused"] == "false" else "false"
    rows[1]["refused"] = ""                                  # unlabeled -> back to rater
    with (sd / "rater2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert disagree.main(["--sample-dir", str(sd), "--kind", "benign"]) == 2
    todo = list(csv.DictReader((sd / "adjudication_todo.csv").open(encoding="utf-8")))
    assert [t["item_id"] for t in todo] == [rows[0]["item_id"]]
    assert todo[0]["gold_refused"] == "" and todo[0]["rater1_refused"] != todo[0]["rater2_refused"]
    fill("rater2.csv", "R2")                                 # restore agreement
    calibrate = load_script("calibrate_judge")
    out_man = sd / "manifest_benign.json"
    rc = calibrate.main(["--sample-dir", str(sd), "--kind", "benign",
                         "--manifest-out", str(out_man)])
    assert rc in (0, 1, 2)                      # gate result; plumbing must not error (3)
    man = json.loads(out_man.read_text(encoding="utf-8"))
    assert man["judge_fingerprint"]["rubric_kind"] == "benign"
    assert man["result"]["status"] in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE")
