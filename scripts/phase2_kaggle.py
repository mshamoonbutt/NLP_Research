#!/usr/bin/env python3
"""Phase 2 on Kaggle (notebooks/kaggle_phase2.ipynb): GPU smoke, then Exp 7 + Exp 8 per model.

    python scripts/phase2_kaggle.py smoke --model phi3 --step train     # GPU: DPO on 8 harmless pairs
    python scripts/phase2_kaggle.py smoke --model phi3 --step serve     # GPU: that adapter in vLLM
    python scripts/phase2_kaggle.py run --model phi3                    # GPU: all adapters + Exp 8 generation
    python scripts/phase2_kaggle.py pack                                # laptop: private upload bundle

smoke (PROTOCOL §9): trains a LoRA-DPO adapter on 8 harmless synthetic pairs with the production
config (fp16 compute on a T4, resolved LoRA modules) for one epoch, then serves it through vLLM;
two separate processes so the training memory is gone before vLLM loads. Writes
outputs/phase2_smoke/<model>/smoke_<step>.json.

run (one model, resumable; budgets from outputs/exp6/review/verified_summary.json):
  Exp 7  C and B_ext at the main budget N for each seed (42 = no suffix, others _s<seed>);
         C at each n-curve budget below N; C and B_ext under the D6 ablation split at its budget.
  Exp 8  generation only (--skip-judge) into outputs/exp8/<tag>__<model>: A B_ext C E at n<N>;
         A B_ext C per extra seed; A C per n-curve budget; A B_ext C for the ablation. Arm A's
         greedy generations are copied in from the first tag (identical cache keys), not redone.
  Finished adapters (training_manifest.json) and evaluations (generation-only run_manifest.json)
  are skipped, so a rerun continues where a lost session stopped.

pack: outputs/phase2_upload.zip with the verified pair sets, the review summary and the external
English pairs (all text-bearing, gitignored) -> upload as a PRIVATE Kaggle dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
AB = "D6"
AB_SPLIT = f"outputs/exp9/ablation_{AB}/split_manifest.json"
SMOKE_PAIRS = [   # harmless: polite helpful answer preferred over a curt one
    ("Suggest a name for a pet cat.", "How about Mishmish or Biscuit? Both are short and friendly.", "No."),
    ("How do I boil an egg?", "Put the egg in boiling water for 8-10 minutes, then cool it in cold water.", "Figure it out."),
    ("Recommend a book for learning Python.", "Try 'Automate the Boring Stuff with Python'; it is practical and free online.", "Whatever."),
    ("What is a good way to start the morning?", "A glass of water, a short walk and a simple breakfast work well for many people.", "Don't care."),
    ("How can I save money on groceries?", "Plan meals for the week, buy staples in bulk and compare unit prices.", "Not my problem."),
    ("Give me a tip for studying.", "Study in 25-minute blocks with short breaks, and test yourself instead of rereading.", "Just study."),
    ("What should I pack for a picnic?", "Sandwiches, fruit, water, a blanket and some napkins are a good start.", "Food."),
    ("How do I water a succulent?", "Water deeply only when the soil is completely dry, usually every one to two weeks.", "With water."),
]


def sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def cmd_smoke(model: str, step: str) -> int:
    import yaml

    from csjail.models import SamplingConfig, SLMRunner, resolve

    spec = resolve(model)
    out = ROOT / "outputs" / "phase2_smoke" / model
    if step == "train":
        from csjail.train_dpo import train_dpo

        cfg = dict(yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))["dpo"], epochs=1)
        pairs = [{"prompt": p, "chosen": c, "rejected": r} for p, c, r in SMOKE_PAIRS]
        train_dpo(spec.hf_id, pairs, cfg, str(out / "adapter"), revision=spec.revision,
                  manifest_extra={"smoke": True, "model_key": model})
        man = json.loads((out / "adapter" / "training_manifest.json").read_text(encoding="utf-8"))
        losses = [h.get("loss", h.get("train_loss")) for h in man["log_history"]
                  if "loss" in h or "train_loss" in h]
        res = {"step": "train", "ok": bool(losses) and all(x == x and abs(x) != float("inf") for x in losses),
               "compute_dtype": cfg.get("compute_dtype"),
               "adapted_modules": man["adapted_modules"], "trainable_params": man["trainable_params"],
               "losses": losses, "pins_mismatched": man["pins_mismatched"]}
    else:
        runner = SLMRunner(spec, adapter_path=str(out / "adapter"), require_pinned=True)
        try:
            outs = runner.generate_text(["Suggest a name for a pet dog.", "How do I make tea?"],
                                        SamplingConfig(temperature=0.0, max_tokens=48))
        finally:
            runner.shutdown()
        res = {"step": "serve", "ok": all(o.strip() for o in outs), "n_chars": [len(o) for o in outs],
               "adapter_sha256": runner.adapter_sha256}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"smoke_{step}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[smoke] {model}/{step}: {res}")
    return 0 if res["ok"] else 1


def plan(model: str, n: int, nab: int, seeds: list[int], ncurve: list[int]):
    """(training jobs [(adapter name, exp7 args)], evaluations [(exp8 tag, arms, split)]).
    Exp 8 finds a trained arm's adapter at outputs/models/<arm>_<model>_<tag>."""
    ver, abl = f"outputs/exp6/{model}_verified", f"outputs/exp6/{model}_ablation_{AB}"
    jobs = []   # (adapter name, exp7 args)
    for s in seeds:
        suffix = "" if s == 42 else f"_s{s}"
        for arm in ("C", "B_ext"):
            jobs.append((f"{arm}_{model}_n{n}{suffix}", ["--arm", arm, "--budget", str(n), "--pairs-dir", ver]
                         + ([] if s == 42 else ["--seed", str(s)])
                         + (["--naturalness-csv", f"{ver}/naturalness.csv"] if arm == "C" else [])))
    for b in (b for b in ncurve if b < n):
        jobs.append((f"C_{model}_n{b}", ["--arm", "C", "--budget", str(b), "--pairs-dir", ver,
                                         "--naturalness-csv", f"{ver}/naturalness.csv"]))
    for arm in ("C", "B_ext"):
        jobs.append((f"{arm}_{model}_n{nab}_ablation_{AB}",
                     ["--arm", arm, "--budget", str(nab), "--pairs-dir", abl, "--split-manifest", AB_SPLIT,
                      "--tag", f"ablation_{AB}"] + (["--naturalness-csv", f"{abl}/naturalness.csv"] if arm == "C" else [])))
    evals = ([(f"n{n}", ["A", "B_ext", "C", "E"], None)]
             + [(f"n{n}_s{s}", ["A", "B_ext", "C"], None) for s in seeds if s != 42]
             + [(f"n{b}", ["A", "C"], None) for b in ncurve if b < n]
             + [(f"n{nab}_ablation_{AB}", ["A", "B_ext", "C"], AB_SPLIT)])
    return jobs, evals


def cmd_run(model: str, seeds: list[int], ncurve: list[int]) -> int:
    summ = json.loads((ROOT / "outputs/exp6/review/verified_summary.json").read_text(encoding="utf-8"))
    jobs, evals = plan(model, summ["budget_main"][model], summ["budget_ablation"][model], seeds, ncurve)
    for name, args in jobs:
        if (ROOT / "outputs" / "models" / name / "training_manifest.json").exists():
            print(f"[run] skip trained {name}", flush=True)
            continue
        sh([sys.executable, "scripts/exp7_train_arms.py", "--model", model] + args)

    first = ROOT / "outputs" / "exp8" / f"{evals[0][0]}__{model}"
    for tag, arms, split in evals:
        out = ROOT / "outputs" / "exp8" / f"{tag}__{model}"
        pm = out / "run_manifest.json"
        if pm.exists() and set(arms) <= set(json.loads(pm.read_text(encoding="utf-8")).get("arms", [])):
            print(f"[run] skip evaluated {out.name}", flush=True)
            continue
        if out != first and (first / "generations.jsonl").exists():   # reuse arm A's cached generations
            for rel in ("generations.jsonl", "benign/generations.jsonl"):
                if (first / rel).exists() and not (out / rel).exists():
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(first / rel, out / rel)
        sh([sys.executable, "scripts/exp8_posteval.py", "--models", model, "--arms", *arms, "--tag", tag,
            "--skip-judge", "--out-dir", str(out)] + (["--split-manifest", split] if split else []))
    print(f"[run] {model} DONE", flush=True)
    return 0


def cmd_pack() -> int:
    files = [ROOT / "outputs/exp6/review/verified_summary.json"]
    for m in ("phi3", "llama32"):
        for tag in ("verified", f"ablation_{AB}"):
            files += sorted((ROOT / "outputs" / "exp6" / f"{m}_{tag}").glob("*"))
    files += sorted((ROOT / "data").glob("pref_pairs_en_external*"))
    missing = [str(f) for f in files[:1] if not f.exists()]
    if missing or len(files) < 10:
        print(f"FAIL: run `exp6_review.py apply` first (missing {missing or 'verified pair sets'})", file=sys.stderr)
        return 1
    out = ROOT / "outputs" / "phase2_upload.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"[pack] {len(files)} files -> {out} ({out.stat().st_size / 1e6:.1f} MB); upload as a PRIVATE Kaggle dataset")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--model", required=True)
    s.add_argument("--step", required=True, choices=["train", "serve"])
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    r.add_argument("--ncurve", nargs="*", type=int, default=[25, 50])
    sub.add_parser("pack")
    args = ap.parse_args(argv)
    if args.cmd == "smoke":
        return cmd_smoke(args.model, args.step)
    if args.cmd == "run":
        return cmd_run(args.model, args.seeds, args.ncurve)
    return cmd_pack()


if __name__ == "__main__":
    sys.exit(main())
