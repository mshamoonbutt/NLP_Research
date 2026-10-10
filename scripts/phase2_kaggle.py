#!/usr/bin/env python3
"""Phase 2 on Kaggle (notebooks/kaggle_phase2.ipynb): GPU smoke, then Exp 7 + Exp 8 per model.

    python scripts/phase2_kaggle.py smoke --model phi3 --step train     # GPU: DPO on 8 harmless pairs
    python scripts/phase2_kaggle.py smoke --model phi3 --step serve     # GPU: that adapter in vLLM
    python scripts/phase2_kaggle.py run --model phi3 --phase gate --epochs 2   # GPU: seed-42 C/B_ext + learning check
    python scripts/phase2_kaggle.py decide --models phi3 llama32 --epochs 2      # exit 0 = all passed
    python scripts/phase2_kaggle.py run --model phi3 --phase rest --epochs 2   # GPU: the rest + Exp 8 generation
    python scripts/phase2_kaggle.py pack                                # laptop: private upload bundle
    python scripts/phase2_kaggle.py judge                               # laptop: judge every Exp 8 folder

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
import statistics
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


def learning_check(manifest: dict, epochs: int, lc: dict) -> dict:
    """The declared check (configs/dpo.yaml learning_check) on one adapter's TRAINING log: mean
    loss and mean rewards/accuracies over the steps of the final epoch."""
    steps = [h for h in manifest.get("log_history", []) if "loss" in h]
    final = [h for h in steps if h.get("epoch", 0) > epochs - 1] or steps[-1:]
    acc = [h["rewards/accuracies"] for h in final if "rewards/accuracies" in h]
    loss = statistics.fmean(h["loss"] for h in final) if final else None
    acc = statistics.fmean(acc) if acc else None
    ok = (loss is not None and acc is not None and loss <= lc["max_final_epoch_loss"]
          and acc >= lc["min_final_epoch_reward_accuracy"])
    return {"epochs": epochs, "final_epoch_loss": loss, "final_epoch_reward_accuracy": acc,
            "n_final_steps": len(final), "pass": ok}


def trained_at(name: str, epochs: int) -> bool:
    tm = ROOT / "outputs" / "models" / name / "training_manifest.json"
    return tm.exists() and json.loads(tm.read_text(encoding="utf-8"))["dpo_config"].get("epochs") == epochs


def cmd_run(model: str, seeds: list[int], ncurve: list[int], phase: str, epochs: int) -> int:
    """phase gate: the seed-42 C and B_ext adapters + the learning check (outputs/phase2_gate/
    <model>_e<epochs>.json); phase rest: every other adapter and all Exp 8 generations at `epochs`.
    An adapter or evaluation made at a different epochs value is redone, never reused."""
    import yaml

    summ = json.loads((ROOT / "outputs/exp6/review/verified_summary.json").read_text(encoding="utf-8"))
    n = summ["budget_main"][model]
    jobs, evals = plan(model, n, summ["budget_ablation"][model], seeds, ncurve)
    gate = [f"C_{model}_n{n}", f"B_ext_{model}_n{n}"]
    for name, args in jobs:
        if phase == "gate" and name not in gate:
            continue
        if trained_at(name, epochs):
            print(f"[run] skip trained {name} (epochs {epochs})", flush=True)
            continue
        sh([sys.executable, "scripts/exp7_train_arms.py", "--model", model, "--epochs", str(epochs)] + args)
    if phase == "gate":
        lc = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))["learning_check"]
        res = {name: learning_check(json.loads((ROOT / "outputs" / "models" / name / "training_manifest.json")
                                               .read_text(encoding="utf-8")), epochs, lc) for name in gate}
        out = ROOT / "outputs" / "phase2_gate" / f"{model}_e{epochs}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": model, "epochs": epochs, "criteria": lc, "adapters": res,
                                   "pass": all(r["pass"] for r in res.values())}, indent=2), encoding="utf-8")
        print(f"[gate] {model} epochs {epochs}: {res}", flush=True)
        return 0

    first = ROOT / "outputs" / "exp8" / f"{evals[0][0]}__{model}"
    for tag, arms, split in evals:
        out = ROOT / "outputs" / "exp8" / f"{tag}__{model}"
        pm, meta = out / "run_manifest.json", out / "phase2_meta.json"
        if (pm.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("epochs") == epochs
                and set(arms) <= set(json.loads(pm.read_text(encoding="utf-8")).get("arms", []))):
            print(f"[run] skip evaluated {out.name}", flush=True)
            continue
        if out != first and (first / "generations.jsonl").exists():   # reuse arm A's cached generations
            for rel in ("generations.jsonl", "benign/generations.jsonl"):
                if (first / rel).exists() and not (out / rel).exists():
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(first / rel, out / rel)
        sh([sys.executable, "scripts/exp8_posteval.py", "--models", model, "--arms", *arms, "--tag", tag,
            "--skip-judge", "--out-dir", str(out)] + (["--split-manifest", split] if split else []))
        meta.write_text(json.dumps({"epochs": epochs}), encoding="utf-8")
    print(f"[run] {model} DONE", flush=True)
    return 0


def cmd_decide(models: list[str], epochs: int) -> int:
    """Exit 0 when every model's gate at `epochs` passed, 10 otherwise; records the decision."""
    gates = {m: json.loads((ROOT / "outputs" / "phase2_gate" / f"{m}_e{epochs}.json").read_text(encoding="utf-8"))
             for m in models}
    ok = all(g["pass"] for g in gates.values())
    (ROOT / "outputs" / "phase2_gate" / "decision.json").write_text(json.dumps(
        {"epochs": epochs, "all_pass": ok, "gates": gates}, indent=2), encoding="utf-8")
    print(f"[decide] epochs {epochs}: {'PASS' if ok else 'FAIL'} "
          + str({m: {a: (r['final_epoch_loss'], r['final_epoch_reward_accuracy']) for a, r in g['adapters'].items()}
                 for m, g in gates.items()}), flush=True)
    return 0 if ok else 10


def bundle_files() -> list[Path]:
    """Repo-relative paths the private bundle must provide (pack writes them; stage checks them)."""
    files = [Path("outputs/exp6/review/verified_summary.json")]
    for m in ("phi3", "llama32"):
        for tag in ("verified", f"ablation_{AB}"):
            files += [Path(f"outputs/exp6/{m}_{tag}") / f for f in ("pairs_cs_all.jsonl", "pairs_manifest.json",
                                                                    "naturalness.csv")]
    return files + [Path("data/pref_pairs_en_external.jsonl"), Path(f"data/pref_pairs_en_external_no{AB}.jsonl")]


def cmd_stage(search: Path) -> int:
    """Kaggle: find the private bundle anywhere under `search` (the zip itself, or the tree Kaggle
    unpacked it into, at any depth) and copy it into the repo; list what is there if not found."""
    zips = sorted(search.glob("**/phase2_upload.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(ROOT)
        src = f"zip {zips[0]}"
    else:
        hits = sorted(search.glob("**/outputs/exp6/review/verified_summary.json"))
        if not hits:
            print(f"FAIL: no phase2_upload.zip or outputs/exp6/review/verified_summary.json under {search}. Found:",
                  file=sys.stderr)
            for p in sorted(search.glob("*/*"))[:40] + sorted(search.glob("*/*/*"))[:40]:
                print("  ", p, file=sys.stderr)
            return 1
        root = hits[0].parents[3]          # <root>/outputs/exp6/review/verified_summary.json
        for d in ("outputs", "data"):
            shutil.copytree(root / d, ROOT / d, dirs_exist_ok=True)
        src = f"tree {root}"
    missing = [str(f) for f in bundle_files() if not (ROOT / f).exists()]
    if missing:
        print(f"FAIL: bundle from {src} lacks {missing}", file=sys.stderr)
        return 1
    print(f"[stage] bundle from {src}: all {len(bundle_files())} files in place", flush=True)
    return 0


def cmd_judge(models: list[str], seeds: list[int], ncurve: list[int]) -> int:
    """Laptop: judge every Exp 8 generation folder (exp8_posteval --judge-only; resumable, each
    folder caches its judgments). Arm A is identical in every folder, so the first folder's
    judgment cache is copied into the others first and A is judged once per model."""
    summ = json.loads((ROOT / "outputs/exp6/review/verified_summary.json").read_text(encoding="utf-8"))
    for model in models:
        _, evals = plan(model, summ["budget_main"][model], summ["budget_ablation"][model], seeds, ncurve)
        first = ROOT / "outputs" / "exp8" / f"{evals[0][0]}__{model}"
        for tag, arms, split in evals:
            out = ROOT / "outputs" / "exp8" / f"{tag}__{model}"
            for rel in ("judgments.jsonl", "benign/judgments.jsonl"):
                if out != first and (first / rel).exists() and not (out / rel).exists():
                    shutil.copy2(first / rel, out / rel)
            sh([sys.executable, "scripts/exp8_posteval.py", "--judge-only", "--models", model, "--arms", *arms,
                "--tag", tag, "--out-dir", str(out)] + (["--split-manifest", split] if split else []))
    print(f"[judge] all Exp 8 folders judged for {models}", flush=True)
    return 0


def cmd_pack() -> int:
    files = [ROOT / f for f in bundle_files()]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"FAIL: run `exp6_review.py apply` first (missing {missing})", file=sys.stderr)
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
    r.add_argument("--phase", required=True, choices=["gate", "rest"])
    r.add_argument("--epochs", required=True, type=int)
    d = sub.add_parser("decide")
    d.add_argument("--models", nargs="+", required=True)
    d.add_argument("--epochs", required=True, type=int)
    sub.add_parser("pack")
    j = sub.add_parser("judge")
    j.add_argument("--models", nargs="+", default=["phi3", "llama32"])
    j.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    j.add_argument("--ncurve", nargs="*", type=int, default=[25, 50])
    st = sub.add_parser("stage")
    st.add_argument("--search", default="/kaggle/input")
    args = ap.parse_args(argv)
    if args.cmd == "stage":
        return cmd_stage(Path(args.search))
    if args.cmd == "smoke":
        return cmd_smoke(args.model, args.step)
    if args.cmd == "run":
        return cmd_run(args.model, args.seeds, args.ncurve, args.phase, args.epochs)
    if args.cmd == "decide":
        return cmd_decide(args.models, args.epochs)
    if args.cmd == "judge":
        return cmd_judge(args.models, args.seeds, args.ncurve)
    return cmd_pack()


if __name__ == "__main__":
    sys.exit(main())
