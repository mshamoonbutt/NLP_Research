#!/usr/bin/env python3
"""Phase 2 on Kaggle (notebooks/kaggle_phase2.ipynb): GPU smoke, then Exp 7 + Exp 8 per model.

    python scripts/phase2_kaggle.py smoke --model phi3 --step train     # GPU: DPO on 8 harmless pairs
    python scripts/phase2_kaggle.py smoke --model phi3 --step serve     # GPU: that adapter in vLLM
    python scripts/phase2_kaggle.py parts --n-parts 3 [--part 2]        # the split over Kaggle accounts
    python scripts/phase2_kaggle.py queue --part 2 --n-parts 3 --gpu 0 --epochs 4   # GPU: this GPU's units
    python scripts/phase2_kaggle.py decide --models phi3 llama32 --epochs 4      # laptop: the learning-check record
    python scripts/phase2_kaggle.py pack                                # laptop: private upload bundle
    python scripts/phase2_kaggle.py judge                               # laptop: judge every Exp 8 folder

smoke (PROTOCOL §9): trains a LoRA-DPO adapter on 8 harmless synthetic pairs with the production
config (fp16 compute on a T4, resolved LoRA modules) for one epoch, then serves it through vLLM;
two separate processes so the training memory is gone before vLLM loads. Writes
outputs/phase2_smoke/<model>/smoke_<step>.json.

Units (budgets from outputs/exp6/review/verified_summary.json): one per Exp 8 folder
outputs/exp8/<tag>__<model> -- its arms (A and E only in the main folder n<N>; the others are
compared with it) and the adapters they need (Exp 7: C and B_ext per seed, C per n-curve budget,
C and B_ext under the D6 ablation split) -- plus a training-only unit that records the learning
check at the ladder's first value (outputs/models_e2, never evaluated). Units share nothing, so
`parts` deals them over 1-3 Kaggle accounts x 2 GPUs (Llama on the first queues, so a Phi-3-only
part needs no HF token) and `queue` runs one GPU's share. Epochs: the learning check of the first
full run (2026-10-10: 2 epochs failed, 4 passed) fixes them; the main units re-record the check
(outputs/phase2_gate/<model>_e<E>.json). Finished adapters (manifest + weights) and folders
(generation-only run_manifest.json at these epochs) are skipped, so a rerun resumes.

pack: outputs/phase2_upload.zip with the verified pair sets, the review summary and the external
English pairs (all text-bearing, gitignored) -> upload as a PRIVATE Kaggle dataset.
"""
from __future__ import annotations

import argparse
import json
import os
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
SEEDS, NCURVE = [42, 43, 44], [25, 50]
GATED = {"llama32"}                      # needs an HF_TOKEN with Llama-3.2 access
MIN_PER_ADAPTER, MIN_PER_ARM = 6, 30     # planning estimates on a T4 (4 epochs; one Exp 8 arm incl. MMLU)
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
    Exp 8 finds a trained arm's adapter at outputs/models/<arm>_<model>_<tag>. The untrained arm A
    and the prompt arm E are generated once, in the main folder n<N>; every comparison with A
    (phase2_analysis.py) uses that folder."""
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
             + [(f"n{n}_s{s}", ["B_ext", "C"], None) for s in seeds if s != 42]
             + [(f"n{b}", ["C"], None) for b in ncurve if b < n]
             + [(f"n{nab}_ablation_{AB}", ["B_ext", "C"], AB_SPLIT)])
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


def trained_at(name: str, epochs: int, root: str = "models") -> bool:
    """Finished at `epochs`: the manifest (written last) AND the weights (a restored outputs zip
    carries manifests without them)."""
    d = ROOT / "outputs" / root / name
    tm = d / "training_manifest.json"
    return (tm.exists() and any((d / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin"))
            and json.loads(tm.read_text(encoding="utf-8"))["dpo_config"].get("epochs") == epochs)


def learning_cfg() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))["learning_check"]


def budgets(model: str) -> tuple[int, int]:
    summ = json.loads((ROOT / "outputs/exp6/review/verified_summary.json").read_text(encoding="utf-8"))
    return summ["budget_main"][model], summ["budget_ablation"][model]


def units(model: str, seeds: list[int] = SEEDS, ncurve: list[int] = NCURVE) -> list[dict]:
    """Independent pieces of work: one per Exp 8 folder (its arms + the adapters they need), plus the
    learning-check record at the ladder's first value (seed-42 C and B_ext, training only)."""
    n, nab = budgets(model)
    jobs, evals = plan(model, n, nab, seeds, ncurve)
    jobs = dict(jobs)
    first = learning_cfg()["epochs_ladder"][0]
    out = [{"id": f"gate_e{first}__{model}", "model": model, "tag": None, "arms": [], "split": None,
            "epochs": first, "root": f"models_e{first}",
            "train": [(g, jobs[g]) for g in (f"B_ext_{model}_n{n}", f"C_{model}_n{n}")]}]
    for tag, arms, split in evals:
        out.append({"id": f"{tag}__{model}", "model": model, "tag": tag, "arms": arms, "split": split,
                    "epochs": None, "root": "models",
                    "train": [(f"{a}_{model}_{tag}", jobs[f"{a}_{model}_{tag}"]) for a in arms if a in ("B_ext", "C")]})
    for u in out:
        u["minutes"] = MIN_PER_ADAPTER * len(u["train"]) + MIN_PER_ARM * len(u["arms"])
    return out


def assign(n_parts: int, seeds: list[int] = SEEDS, ncurve: list[int] = NCURVE) -> dict[tuple[int, int], list[dict]]:
    """(part, gpu) -> units over n_parts accounts x 2 GPUs. Llama takes the first n_parts queues and
    Phi-3 the rest (a part without Llama needs no HF token); within a model the longest unit goes to
    the least-loaded queue. Deterministic, so every account computes the same split."""
    queues = [(q // 2 + 1, q % 2) for q in range(2 * n_parts)]
    out = {q: [] for q in queues}
    for model, qs in (("llama32", queues[:n_parts]), ("phi3", queues[n_parts:])):
        for u in sorted(units(model, seeds, ncurve), key=lambda u: (-u["minutes"], u["id"])):
            out[min(qs, key=lambda q: (sum(x["minutes"] for x in out[q]), q))].append(u)
    return out


def record_gate(model: str, names: list[str], epochs: int, root: str) -> dict:
    """outputs/phase2_gate/<model>_e<epochs>.json: the declared check on these adapters' training logs."""
    lc = learning_cfg()
    res = {nm: learning_check(json.loads((ROOT / "outputs" / root / nm / "training_manifest.json")
                                         .read_text(encoding="utf-8")), epochs, lc) for nm in names}
    rec = {"model": model, "epochs": epochs, "criteria": lc, "adapters": res, "pass": all(r["pass"] for r in res.values())}
    out = ROOT / "outputs" / "phase2_gate" / f"{model}_e{epochs}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[gate] {model} epochs {epochs}: {'PASS' if rec['pass'] else 'FAIL'} {res}", flush=True)
    return rec


def evaluated(out: Path, arms: list[str], epochs: int) -> bool:
    pm, meta = out / "run_manifest.json", out / "phase2_meta.json"
    return (pm.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("epochs") == epochs
            and set(arms) <= set(json.loads(pm.read_text(encoding="utf-8")).get("arms", [])))


def cmd_queue(part: int, n_parts: int, gpu: int, epochs: int) -> int:
    """Kaggle: run this GPU's units in order (resumable: finished adapters and folders are skipped)."""
    mine = assign(n_parts)[(part, gpu)]
    print(f"[queue] part {part}/{n_parts} gpu {gpu}: {[u['id'] for u in mine]}", flush=True)
    for u in mine:
        model, ep = u["model"], u["epochs"] or epochs
        for name, args in u["train"]:
            if trained_at(name, ep, u["root"]):
                print(f"[queue] skip trained {u['root']}/{name} (epochs {ep})", flush=True)
                continue
            sh([sys.executable, "scripts/exp7_train_arms.py", "--model", model, "--epochs", str(ep),
                "--out-root", str(ROOT / "outputs" / u["root"])] + args)
        if u["tag"] is None or u["tag"] == f"n{budgets(model)[0]}":
            rec = record_gate(model, [nm for nm, _ in u["train"]], ep, u["root"])
            if u["tag"] and not rec["pass"]:
                print(f"WARNING: the learning check at {ep} epochs FAILED on this rerun ({model}); the epochs were "
                      "fixed by the first run -- report this; continuing", flush=True)
        if u["tag"] is None:
            continue
        out = ROOT / "outputs" / "exp8" / f"{u['tag']}__{model}"
        if evaluated(out, u["arms"], epochs):
            print(f"[queue] skip evaluated {out.name}", flush=True)
            continue
        sh([sys.executable, "scripts/exp8_posteval.py", "--models", model, "--arms", *u["arms"], "--tag", u["tag"],
            "--skip-judge", "--out-dir", str(out)] + (["--split-manifest", u["split"]] if u["split"] else [])
           + ([] if "A" in u["arms"] else ["--no-baseline"]))
        (out / "phase2_meta.json").write_text(json.dumps({"epochs": epochs, "part": part, "n_parts": n_parts}),
                                              encoding="utf-8")
    print(f"[queue] part {part} gpu {gpu} DONE", flush=True)
    return 0


def cmd_parts(n_parts: int, part: int | None = None) -> int:
    """Print the split; with --part, exit 3 when that part runs Llama but HF_TOKEN is not set."""
    a = assign(n_parts)
    for (p, g), us in sorted(a.items()):
        print(f"part {p} gpu {g}: ~{sum(u['minutes'] for u in us)} min | " + ", ".join(u["id"] for u in us))
    if part is not None:
        models = {u["model"] for (p, _), us in a.items() if p == part for u in us}
        if models & GATED and not os.environ.get("HF_TOKEN"):
            print(f"FAIL: part {part} runs {sorted(models & GATED)}: attach the HF_TOKEN secret (Llama-3.2 access)",
                  file=sys.stderr)
            return 3
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


def restore_previous(search: Path) -> list[str]:
    """Kaggle resume: an earlier version's output attached as input (its phase2_*.zip files and/or
    its NLP_Research tree) is copied in, so `run` skips the adapters and evaluations it finished
    and the notebook reuses its learning-check decision. Trees go last: they are newer than zips."""
    got = []
    for name in ("phase2_adapters*.zip", "phase2_outputs*.zip"):
        for z in sorted(search.glob(f"**/{name}")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ROOT)
            got.append(str(z))
    for d in ("models*", "exp8", "phase2_gate"):
        for src in sorted(search.glob(f"**/outputs/{d}")):
            if src.is_dir():
                shutil.copytree(src, ROOT / "outputs" / src.name, dirs_exist_ok=True)
                got.append(str(src))
    return got


def cmd_stage(search: Path) -> int:
    """Kaggle: find the private bundle anywhere under `search` (the zip itself, or the tree Kaggle
    unpacked it into, at any depth) and copy it into the repo; list what is there if not found.
    Then restore an earlier version's outputs if one is attached (restore_previous)."""
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
    prev = restore_previous(search)
    print(f"[stage] resumed from {prev}" if prev else "[stage] no earlier outputs attached: fresh run", flush=True)
    return 0


def cmd_judge(models: list[str], seeds: list[int], ncurve: list[int]) -> int:
    """Laptop, after every part's phase2_outputs_part*.zip is unzipped into the repo: judge every
    Exp 8 folder (exp8_posteval --judge-only; resumable, each folder caches its judgments). A
    missing folder aborts before any judging."""
    todo = [u for m in models for u in units(m, seeds, ncurve) if u["tag"]]
    missing = [u["id"] for u in todo if not (ROOT / "outputs" / "exp8" / u["id"] / "run_manifest.json").exists()]
    if missing:
        print(f"FAIL: not generated yet (unzip every part first): {missing}", file=sys.stderr)
        return 1
    for u in todo:
        sh([sys.executable, "scripts/exp8_posteval.py", "--judge-only", "--models", u["model"], "--arms", *u["arms"],
            "--tag", u["tag"], "--out-dir", str(ROOT / "outputs" / "exp8" / u["id"])]
           + (["--split-manifest", u["split"]] if u["split"] else []) + ([] if "A" in u["arms"] else ["--no-baseline"]))
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
    q = sub.add_parser("queue")
    q.add_argument("--part", required=True, type=int)
    q.add_argument("--n-parts", required=True, type=int, choices=[1, 2, 3])
    q.add_argument("--gpu", required=True, type=int, choices=[0, 1])
    q.add_argument("--epochs", required=True, type=int)
    pa = sub.add_parser("parts")
    pa.add_argument("--n-parts", required=True, type=int, choices=[1, 2, 3])
    pa.add_argument("--part", type=int, default=None)
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
    if args.cmd == "queue":
        if not 1 <= args.part <= args.n_parts:
            ap.error(f"--part must be 1..{args.n_parts}")
        return cmd_queue(args.part, args.n_parts, args.gpu, args.epochs)
    if args.cmd == "parts":
        return cmd_parts(args.n_parts, args.part)
    if args.cmd == "decide":
        return cmd_decide(args.models, args.epochs)
    if args.cmd == "judge":
        return cmd_judge(args.models, args.seeds, args.ncurve)
    return cmd_pack()


if __name__ == "__main__":
    sys.exit(main())
