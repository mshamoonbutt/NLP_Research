#!/usr/bin/env python3
"""Stage C driver for Kaggle: judge gates + Exp 2 judging, resumable and time-bounded.

    python scripts/stagec_run.py --input-root /kaggle/input --out-zip /kaggle/working/stageC_outputs.zip

1. Inputs: every bundle under --input-root (or inside a stageC_*.zip there,
   which is how a previous version's output arrives) marked
   CSJAIL_STAGEC_RESUME (a previous run's output) or CSJAIL_STAGEC_UPLOAD (the
   human-made upload) is copied into ./outputs: resume bundles oldest to
   newest, then uploads last. Judge outputs exist only in resume bundles (the
   newest wins); annotation files come from the upload, so corrected
   adjudications in a re-uploaded dataset replace the copies in old bundles.
2. Prints the configured judge and its fingerprint (production judging is
   gated by require_validated_judge: the PASS manifest must match it).
3. Steps, each bounded by the remaining time budget (killed at the limit). Modes:
   --dev-sample-dir      rubric iteration on a development sample (feedback
                         report, never a manifest)
   --validation-predict  two-phase validation phase 1: the frozen judge's
                         predictions on the validation set, no verdict
   --screen              measurement option 2: benign gate -> the configured
                         judge SCREENS Exp 2 main, then robustness (eval_main
                         families only; scripts/exp2_verify.py); humans label
                         the flagged responses + a random audit afterwards
   (default, final)      benign gate (if uploaded) -> Exp 2 main judging (only
                         with the committed PASS harm manifest from the
                         two-phase scoring) -> Exp 2 robustness (after main)
   Every step caches its work as it goes, so a killed or budget-starved step
   loses at most one chunk and the next run continues from the cache.
4. ALWAYS (even after errors) writes --out-zip: a resume bundle with the
   CSJAIL_STAGEC_RESUME marker, STAGEC_STATUS.json and all outputs. Add it as
   an input to the next run to continue.

Step exit codes: 0 done/PASS, 1 FAIL, 2 INSUFFICIENT_EVIDENCE, 3 input error,
4 incomplete (API/budget; resumable), 124 time budget reached (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UPLOAD, RESUME = "CSJAIL_STAGEC_UPLOAD", "CSJAIL_STAGEC_RESUME"
HARM_DIR = "outputs/exp1/validation-v2-merged"     # top-up 1 (EN/UR) + top-up 2 (CS/RU)
BENIGN_DIR = "outputs/exp1/benign-validation-merged"
DEV_DIR = "outputs/exp1/rubric-dev-01"             # rubric-iteration set (base 720 items)
HARM_MANIFEST = "outputs/exp1/judge_validation_manifest.json"
KEEP = ("outputs/exp1/judge_validation_manifest.json",
        "outputs/exp1/judge_validation_manifest_benign.json", HARM_DIR, BENIGN_DIR, DEV_DIR,
        "outputs/exp2")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_inputs(input_root: Path, scratch: Path, dest: Path) -> list[str]:
    """Copy marked bundles into dest/outputs; returns the bundle dirs used, in order."""
    for z in sorted(input_root.rglob("stageC_*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(scratch / z.stem)
    roots = [r for r in (input_root, scratch) if r.exists()]
    uploads = [m for r in roots for m in r.rglob(UPLOAD)]
    resumes = sorted((m for r in roots for m in r.rglob(RESUME)),
                     key=lambda m: m.read_text(encoding="utf-8").strip())
    used = []
    for m in resumes + uploads:
        if (m.parent / "outputs").is_dir():
            shutil.copytree(m.parent / "outputs", dest / "outputs", dirs_exist_ok=True)
            used.append(str(m.parent))
    return used


def harm_passed() -> bool:
    from csjail.judge import load_judge_config
    from csjail.judge_validation import UnvalidatedJudgeError, require_validated_judge
    try:
        require_validated_judge(ROOT / HARM_MANIFEST, load_judge_config().fingerprint("harm"))
        return True
    except (UnvalidatedJudgeError, KeyError, ValueError):
        return False


def run_step(cmd: list[str], timeout_s: int) -> int:
    print(f"\n$ {' '.join(cmd)}   [time left {timeout_s // 60} min]", flush=True)
    try:
        return subprocess.run(cmd, cwd=ROOT, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        print("[stagec] time budget reached; step stopped (its work so far is cached)", flush=True)
        return 124


def plan(robustness: bool, dev_sample_dir: str | None = None,
         dev_judge_models: tuple[str, ...] = (),
         validation_predict: bool = False, screen: bool = False) -> list[tuple[str, list[str], callable]]:
    py = sys.executable
    benign = ("benign_gate", [py, "scripts/calibrate_judge.py", "--kind", "benign",
                              "--sample-dir", BENIGN_DIR], lambda st: (ROOT / BENIGN_DIR).is_dir())
    if screen:
        scr = [py, "scripts/exp2_verify.py", "screen", "--run-dir"]
        return [benign,
                ("screen_main", scr + ["outputs/exp2/main"],
                 lambda st: (ROOT / "outputs/exp2/main/generations.jsonl").exists()),
                ("screen_robustness", scr + ["outputs/exp2/robustness"],
                 lambda st: robustness and st["steps"].get("screen_main") == 0)]
    if dev_sample_dir:
        ready = lambda st: (ROOT / dev_sample_dir).is_dir()  # noqa: E731
        steps = [("rubric_dev", [py, "scripts/calibrate_judge.py", "--sample-dir", dev_sample_dir], ready)]
        for m in dev_judge_models:   # same rubric, other judge models, for comparison only
            steps.append((f"rubric_dev[{m}]", [py, "scripts/calibrate_judge.py", "--sample-dir",
                                                dev_sample_dir, "--judge-model", m], ready))
        return steps
    if validation_predict:
        # Two-phase validation, phase 1: the frozen judge's predictions on the validation
        # set, no verdict. The verdict comes from scripts/exp1_gold_audit.py score-two-phase.
        return [("validation_predict", [py, "scripts/calibrate_judge.py", "--sample-dir", HARM_DIR,
                                        "--predict-only"],
                 lambda st: (ROOT / HARM_DIR).is_dir())]
    # Final run. The harm verdict is the committed two-phase manifest (the original
    # validation gold was shown to be lenient, so it is not re-scored here).
    return [
        benign,
        ("exp2_main", [py, "-m", "csjail.run_eval", "--out-dir", "outputs/exp2/main", "--judge-only"],
         lambda st: harm_passed()),
        ("exp2_robustness", [py, "scripts/exp2_robustness.py", "--judge-only",
                             "--greedy-results", "outputs/exp2/main"],
         lambda st: robustness and st["steps"].get("exp2_main") == 0),
    ]


def judgment_counts(path: Path, key: str = "judge_key") -> dict:
    if not path.exists():
        return {}
    from csjail.pipeline import JsonlCache
    out: dict = {}
    for r in JsonlCache(path, key).records.values():
        out[r.get("judge_status")] = out.get(r.get("judge_status"), 0) + 1
    return out


def gate_progress(sample_dir: Path, kind: str) -> dict:
    """{'ok': n, 'api_error': m, ...} over the gate's saved predictions, plus the
    sample size, so a resumed run shows how much is left to judge (and pay for)."""
    import csv
    preds = sorted(sample_dir.glob(f"judge_predictions_{kind}_*.jsonl"))
    items = sample_dir / "items.csv"
    if items.exists():   # csv rows, not lines: cells contain line breaks
        with items.open(encoding="utf-8-sig", newline="") as f:
            n_items = sum(1 for _ in csv.DictReader(f))
    else:
        n_items = None
    counts = judgment_counts(preds[-1], "item_id") if preds else {}
    return {"n_items": n_items, "predictions": counts,
            "remaining": (n_items - counts.get("ok", 0)) if n_items is not None else None}


def package(out_zip: Path, status: dict) -> None:
    tmp = out_zip.with_suffix(".partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(RESUME, status["finished_utc"] + "\n")
        z.writestr("STAGEC_STATUS.json", json.dumps(status, indent=2))
        for k in KEEP:
            p = ROOT / k
            files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else []
            for f in files:
                z.write(f, f.relative_to(ROOT).as_posix())
    os.replace(tmp, out_zip)
    print(f"[stagec] resume bundle -> {out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--scratch", default="/tmp/stagec_inputs")
    ap.add_argument("--out-zip", default="/kaggle/working/stageC_outputs.zip")
    ap.add_argument("--time-budget-h", type=float, default=11.0,
                    help="no step runs past this many hours from the start (Kaggle stops at 12)")
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--dev-sample-dir", default=None,
                    help="rubric iteration: judge only this development sample (feedback "
                         "report, no manifest, no Exp 2)")
    ap.add_argument("--dev-judge-models", nargs="*", default=[],
                    help="with --dev-sample-dir: also score the sample with these judge models "
                         "(same rubric) for comparison")
    ap.add_argument("--validation-predict", action="store_true",
                    help="two-phase validation phase 1: cache the frozen judge's predictions on "
                         "the validation set; no verdict, no Exp 2")
    ap.add_argument("--screen", action="store_true",
                    help="measurement option 2: benign gate, then screen Exp 2 (eval_main) with the "
                         "configured judge for human verification")
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    left = lambda: int(args.time_budget_h * 3600 - (time.monotonic() - t0))  # noqa: E731
    mode = ("rubric_dev" if args.dev_sample_dir else
            "validation_predict" if args.validation_predict else
            "screen" if args.screen else "final")
    status = {"started_utc": now(), "mode": mode, "inputs": [], "steps": {}, "errors": []}
    try:
        status["inputs"] = collect_inputs(Path(args.input_root), Path(args.scratch), ROOT)
        if not status["inputs"]:
            raise RuntimeError("no input bundle found: add the private dataset made from "
                               "stageC_upload.zip (and the previous stageC_outputs.zip to resume)")
        print("[stagec] inputs (applied in this order):", *status["inputs"], sep="\n  ")
        from csjail.judge import load_judge_config
        c = load_judge_config()
        status["judge"] = {"model": c.model_id, "harm_rubric_version": c.harm_rubric_version,
                           "harm_fingerprint": c.fingerprint("harm")["fingerprint_id"],
                           "benign_fingerprint": c.fingerprint("benign")["fingerprint_id"]}
        print(f"[stagec] judge {status['judge']}")
        for name, cmd, ready in plan(not args.no_robustness, args.dev_sample_dir,
                                     tuple(args.dev_judge_models), args.validation_predict,
                                     args.screen):
            if not ready(status):
                status["steps"][name] = "skipped: prerequisite not met"
            elif left() < 300:
                status["steps"][name] = "skipped: time budget used up"
            else:
                status["steps"][name] = run_step(cmd, left())
            print(f"[stagec] {name}: {status['steps'][name]}", flush=True)
    except Exception as e:  # recorded, never allowed to skip packaging
        status["errors"].append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        status["finished_utc"] = now()
        status["harm_gate_pass"] = harm_passed()
        status["gates"] = {"harm": gate_progress(ROOT / HARM_DIR, "harm"),
                           "benign": gate_progress(ROOT / BENIGN_DIR, "benign")}
        if args.dev_sample_dir:
            status["gates"]["rubric_dev"] = gate_progress(ROOT / args.dev_sample_dir, "harm")
        status["judgments"] = {k: judgment_counts(ROOT / k / "judgments.jsonl")
                               for k in ("outputs/exp2/main", "outputs/exp2/robustness")}
        print("[stagec] status:", json.dumps(status, indent=2), flush=True)
        package(Path(args.out_zip), status)
    return 1 if status["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
