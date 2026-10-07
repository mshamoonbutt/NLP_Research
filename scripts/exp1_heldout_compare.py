#!/usr/bin/env python3
"""Exp 1 held-out test (2026-10-08): the chosen judge (DeepSeek V4.1 Flash) and the runner-up
(DeepSeek V4 Pro) on the 960 held-out responses (outputs/exp1/heldout-960, 160 families that
played no part in choosing the judge), next to the 720 development responses and both combined.
Numbers only: never prints prompt, response or rationale text.

    python scripts/calibrate_judge.py --sample-dir outputs/exp1/heldout-960 \
        --gold-csv outputs/exp1/heldout-960/gold.csv --judge-provider ollama \
        --judge-model deepseek-v4.1-flash:cloud --judge-concurrency 3     # and deepseek-v4-pro:cloud
    python scripts/exp1_heldout_compare.py

Reports overall / per-language / per model x language metrics, Flash - Pro and held-out - dev
with family-bootstrap intervals, held-out without the carried-over labels, and reviewer vs
original-annotator agreement. Output: outputs/exp1/heldout_comparison.json and a printed table.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp1_gold_audit as ga  # noqa: E402
from exp1_compare_judges import CONDS, load_predictions, metrics, returned  # noqa: E402
from csjail.utils.io import write_text_lf  # noqa: E402

SETS = {"dev": ROOT / "outputs/exp1/rubric-dev-01", "heldout": ROOT / "outputs/exp1/heldout-960"}
JUDGES = {"Flash": "b0248df31f1d9da2", "Pro": "403d0bc087c8abd5"}
MODEL_NAMES = {"Flash": "deepseek-v4.1-flash:cloud", "Pro": "deepseek-v4-pro:cloud"}
MODELS = ("qwen25", "phi3", "llama32")
DRAWS, SEED = 2000, 0
OUT = ROOT / "outputs/exp1/heldout_comparison.json"


def run_info(path: Path, fp: str) -> dict:
    recs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["item_id"]] = r          # last record per item wins, as in the cache
    assert all(r["judge_fingerprint_id"] == fp for r in recs.values()), f"{path.name}: wrong fingerprint"
    days = sorted(r["created_utc"][:10] for r in recs.values())
    return {"n": len(recs), "status": dict(Counter(r.get("judge_status") for r in recs.values())),
            "dates": [days[0], days[-1]]}


def load(name: str, folder: Path) -> tuple[list[dict], dict]:
    meta = {r["item_id"]: r for r in ga.read_csv(folder / "items.csv")}
    lab = {r["item_id"]: r for r in ga.read_csv(folder / "final_labels.csv")}
    assert set(meta) == set(lab)
    info = {j: run_info(folder / f"judge_predictions_harm_{fp}.jsonl", fp) for j, fp in JUDGES.items()}
    preds = {j: load_predictions(folder / f"judge_predictions_harm_{fp}.jsonl", set(lab))
             for j, fp in JUDGES.items()}
    recs = [{"set": name, "family": f"{name}:{meta[i]['base_id']}", "model": meta[i]["model"],
             "cond": meta[i]["condition"], "gold": lab[i]["unsafe"] == "true",
             "orig": lab[i]["original_unsafe"] == "true",
             "carried": lab[i]["label_source"].startswith("carried"),
             **{j: preds[j][i] for j in JUDGES}}            # KeyError = a missing/failed prediction
            for i in lab]
    return recs, info


def m(recs, j):
    return metrics([(r[j], r["gold"]) for r in recs])


def far(x):
    return x["fp"] / (x["fp"] + x["tn"]) if x["fp"] + x["tn"] else None


def per_language(recs, j):
    per = {c: m([r for r in recs if r["cond"] == c], j) for c in CONDS}
    return per, sum(per[c]["f1"] or 0.0 for c in CONDS) / len(CONDS)


def boot(recs, rng) -> dict:
    """Family-level bootstrap: per judge, (DRAWS, 4) resampled tp/fp/fn/tn totals. Both
    judges share the same resampled families, so their difference is paired."""
    fams = sorted({r["family"] for r in recs})
    ix = {f: k for k, f in enumerate(fams)}
    picks = rng.integers(0, len(fams), (DRAWS, len(fams)))
    w = np.zeros((DRAWS, len(fams)))
    np.add.at(w, (np.arange(DRAWS)[:, None], picks), 1)
    out = {}
    for j in JUDGES:
        c = np.zeros((len(fams), 4))
        for r in recs:
            c[ix[r["family"]], (0 if r["gold"] else 1) if r[j] else (2 if r["gold"] else 3)] += 1
        out[j] = w @ c
    return out


def stats(t: np.ndarray) -> dict:
    tp, fp, fn, tn = t.T
    n = t.sum(1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return {"f1": 2 * tp / (2 * tp + fp + fn), "kappa": ((tp + tn) / n - pe) / (1 - pe),
            "recall": tp / (tp + fn), "far": fp / (fp + tn)}


def interval(point, draws) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": point, "ci95": [float(lo), float(hi)], "share_gt0": float((draws > 0).mean())}


def flash_minus_pro(recs) -> dict:
    t = boot(recs, np.random.default_rng(SEED))
    a, b = stats(t["Flash"]), stats(t["Pro"])
    mf, mp = m(recs, "Flash"), m(recs, "Pro")
    return {k: interval(mf[k] - mp[k], a[k] - b[k]) for k in ("f1", "kappa")}


def heldout_minus_dev(dev, ho) -> dict:
    rng = np.random.default_rng(SEED)
    td, th = boot(dev, rng), boot(ho, rng)                  # independent family resamples
    out = {}
    for j in JUDGES:
        sd, sh = stats(td[j]), stats(th[j])
        md, mh = m(dev, j), m(ho, j)
        pt = {"f1": mh["f1"] - md["f1"], "kappa": mh["kappa"] - md["kappa"],
              "recall": mh["recall"] - md["recall"], "far": far(mh) - far(md)}
        out[j] = {k: interval(pt[k], sh[k] - sd[k]) for k in pt}
    return out


def cell_table(sets: dict, j: str) -> dict:
    out = {}
    for mo in MODELS:
        for c in CONDS:
            row = {}
            for s, recs in sets.items():
                x = m([r for r in recs if (r["model"], r["cond"]) == (mo, c)], j)
                row[s] = {"n": x["n"], "harmful": x["tp"] + x["fn"], "recall": x["recall"],
                          "far": far(x), "tp": x["tp"], "fn": x["fn"], "fp": x["fp"], "tn": x["tn"]}
            d, h = row["dev"], row["heldout"]
            row["fisher_p_recall"] = (fisher_exact([[d["tp"], d["fn"]], [h["tp"], h["fn"]]])[1]
                                      if d["harmful"] and h["harmful"] else None)
            row["fisher_p_far"] = fisher_exact([[d["fp"], d["tn"]], [h["fp"], h["tn"]]])[1]
            out[f"{mo}|{c}"] = row
    return out


def reviewer_ids() -> dict:
    out = {}
    for d in (ga.OUT, ga.OUT2, ga.OUT3, ga.OUT4):
        rows = ga.load_review(returned(d))
        out[d.name] = {"n": len(rows), "reviewer_id_blank": sum(not (r.get("reviewer_id") or "").strip()
                                                               for r in rows)}
    return out


def f(x, w=5):
    return f"{'--':>{w}}" if x is None else f"{x:{w}.2f}"


def main() -> int:
    dev, dinfo = load("dev", SETS["dev"])
    ho, hinfo = load("heldout", SETS["heldout"])
    sets = {"dev": dev, "heldout": ho, "combined": dev + ho}
    res = {"kind": "exp1_heldout_comparison", "created_utc": datetime.now(timezone.utc).isoformat(),
           "judges": {j: {"model": MODEL_NAMES[j], "provider": "ollama", "rubric": "harm-v2",
                          "fingerprint": fp} for j, fp in JUDGES.items()},
           "bootstrap": {"unit": "base_id family", "draws": DRAWS, "seed": SEED},
           "runs": {"dev": dinfo, "heldout": hinfo}, "overall": {}, "per_language": {},
           "macro_f1": {}, "per_model_language": {}, "flash_minus_pro": {}}
    for line in (f"{s} {j}: {i['n']} predictions, status {i['status']}, dates {i['dates']}"
                 for s, info in res["runs"].items() for j, i in info.items()):
        print(line)

    print("\n(a) overall")
    print(f"{'set':9} {'judge':5} {'n':>5} {'harm':>4} {'flag':>4}  {'P':>5} {'R':>5} {'F1':>5} "
          f"{'kappa':>5} {'spec':>5} {'flagR':>5}")
    for s, recs in sets.items():
        for j in JUDGES:
            x = res["overall"].setdefault(s, {})[j] = m(recs, j)
            print(f"{s:9} {j:5} {x['n']:5} {x['tp'] + x['fn']:4} {x['tp'] + x['fp']:4}  "
                  f"{f(x['precision'])} {f(x['recall'])} {f(x['f1'])} {f(x['kappa'])} "
                  f"{f(x['specificity'])} {f(x['flag_ratio'])}")

    print("\n(b) per language: P / R / F1 / flag ratio (harmful n)")
    for s, recs in sets.items():
        for j in JUDGES:
            per, mf = per_language(recs, j)
            res["per_language"].setdefault(s, {})[j], res["macro_f1"].setdefault(s, {})[j] = per, mf
            print(f"{s:9} {j:5} macroF1 {mf:.3f}  " + "  ".join(
                f"{c} {f(p['precision'], 4)}/{f(p['recall'], 4)}/{f(p['f1'], 4)}/{f(p['flag_ratio'], 4)} "
                f"({p['tp'] + p['fn']})" for c, p in per.items()))

    print("\n(c) per model x language: harmful/n, recall, false-alarm rate  [dev | heldout | combined]")
    for j in JUDGES:
        t = res["per_model_language"][j] = cell_table(sets, j)
        print(f"-- {j}")
        for k, row in t.items():
            print(f"{k:11} " + " | ".join(
                f"{row[s]['harmful']:2}/{row[s]['n']:3} R {f(row[s]['recall'], 4)} FAR {f(row[s]['far'], 4)}"
                for s in sets) + f" | Fisher p R {f(row['fisher_p_recall'], 4)} FAR {f(row['fisher_p_far'], 4)}")

    print(f"\n(d) Flash - Pro, family bootstrap ({DRAWS} draws, seed {SEED})")
    for s in ("heldout", "combined"):
        d = res["flash_minus_pro"][s] = flash_minus_pro(sets[s])
        print(f"{s:9} " + "  ".join(f"{k} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                                    f"Flash ahead {v['share_gt0']:.1%}" for k, v in d.items()))
    print("    also on dev (context):")
    d = res["flash_minus_pro"]["dev"] = flash_minus_pro(dev)
    print("dev       " + "  ".join(f"{k} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                                   f"Flash ahead {v['share_gt0']:.1%}" for k, v in d.items()))

    print("\n    held-out minus dev, per judge (independent family bootstraps)")
    g = res["heldout_minus_dev"] = heldout_minus_dev(dev, ho)
    for j, d in g.items():
        print(f"{j:5} " + "  ".join(f"{k} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                    for k, v in d.items()))
    res["heldout_minus_dev_per_language"] = {}
    for c in CONDS:
        g = res["heldout_minus_dev_per_language"][c] = heldout_minus_dev(
            [r for r in dev if r["cond"] == c], [r for r in ho if r["cond"] == c])
        print(f"  {c} F1 " + "  ".join(f"{j} {g[j]['f1']['diff']:+.3f} [{g[j]['f1']['ci95'][0]:+.3f}, "
                                      f"{g[j]['f1']['ci95'][1]:+.3f}]" for j in JUDGES))

    print("\n(e) held-out without the carried labels")
    nc = [r for r in ho if not r["carried"]]
    res["heldout_no_carried"] = {}
    for j in JUDGES:
        x = m(nc, j)
        per, mf = per_language(nc, j)
        carried_flags = sum(r[j] for r in ho if r["carried"])
        res["heldout_no_carried"][j] = {"overall": x, "per_language": per, "macro_f1": mf,
                                        "flags_on_carried": carried_flags}
        print(f"{j:5} n {x['n']} harm {x['tp'] + x['fn']}  P {f(x['precision'])} R {f(x['recall'])} "
              f"F1 {f(x['f1'])} kappa {f(x['kappa'])} spec {f(x['specificity'])} "
              f"flagR {f(x['flag_ratio'])} macroF1 {mf:.3f} | flags on the "
              f"{len(ho) - len(nc)} carried (all safe): {carried_flags}")
    d = res["flash_minus_pro"]["heldout_no_carried"] = flash_minus_pro(nc)
    print("Flash-Pro " + "  ".join(f"{k} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                                   f"Flash ahead {v['share_gt0']:.1%}" for k, v in d.items()))

    print("\n(f) reviewer vs original annotators (kappa)")
    res["reviewer_vs_original"] = {}
    for s, recs in (("heldout", ho), ("heldout_no_carried", nc), ("dev", dev)):
        x = metrics([(r["gold"], r["orig"]) for r in recs])
        res["reviewer_vs_original"][s] = {"n": x["n"], "kappa": x["kappa"], "agree": x["tp"] + x["tn"],
                                          "orig_safe_to_harmful": x["fp"], "orig_harmful_to_safe": x["fn"]}
        print(f"{s:19} n {x['n']} kappa {x['kappa']:.3f} agree {x['tp'] + x['tn']} "
              f"original safe->reviewer harmful {x['fp']}, original harmful->reviewer safe {x['fn']}")

    res["reviewer_id"] = reviewer_ids()
    print(f"\nreviewer_id blank per review file: {res['reviewer_id']}")
    write_text_lf(OUT, json.dumps(res, indent=2))
    print(f"-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
