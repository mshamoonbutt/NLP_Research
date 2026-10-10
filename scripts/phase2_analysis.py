#!/usr/bin/env python3
"""Phase 2 analysis across models and runs (Exp 8 + Exp 9), by the rules declared in
configs/dpo.yaml `analysis` (2026-10-10, before any Phase 2 output was judged).

    python scripts/phase2_analysis.py            # after phase2_kaggle.py judge (+ exp8_audit.py apply)

Reads outputs/exp8/<tag>__<model>/ (judged), outputs/exp6/review/verified_summary.json (budgets),
the Phase 1 judge manifest and, once applied, outputs/exp8/audit/audit_result.json.
  primary    RQ4 per model: C vs B_ext on CS, each family's outcome = mean judge label over the three
             seeds; sign-flip permutation p, Holm across models; finding = Holm-significant AND the
             corrected CI excludes 0 with the raw difference's sign. Per-seed McNemar as robustness.
  rates      ASR per model x arm x condition (C and B_ext pooled over seeds), raw and corrected.
  vs_A       every arm vs the untrained model per condition (EN drift, RU/UR transfer; secondary,
             unadjusted p).
  overrefusal  harmless-probe refusal per arm and form, vs A (raw; benign judge checked by the audit).
  capability MMLU / UrduMMLU per arm (seeds averaged) and retention vs A.
  ncurve     C at each budget (seed 42) vs A.
  ablation   unseen domain: adapters trained without it, on its held-out families.
Judge correction: Phase 1 error counts per model x condition, unless the audit switched that cell
(then every arm of it uses the audit's counts). Without audit_result.json the results are marked
PROVISIONAL. Writes outputs/phase2_analysis/analysis.json, rates.csv, comparisons.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from csjail.aggregate import (judge_error_counts, pooled_cells, predictive_value_asr,  # noqa: E402
                              predictive_value_diff, write_csv)
from csjail.asr import bootstrap_mean, mcnemar_paired  # noqa: E402
from csjail.data import CONDITIONS  # noqa: E402
from csjail.metrics import holm_bonferroni  # noqa: E402
from csjail.outcomes import benign_refused, primary_unsafe  # noqa: E402
from csjail.utils.io import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("phi3", "llama32")
NCURVE = (25, 50)


def load(tag: str, model: str) -> list[dict]:
    f = ROOT / "outputs" / "exp8" / f"{tag}__{model}" / "results.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: judge it first (phase2_kaggle.py judge)")
    return read_jsonl(str(f))


def fam_map(recs: list[dict], arm: str, cond: str, *, probe: bool = False, domain: Optional[str] = None) -> dict:
    """{family: outcome or None} for one arm and form (the harmless probe with probe=True)."""
    pred = benign_refused if probe else primary_unsafe
    return {r["base_id"]: pred(r) for r in recs
            if r["arm"] == arm and r["condition"] == cond and bool(r.get("probe")) == probe
            and (domain is None or r.get("domain_id") == domain)}


def pooled(maps: list[dict]) -> dict:
    """Per family, the mean of the labelled seeds (None when no seed has a label)."""
    out = {}
    for f in set().union(*maps):
        v = [m[f] for m in maps if m.get(f) is not None]
        out[f] = float(np.mean(v)) if v else None
    return out


def sign_flip_p(d: np.ndarray, n: int, seed: int) -> float:
    """Two-sided paired permutation p of mean(d): each family's sign flipped at random."""
    if not len(d):
        return None
    rng, obs, hits = np.random.default_rng(seed), abs(d.mean()), 0
    for start in range(0, n, 10_000):
        s = rng.choice((-1.0, 1.0), size=(min(10_000, n - start), len(d)))
        hits += int(np.sum(np.abs(s @ d) / len(d) >= obs - 1e-12))
    return (hits + 1) / (n + 1)


class Ctx:
    def __init__(self, cells: dict, strata: dict, st: dict, n_perm: int):
        self.cells, self.strata, self.st, self.n_perm = cells, strata, st, n_perm

    def counts(self, model: str, cond: str, probe: bool):
        if probe:
            return None, None
        return self.cells.get(f"{model}|{cond}"), pooled_cells(self.cells, model)

    def rate(self, m: dict, model: str, cond: str, probe: bool = False) -> dict:
        fams = sorted(f for f, v in m.items() if v is not None)
        v = np.array([float(m[f]) for f in fams])
        out = {"n_families": len(fams), "n_missing": len(m) - len(fams), "rate": float(v.mean()) if len(v) else None}
        out["ci_lo"], out["ci_hi"] = bootstrap_mean(v, fams, strata=[self.strata.get(f) for f in fams],
                                                     bootstrap_n=self.st["bootstrap_n"], seed=self.st["bootstrap_seed"])
        counts, fb = self.counts(model, cond, probe)
        if counts and len(v):
            c = predictive_value_asr(list(v), fams, [self.strata.get(f) for f in fams], counts, fb,
                                     bootstrap_n=self.st["bootstrap_n"], ci_alpha=self.st["ci_alpha"],
                                     seed=self.st["bootstrap_seed"])
            out.update({"rate_corrected": c["asr_corrected"], "rate_corrected_ci_lo": c["asr_corrected_ci_lo"],
                        "rate_corrected_ci_hi": c["asr_corrected_ci_hi"]})
        return out

    def compare(self, a: dict, b: dict, model: str, cond: str, probe: bool = False) -> dict:
        """Paired a - b over families labelled in both; one error cell for both arms (same draws)."""
        fams = sorted(f for f in set(a) & set(b) if a[f] is not None and b[f] is not None)
        av, bv = np.array([float(a[f]) for f in fams]), np.array([float(b[f]) for f in fams])
        d, strata = av - bv, [self.strata.get(f) for f in fams]
        out = {"n_families": len(fams), "n_dropped": len(set(a) | set(b)) - len(fams),
               "rate_a": float(av.mean()) if len(fams) else None, "rate_b": float(bv.mean()) if len(fams) else None,
               "diff": float(d.mean()) if len(fams) else None,
               "p_perm": sign_flip_p(d, self.n_perm, self.st["bootstrap_seed"])}
        out["diff_ci_lo"], out["diff_ci_hi"] = bootstrap_mean(d, fams, strata=strata, bootstrap_n=self.st["bootstrap_n"],
                                                               seed=self.st["bootstrap_seed"])
        counts, fb = self.counts(model, cond, probe)
        if counts and len(fams):
            out.update(predictive_value_diff(list(av), list(bv), strata, counts, counts, fb,
                                             bootstrap_n=self.st["bootstrap_n"], ci_alpha=self.st["ci_alpha"],
                                             seed=self.st["bootstrap_seed"]))
        return out


def training_table(data: dict, main_tag: dict) -> list[dict]:
    """Every adapter's training log against the declared learning check (configs/dpo.yaml), its
    effective optimizer steps (finite gradient, nonzero learning rate: fp16 loss scaling skips the
    first steps whose gradients overflow) and the share of its responses identical to arm A's."""
    import math

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase2_kaggle import learning_check   # the same check the gate used

    lc = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))["learning_check"]
    rows = []
    for tm in sorted((ROOT / "outputs" / "models").glob("*/training_manifest.json")):
        man = json.loads(tm.read_text(encoding="utf-8"))
        model, arm, ep = man["model_key"], man["arm"], man["dpo_config"]["epochs"]
        tag = tm.parent.name.split(f"_{model}_", 1)[1]
        steps = [h for h in man["log_history"] if "loss" in h]
        chk = learning_check(man, ep, lc)
        row = {"adapter": tm.parent.name, "model": model, "arm": arm, "tag": tag, "epochs": ep, "n_pairs": man["n_pairs"],
               "steps": len(steps),
               "effective_steps": sum(1 for h in steps if h.get("learning_rate", 0) > 0 and h.get("grad_norm") is not None
                                      and math.isfinite(h["grad_norm"])),
               "final_epoch_loss": chk["final_epoch_loss"], "final_epoch_reward_accuracy": chk["final_epoch_reward_accuracy"],
               "learned": chk["pass"]}
        recs = data.get(model, {}).get(tag)
        if recs is not None:
            a = {(r["row_id"], bool(r.get("probe"))): r["response_sha256"] for r in data[model][main_tag[model]] if r["arm"] == "A"}
            x = {(r["row_id"], bool(r.get("probe"))): r["response_sha256"] for r in recs if r["arm"] == arm}
            keys = set(a) & set(x)
            row["identical_to_A"] = sum(a[k] == x[k] for k in keys) / len(keys) if keys else None
        rows.append(row)
    return rows


def error_cells(judge_manifest: Path, recs: list[dict]) -> tuple[dict, dict, str]:
    """Per model x condition the counts the correction uses (one dict object per cell, shared by its arms)."""
    cells = judge_error_counts(judge_manifest, recs)
    if cells is None:
        raise SystemExit(f"FAIL: the Exp 8 judgments are not from the judge of {judge_manifest}")
    src = {c: "phase1" for c in cells}
    audit = ROOT / "outputs" / "exp8" / "audit" / "audit_result.json"
    if not audit.exists():
        return cells, src, "NOT_DONE: Phase 1 judge-error counts throughout; results are PROVISIONAL"
    for c, r in json.loads(audit.read_text(encoding="utf-8"))["cells"].items():
        if r["switch"]:
            cells[c] = {k: r["audit"][k] for k in ("tp", "fn", "tn", "fp")}
            src[c] = "audit"
    return cells, src, "APPLIED"


def finding(t: dict, reject: Optional[bool]) -> bool:
    lo, hi, d = t.get("diff_corrected_ci_lo"), t.get("diff_corrected_ci_hi"), t.get("diff")
    return bool(reject and None not in (lo, hi, d) and d != 0
                and ((lo > 0 and hi > 0 and d > 0) or (lo < 0 and hi < 0 and d < 0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--judge-manifest", default=str(ROOT / "outputs/exp1/judge_validation_manifest.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "phase2_analysis"))
    args = ap.parse_args(argv)
    acfg = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))["analysis"]
    st = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))["stats"]
    summ = json.loads((ROOT / "outputs/exp6/review/verified_summary.json").read_text(encoding="utf-8"))
    ab_split = sorted((ROOT / "outputs" / "exp9").glob("ablation_*/split_manifest.json"))
    ab = json.loads(ab_split[0].read_text(encoding="utf-8"))["meta"]["ablation_domain"] if ab_split else None
    seeds = acfg["seeds"]

    data, rates, comps, primary, cap = {}, [], [], [], []
    for model in args.models:
        n, nab = summ["budget_main"][model], summ["budget_ablation"][model]
        main_tags = [f"n{n}"] + [f"n{n}_s{s}" for s in seeds[1:]]
        tags = main_tags + [f"n{b}" for b in NCURVE if b < n] + ([f"n{nab}_ablation_{ab}"] if ab else [])
        data[model] = {t: load(t, model) for t in tags}
    every = [r for m in data.values() for rs in m.values() for r in rs if not r.get("probe")]
    cells, cell_src, audit_status = error_cells(Path(args.judge_manifest), every)
    strata = {r["base_id"]: r["domain_id"] for m in data.values() for rs in m.values() for r in rs}   # probe ids BN-* differ
    ctx = Ctx(cells, strata, st, acfg["permutation_n"])

    for model in args.models:
        n, nab = summ["budget_main"][model], summ["budget_ablation"][model]
        d = data[model]
        main_tags = [f"n{n}"] + [f"n{n}_s{s}" for s in seeds[1:]]

        def arm_map(arm, cond, probe=False, domain=None, tags=main_tags):
            src = tags if arm in ("C", "B_ext") else tags[:1]      # A and E: the seed-42 folder
            return pooled([fam_map(d[t], arm, cond, probe=probe, domain=domain) for t in src])

        # RQ4 primary
        t = ctx.compare(arm_map("C", "CS"), arm_map("B_ext", "CS"), model, "CS")
        t["per_seed_mcnemar"] = {tag: mcnemar_paired(fam_map(d[tag], "C", "CS"), fam_map(d[tag], "B_ext", "CS"),
                                                     cond_a="C:CS", cond_b="B_ext:CS", strata=strata,
                                                     bootstrap_n=st["bootstrap_n"], seed=st["bootstrap_seed"]).as_dict()
                                 for tag in main_tags}
        primary.append({"model": model, "budget": n, **t})

        for probe in (False, True):
            arms = ("A", "E", "B_ext", "C")
            for cond in CONDITIONS:
                maps = {a: arm_map(a, cond, probe) for a in arms}
                for a in arms:
                    rates.append({"kind": "overrefusal" if probe else "asr", "model": model, "arm": a, "condition": cond,
                                  "budget": n, "pooled_seeds": len(main_tags) if a in ("C", "B_ext") else 1,
                                  **ctx.rate(maps[a], model, cond, probe)})
                    if a != "A":
                        comps.append({"kind": ("overrefusal_vs_A" if probe else "vs_A"), "model": model,
                                      "comparison": f"{a}-A", "condition": cond,
                                      **ctx.compare(maps[a], maps["A"], model, cond, probe)})

        for b in [x for x in NCURVE if x < n] + [n]:
            tag = f"n{b}"
            for cond in ("CS", "EN"):
                cm, am = fam_map(d[tag], "C", cond), fam_map(d[main_tags[0]], "A", cond)
                comps.append({"kind": "ncurve", "model": model, "comparison": f"C_n{b}-A", "condition": cond,
                              "budget": b, **ctx.compare(cm, am, model, cond)})

        if ab:
            tag = f"n{nab}_ablation_{ab}"
            for cond in CONDITIONS:
                m = {a: fam_map(d[tag], a, cond, domain=ab) for a in ("B_ext", "C")}
                m["A"] = fam_map(d[main_tags[0]], "A", cond, domain=ab)    # A is generated in the main folder only
                for a, b in (("C", "A"), ("B_ext", "A"), ("C", "B_ext")):
                    comps.append({"kind": "ablation", "model": model, "comparison": f"{a}-{b}", "condition": cond,
                                  "domain": ab, "budget": nab, **ctx.compare(m[a], m[b], model, cond)})
                comps.append({"kind": "ablation_reference", "model": model, "comparison": "C(main, trained with it)-A",
                              "condition": cond, "domain": ab, "budget": n,
                              **ctx.compare(arm_map("C", cond, domain=ab), arm_map("A", cond, domain=ab), model, cond)})

        for arm in ("A", "E", "B_ext", "C"):
            src = main_tags if arm in ("C", "B_ext") else main_tags[:1]
            got = []
            for tag in src:
                man = json.loads((ROOT / "outputs" / "exp8" / f"{tag}__{model}" / "run_manifest.json").read_text(encoding="utf-8"))
                got.append(man.get("capability", {}).get(f"{model}/{arm}") or {})
            row = {"model": model, "arm": arm, "n_seeds": len(src)}
            for k in ("mmlu", "urdummlu"):
                v = [g[k] for g in got if g.get(k) is not None]
                row[k] = float(np.mean(v)) if v else None
            cap.append(row)
    base = {(r["model"]): r for r in cap if r["arm"] == "A"}
    for r in cap:
        for k in ("mmlu", "urdummlu"):
            a = base[r["model"]].get(k)
            r[f"{k}_retention"] = r[k] / a if r[k] is not None and a else None

    holm = holm_bonferroni([t["p_perm"] for t in primary], alpha=acfg["alpha"])
    for t, h in zip(primary, holm):
        t["p_holm"], t["holm_reject"] = h["p_adjusted"], h["reject"]
        t["finding"] = finding(t, h["reject"])

    # Adapters that never effectively trained (declared learning check on their own logs) make
    # their comparisons uninformative: flagged, not dropped.
    training = training_table(data, {m: f"n{summ['budget_main'][m]}" for m in args.models})
    learned = {t["adapter"]: t["learned"] for t in training}
    for c in comps:
        if c["kind"] == "ncurve":
            c["adapter_learned"] = learned.get(f"C_{c['model']}_n{c['budget']}")
        elif c["kind"] == "ablation":
            c["adapter_learned"] = all(learned.get(f"{a}_{c['model']}_n{c['budget']}_ablation_{ab}")
                                       for a in c["comparison"].split("-") if a != "A")
    for t in primary:
        tags = [f"n{t['budget']}"] + [f"n{t['budget']}_s{s}" for s in seeds[1:]]
        t["adapters_learned"] = all(learned.get(f"{a}_{t['model']}_{g}") for a in ("C", "B_ext") for g in tags)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps({
        "kind": "phase2_analysis", "rules": acfg, "audit": audit_status, "error_cell_source": cell_src,
        "ablation_domain": ab, "primary": primary, "capability": cap,
        "notes": ["vs_A, ncurve, ablation and overrefusal p-values are unadjusted (secondary, reporting only)",
                  "over-refusal is uncorrected; the audit reports the benign judge's RU/UR agreement",
                  "RU/UR corrections use Phase 1 counts (not audited)",
                  "training.csv: adapters failing the declared learning check did not effectively train (fp16 loss "
                  "scaling skips the first steps whose gradients overflow); comparisons using them carry "
                  "adapter_learned=False and are not evidence"]}, indent=2, default=str), encoding="utf-8")
    write_csv(out / "rates.csv", rates)
    write_csv(out / "comparisons.csv", comps)
    if training:
        write_csv(out / "training.csv", training)
    print(f"[phase2] audit: {audit_status}")
    print(f"[phase2] adapters that did not learn: {[t['adapter'] for t in training if not t['learned']] or 'none'}")
    for t in primary:
        print(f"[phase2] RQ4 {t['model']} (n={t['budget']}, {t['n_families']} families): C-B_ext on CS "
              f"{t['diff']:+.3f} [{t['diff_ci_lo']:+.3f}, {t['diff_ci_hi']:+.3f}], corrected "
              f"{t.get('diff_corrected')} [{t.get('diff_corrected_ci_lo')}, {t.get('diff_corrected_ci_hi')}], "
              f"p_perm={t['p_perm']:.4g} p_holm={t['p_holm']:.4g} -> {'FINDING' if t['finding'] else 'no finding'}")
    print(f"[phase2] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
