#!/usr/bin/env python3
"""Exp 10: paste-ready LaTeX for the Phase 2 results (tables, two pgfplots figures, disclosures).

    python scripts/exp10_report.py            # after phase2_analysis.py (and exp10_residuals.py apply)

Reads outputs/phase2_analysis/{analysis.json, rates.csv, comparisons.csv, training.csv},
outputs/exp8/audit/audit_result.json, outputs/phase2_gate/*.json, the judged Exp 8 records (response
types; local) and, once applied, outputs/exp10/residual_result.json. Writes docs/paper_phase2.tex and
outputs/phase2_analysis/response_composition.csv (the table view of Figure 2). Figures are pgfplots
(compile in Overleaf; needs \\usepackage{pgfplots} and \\usepgfplotslibrary{groupplots}).
Colours: the dataviz reference categorical slots 1-3 (arms) and 1-4 (response types), validated
light-mode (CVD and normal-vision floors pass); every plotted value is also in a table.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.aggregate import write_csv  # noqa: E402
from csjail.utils.io import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "outputs" / "phase2_analysis"
MODELS = ("phi3", "llama32")
NAME = {"phi3": "Phi-3-mini", "llama32": "Llama-3.2"}
ARM = {"A": "A (untrained)", "E": "E (safety prompt)", "B_ext": "B (external English)", "C": "C (own CS failures)"}
SHORT = {"A": "A", "E": "E", "B_ext": "B", "C": "C"}
CONDS = ("EN", "CS", "RU", "UR")
KINDS = {"harmful": {"harmful_compliance"}, "refusal": {"refusal"}, "safe_helpful": {"safe_helpful"},
         "non_response": {"irrelevant", "unintelligible", "empty"}}
ARM_COLOR = {"C": "2A78D6", "B_ext": "EB6834", "E": "1BAF7A"}          # slots 1-3
ARM_MARK = {"C": "*", "B_ext": "square*", "E": "triangle*"}
MARK_SIZE = {"C": "2.5pt", "B_ext": "2.2pt", "E": "3.1pt"}     # similar visual area per shape
KIND_COLOR = {"harmful": "2A78D6", "refusal": "EB6834", "safe_helpful": "1BAF7A", "non_response": "EDA100"}


def num(x, signed=False, scale=100.0, nd=1) -> str:
    if x in (None, ""):
        return "--"
    v = float(x) * scale
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def ci(lo, hi) -> str:
    return "--" if lo in (None, "") else f"[{num(lo)}, {num(hi)}]"


def pval(p) -> str:
    p = float(p)
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def load():
    a = json.loads((AN / "analysis.json").read_text(encoding="utf-8"))
    rates = list(csv.DictReader((AN / "rates.csv").open(encoding="utf-8")))
    comps = list(csv.DictReader((AN / "comparisons.csv").open(encoding="utf-8")))
    training = list(csv.DictReader((AN / "training.csv").open(encoding="utf-8")))
    audit = json.loads((ROOT / "outputs/exp8/audit/audit_result.json").read_text(encoding="utf-8"))
    gates = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "outputs/phase2_gate").glob("*_e*.json")}
    res = ROOT / "outputs/exp10/residual_result.json"
    return a, rates, comps, training, audit, gates, (json.loads(res.read_text(encoding="utf-8")) if res.exists() else None)


def rate(rates, kind, model, arm, cond):
    return next(r for r in rates if r["kind"] == kind and r["model"] == model and r["arm"] == arm and r["condition"] == cond)


def comp(comps, kind, model, comparison, cond):
    return next(r for r in comps if r["kind"] == kind and r["model"] == model and r["comparison"] == comparison
                and r["condition"] == cond)


def composition(a) -> list[dict]:
    """Share of each response type (judge's response_kind) per model x arm x condition, C/B_ext pooled over seeds."""
    seeds = a["rules"]["seeds"]
    rows = []
    for t in a["primary"]:
        model, n = t["model"], t["budget"]
        main = [f"n{n}"] + [f"n{n}_s{s}" for s in seeds[1:]]
        recs = {tag: read_jsonl(str(ROOT / "outputs/exp8" / f"{tag}__{model}" / "results.jsonl")) for tag in main}
        for arm in ("A", "E", "B_ext", "C"):
            tags = main if arm in ("B_ext", "C") else main[:1]
            for cond in CONDS:
                c = collections.Counter()
                for tag in tags:
                    for r in recs[tag]:
                        if r["arm"] == arm and r["condition"] == cond and not r.get("probe"):
                            c[next((k for k, s in KINDS.items() if r.get("judge_response_kind") in s), "other")] += 1
                tot = sum(c.values())
                rows.append({"model": model, "arm": arm, "condition": cond, "n": tot,
                             **{k: c[k] / tot for k in list(KINDS) + ["other"]}})
    return rows


def table_mitigation(a, rates, cap) -> str:
    out = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{@{}llrcccrr@{}}",
           r"\toprule",
           r"Model & Arm & Pairs & CS ASR & EN ASR & Benign ORR & MMLU & UrduMMLU \\", r"\midrule"]
    for t in a["primary"]:
        m = t["model"]
        for i, arm in enumerate(("A", "E", "B_ext", "C")):
            cs, en = rate(rates, "asr", m, arm, "CS"), rate(rates, "asr", m, arm, "EN")
            orr = rate(rates, "overrefusal", m, arm, "ALL")
            c = cap[(m, arm)]
            out.append(f"{NAME[m] if i == 0 else ''} & {ARM[arm]} & {t['budget'] if arm in ('B_ext', 'C') else 0} & "
                       f"{num(cs['rate_corrected'])} {ci(cs['rate_corrected_ci_lo'], cs['rate_corrected_ci_hi'])} & "
                       f"{num(en['rate_corrected'])} {ci(en['rate_corrected_ci_lo'], en['rate_corrected_ci_hi'])} & "
                       f"{num(orr['rate'])} {ci(orr['ci_lo'], orr['ci_hi'])} & {c['mmlu']:.3f} & {c['urdummlu']:.3f} \\\\")
        out.append(r"\midrule" if m != a["primary"][-1]["model"] else r"\bottomrule")
    out += [r"\end{tabular}",
            r"\caption{Held-out mitigation and utility results (200 test families, all four forms per family; "
            r"\% with 95\% intervals). ASR is the harmful-compliance rate corrected for the judge's error in that "
            r"model and condition (raw rates in Table~\ref{tab:p2-transfer}); B and C pool three training seeds. "
            r"Benign ORR: refusal of the 60 harmless prompts, averaged over their four forms (per form in "
            r"Table~\ref{tab:p2-orr}). MMLU: 500 items; UrduMMLU: 300 items (chance 0.25). E is a "
            r"safety system prompt, not trained.}",
            r"\label{tab:mitigation}", r"\end{table*}"]
    return "\n".join(out)


def table_rq4(a) -> str:
    out = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}lrcccccc@{}}", r"\toprule",
           r"Model & Pairs & C & B & C$-$B (raw) & C$-$B (corrected) & $p_{\mathrm{Holm}}$ & Per seed (42/43/44) \\",
           r"\midrule"]
    for t in a["primary"]:
        seeds = " / ".join(num(v["diff"], signed=True) for v in t["per_seed_mcnemar"].values())
        star = r"$^\ast$" if t["finding"] else ""
        out.append(f"{NAME[t['model']]} & {t['budget']} & {num(t['rate_a'])} & {num(t['rate_b'])} & "
                   f"{num(t['diff'], True)} {ci(t['diff_ci_lo'], t['diff_ci_hi'])} & "
                   f"{num(t['diff_corrected'], True)} {ci(t['diff_corrected_ci_lo'], t['diff_corrected_ci_hi'])} & "
                   f"{pval(t['p_holm'])}{star} & {seeds} \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Primary Phase-2 contrast (RQ4): harmful compliance on CS prompts after training on the "
            r"model's own CS failures (C) versus the same number of external English preference pairs (B), "
            r"200 held-out families. Each family's outcome is the mean judge label over three training seeds; "
            r"$p$ from a paired sign-flip permutation test over families (100{,}000 draws), Holm-adjusted across "
            r"the two models. $^\ast$Finding: Holm-significant and the corrected interval excludes zero with the "
            r"same sign. Per-seed values are single-seed paired differences (robustness, not the test).}",
            r"\label{tab:rq4}", r"\end{table*}"]
    return "\n".join(out)


def figure_forest(comps) -> str:
    rows = {c: i for i, c in enumerate(reversed(CONDS), 1)}          # EN on top
    off = {"C": 0.22, "B_ext": 0.0, "E": -0.22}
    vals = [float(r[k]) * 100 for r in comps if r["kind"] == "vs_A" for k in ("diff_corrected_ci_lo", "diff_corrected_ci_hi")]
    lo, hi = 5 * (int(min(vals)) // 5 - 1), 5 * (int(max(vals)) // 5 + 1)
    out = [r"\begin{figure*}[t]", r"\centering",
           *[rf"\definecolor{{ptwo{a.replace('_', '')}}}{{HTML}}{{{c}}}" for a, c in ARM_COLOR.items()],
           r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.9cm, yticklabels at=edge left},",
           rf"  width=0.49\textwidth, height=5cm, xmin={lo}, xmax={hi}, ymin=0.5, ymax=4.5,",
           r"  ytick={1,2,3,4}, yticklabels={UR,RU,CS,EN}, xmajorgrids, grid style={black!10},",
           r"  axis line style={black!35}, tick style={black!35}, tick label style={font=\small, black!75},",
           r"  title style={font=\small},",
           r"  every axis plot/.append style={only marks,",
           r"    error bars/x dir=both, error bars/x explicit, error bars/error bar style={line width=0.8pt}}]"]
    for i, m in enumerate(MODELS):
        out.append(rf"\nextgroupplot[title={{{NAME[m]}}}" + (r", legend to name=ptwolegend, legend columns=3, "
                   r"legend style={draw=none, font=\small, column sep=8pt}" if i == 0 else "") + "]")
        out.append(r"\draw[black!55, line width=0.6pt] (axis cs:0,0.5) -- (axis cs:0,4.5);")
        for arm in ("C", "B_ext", "E"):
            pts = []
            for cond in CONDS:
                r = comp(comps, "vs_A", m, f"{arm}-A", cond)
                x, xl, xh = (float(r[k]) * 100 for k in ("diff_corrected", "diff_corrected_ci_lo", "diff_corrected_ci_hi"))
                pts.append(f"({x:.2f},{rows[cond] + off[arm]:.2f}) += ({xh - x:.2f},0) -= ({x - xl:.2f},0)")
            col = f"ptwo{arm.replace('_', '')}"
            # no "+": the default cycle list would override the marker fill
            out.append(rf"\addplot[color={col}, mark={ARM_MARK[arm]}, mark size={MARK_SIZE[arm]}, "
                       rf"mark options={{fill={col}, draw=white, line width=0.4pt}}, "
                       rf"error bars/error bar style={{color={col}}}] coordinates {{{' '.join(pts)}}};")
            if i == 0:
                out.append(rf"\addlegendentry{{{ARM[arm]}}}")
    out += [r"\end{groupplot}", r"\end{tikzpicture}",
            r"\\[1pt]{\small\color{black!75} Change in harmful compliance vs.\ the untrained model "
            r"(percentage points, judge-corrected)}",
            r"\\[3pt]\pgfplotslegendfromname{ptwolegend}",
            r"\caption{Change in harmful compliance relative to the untrained model, by prompt form (points: "
            r"judge-corrected difference; bars: 95\% family-bootstrap intervals; 200 held-out families; B and C "
            r"pool three seeds). Values in Table~\ref{tab:p2-transfer}.}",
            r"\label{fig:p2-effects}", r"\end{figure*}"]
    return "\n".join(out)


def figure_composition(comp_rows) -> str:
    y = {"A": 4, "E": 3, "B_ext": 2, "C": 1}
    out = [r"\begin{figure*}[t]", r"\centering",
           *[rf"\definecolor{{ptwo{k.replace('_', '')}}}{{HTML}}{{{c}}}" for k, c in KIND_COLOR.items()],
           r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=4 by 2, horizontal sep=0.25cm, vertical sep=0.45cm,",
           r"    yticklabels at=edge left, xticklabels at=edge bottom},",
           r"  width=0.29\textwidth, height=3.1cm, xbar stacked, /pgf/bar width=6pt, xmin=0, xmax=100, ymin=0.4, ymax=4.6,",
           r"  ytick={1,2,3,4}, yticklabels={C,B,E,A}, xtick={0,50,100}, axis line style={black!35},",
           r"  tick style={black!35}, tick label style={font=\scriptsize, black!75}, title style={font=\small},",
           r"  ylabel style={font=\small}, every axis plot/.append style={draw=white, line width=0.6pt}]"]
    first = True
    for m in MODELS:
        for j, cond in enumerate(CONDS):
            opts = []
            if m == MODELS[0]:
                opts.append(f"title={{{cond}}}")
            if m == MODELS[0]:        # per-panel labels override "edge bottom": blank the top row
                opts.append("xticklabels={}")
            elif j < len(CONDS) - 1:  # "100" only on the last column, or it runs into the next panel's "0"
                opts.append("xticklabels={0,50,{}}")
            if j == 0:
                opts.append(f"ylabel={{{NAME[m]}}}")
            if first:
                opts.append(r"legend to name=ptwocomp, legend columns=4, "
                            r"legend style={draw=none, font=\small, column sep=8pt}")
            out.append(r"\nextgroupplot[" + ", ".join(opts) + "]")
            for k in KINDS:
                pts = " ".join(f"({100 * r[k]:.1f},{y[r['arm']]})" for r in comp_rows
                               if r["model"] == m and r["condition"] == cond)
                out.append(rf"\addplot[fill=ptwo{k.replace('_', '')}] coordinates {{{pts}}};")
                if first:
                    out.append(rf"\addlegendentry{{{k.replace('_', '-')}}}")
            first = False
    out += [r"\end{groupplot}", r"\end{tikzpicture}", r"\\[2pt]\pgfplotslegendfromname{ptwocomp}",
            r"\caption{What the models answer instead (\% of responses to the 200 held-out harmful prompts, by "
            r"the judge's response type; non-response = irrelevant, unintelligible or empty). A untrained, E safety "
            r"prompt, B external English pairs, C own CS failures (B and C pool three seeds). Values in "
            r"Table~\ref{tab:p2-composition}.}",
            r"\label{fig:p2-composition}", r"\end{figure*}"]
    return "\n".join(out)


def appendix(a, rates, comps, training, audit, gates, residual, comp_rows) -> str:
    t = []
    # transfer: raw and corrected per form, arm - A
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
          r"Model & Arm $-$ A & EN & CS & RU & UR \\", r"\midrule"]
    for m in MODELS:
        for i, arm in enumerate(("C", "B_ext", "E")):
            rs = [comp(comps, "vs_A", m, f"{arm}-A", cond) for cond in CONDS]
            t.append(f"{NAME[m] if i == 0 else ''} & {SHORT[arm]} & " + " & ".join(
                f"{num(r['diff_corrected'], True)} {ci(r['diff_corrected_ci_lo'], r['diff_corrected_ci_hi'])}" for r in rs)
                + r" \\")
            t.append(r" & \footnotesize raw & " + " & ".join(rf"\footnotesize {num(r['diff'], True)}" for r in rs) + r" \\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    t += [r"\end{tabular}", r"\caption{Change in harmful compliance versus the untrained model per prompt form "
          r"(pp; corrected with 95\% interval, raw below). Secondary analyses: $p$-values unadjusted and not "
          r"shown as tests. RU = Roman Urdu, UR = Urdu script.}", r"\label{tab:p2-transfer}", r"\end{table*}", ""]
    # over-refusal per form
    t += [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
          r"Model & Arm & EN & CS & RU & UR & All \\", r"\midrule"]
    for m in MODELS:
        for i, arm in enumerate(("A", "E", "B_ext", "C")):
            t.append(f"{NAME[m] if i == 0 else ''} & {SHORT[arm]} & "
                     + " & ".join(num(rate(rates, "overrefusal", m, arm, c)["rate"]) for c in CONDS + ("ALL",)) + r" \\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    agree = ", ".join(f"{NAME[k.split('|')[0]]} {k.split('|')[1]} {v['agreement']:.2f}" for k, v in audit["benign"].items())
    t += [r"\end{tabular}", r"\caption{Refusal of the 60 harmless prompts per form (\%). Over-refusal is "
          r"uncorrected; the harmless-prompt judge agreed with the reviewer on RU/UR responses at " + agree
          + r" ($n=15$ each).}", r"\label{tab:p2-orr}", r"\end{table}", ""]
    # n-curve and unseen domain
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}llcl@{}}", r"\toprule",
          r"Model & Comparison & CS, C $-$ A (corrected) & Trained \\", r"\midrule"]
    for r in comps:
        if r["kind"] == "ncurve" and r["condition"] == "CS":
            t.append(f"{NAME[r['model']]} & C, {r['budget']} pairs (seed 42) & {num(r['diff_corrected'], True)} "
                     f"{ci(r['diff_corrected_ci_lo'], r['diff_corrected_ci_hi'])} & "
                     f"{'yes' if r['adapter_learned'] == 'True' else 'no (see text)'} \\\\")
    t.append(r"\midrule")
    for r in comps:
        if r["kind"] in ("ablation", "ablation_reference") and r["condition"] == "CS":
            lab = {"C-A": "D6 withheld: C $-$ A", "B_ext-A": "D6 withheld: B $-$ A", "C-B_ext": "D6 withheld: C $-$ B",
                   "C(main, trained with it)-A": "D6 in training: C $-$ A"}[r["comparison"]]
            learned = "yes" if r["kind"] == "ablation_reference" or r.get("adapter_learned") == "True" else "no (see text)"
            t.append(f"{NAME[r['model']]} & {lab} ($n={r['n_families']}$) & {num(r['diff_corrected'], True)} "
                     f"{ci(r['diff_corrected_ci_lo'], r['diff_corrected_ci_hi'])} & {learned} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Data efficiency (C at each pair budget) and the unseen "
          r"domain (D6, drawn by a rule fixed before the draw; its held-out families only). ``Trained'' applies the "
          r"declared learning check to that adapter's own training log (Table~\ref{tab:p2-training}).}",
          r"\label{tab:p2-ncurve}", r"\end{table*}", ""]
    # training
    t += [r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrrl@{}}", r"\toprule",
          r"Adapter & Pairs & Steps & Effective & Final-epoch loss & Reward acc. & Same as A & Learned \\",
          r"\midrule"]
    for r in training:
        t.append(f"\\texttt{{{r['adapter'].replace('_', chr(92) + '_')}}} & {r['n_pairs']} & {r['steps']} & "
                 f"{r['effective_steps']} & {float(r['final_epoch_loss']):.3f} & {float(r['final_epoch_reward_accuracy']):.2f} & "
                 f"{num(r['identical_to_A'])}\\% & {'yes' if r['learned'] == 'True' else 'no'} \\\\")
    g = {k: v for k, v in gates.items()}
    gtxt = "; ".join(f"{NAME[m]} {e} epochs: " + ", ".join(
        f"{a.split('_')[0] if not a.startswith('B_ext') else 'B'} {r['final_epoch_loss']:.2f}/{r['final_epoch_reward_accuracy']:.2f}"
        for a, r in g[f"{m}_e{e}"]["adapters"].items()) for m in MODELS for e in (2, 4) if f"{m}_e{e}" in g)
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Every adapter (4 epochs). Effective steps: optimizer steps "
          r"with a finite gradient and non-zero learning rate; fp16 loss scaling skips the first steps whose "
          r"gradients overflow. Learned: the declared learning check (final-epoch loss $\le$ 0.60 and reward "
          r"accuracy $\ge$ 0.75). Same as A: share of responses byte-identical to the untrained model's. "
          r"Learning check (loss/accuracy): " + gtxt + r".}", r"\label{tab:p2-training}", r"\end{table*}", ""]
    # audit
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"Cell & Audit TP/FP/FN/TN & Phase 1 TP/FP/FN/TN & Fisher $p$ (PPV, FOR) & Switch \\", r"\midrule"]
    for k, v in audit["cells"].items():
        au, p1, fp = v["audit"], v["phase1"], v["fisher_p"]
        t.append(f"{NAME[k.split('|')[0]]} {k.split('|')[1]} & {au['tp']}/{au['fp']}/{au['fn']}/{au['tn']} & "
                 f"{p1['tp']}/{p1['fp']}/{p1['fn']}/{p1['tn']} & {fp['ppv']:.3f}, {fp['for']:.3f} & "
                 f"{'yes' if v['switch'] else 'no'} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Post-training judge audit: one reviewer, blinded, 30 "
          r"judge-flagged and 30 unflagged responses per cell drawn equally from A, B and C. The declared rule "
          r"switches a cell to the audit's counts when either predictive value differs from Phase 1 ($p<.05$); "
          r"none did.}", r"\label{tab:p2-audit}", r"\end{table*}", ""]
    # composition (table view of Figure 2)
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}ll" + "c" * 4 + r"@{}}", r"\toprule",
          r"Model & Arm & EN & CS & RU & UR \\", r"\midrule"]
    for m in MODELS:
        for i, arm in enumerate(("A", "E", "B_ext", "C")):
            cells = [" / ".join(f"{100 * r[k]:.0f}" for k in KINDS) for cond in CONDS
                     for r in comp_rows if r["model"] == m and r["arm"] == arm and r["condition"] == cond]
            t.append(f"{NAME[m] if i == 0 else ''} & {SHORT[arm]} & " + " & ".join(cells) + r" \\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    t += [r"\end{tabular}", r"\caption{Response types (\% harmful / refusal / safe-helpful / non-response) on "
          r"the held-out harmful prompts; table view of Figure~\ref{fig:p2-composition}.}",
          r"\label{tab:p2-composition}", r"\end{table*}", ""]
    # residual taxonomy
    if residual:
        cnt = residual["counts"]
        pats = [p for p in residual["patterns"] if p != "not_harmful"] + ["not_harmful"]
        t += [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}l" + "r" * 5 + r"@{}}",
              r"\toprule",
              r"Pattern & " + " & ".join(CONDS) + r" & All \\", r"\midrule"]
        for p in pats:
            if p == "not_harmful":
                t.append(r"\midrule")
            t.append(p.replace("_", " ") + " & " + " & ".join(str(cnt.get(f"condition={c}", {}).get(p, 0)) for c in CONDS)
                     + f" & {cnt['all'].get(p, 0)} \\\\")
        conf = []
        for c in CONDS:
            ys = sum(cnt.get(f"cell={m}|{c}", {}).get("harmful_yes", 0) for m in MODELS)
            n = sum(sum(cnt.get(f"cell={m}|{c}", {}).values()) for m in MODELS)
            conf.append(f"{ys}/{n}")
        t.append(r"Confirmed harmful & " + " & ".join(conf) + f" & {sum(cnt['all'].values()) - cnt['all'].get('not_harmful', 0)}"
                 f"/{residual['n_rows']} \\\\")
        by_model = "; ".join(f"{NAME[m]} " + ", ".join(
            f"{c} {cnt.get(f'cell={m}|{c}', {}).get('harmful_yes', 0)}/{sum(cnt.get(f'cell={m}|{c}', {}).values())}"
            for c in CONDS) for m in MODELS)
        t += [r"\bottomrule", r"\end{tabular}", r"\caption{Residual errors after C: " + str(residual["n_rows"])
              + r" judge-flagged responses (8 per model and form, three seeds, sampled by a recorded rule), coded "
              r"blind by one reviewer: how the confirmed harmful responses comply, and how often the judge's flag "
              r"was confirmed (" + by_model + r"). Most RU/UR flags after training were not harmful.}",
              r"\label{tab:p2-residual}", r"\end{table}", ""]
    else:
        t += [r"% Residual-error taxonomy: pending (scripts/exp10_residuals.py apply, then re-run this script).", ""]
    return "\n".join(t)


def text(a, rates, comps, training, audit) -> str:
    ll, ph = (next(t for t in a["primary"] if t["model"] == m) for m in ("llama32", "phi3"))
    bad = [r["adapter"] for r in training if r["learned"] != "True"]
    return "\n".join([
        r"% ---------------------------------------------------------------------",
        r"% Section 5 (Mitigation) -- training and analysis as run",
        r"% ---------------------------------------------------------------------",
        r"We train QLoRA-DPO adapters (4-bit NF4 base, LoRA rank 16 on all attention and MLP projections, "
        r"$\beta=0.1$, learning rate $5\times10^{-5}$, effective batch 16, fp16 on T4 GPUs) on equal pair "
        r"budgets per model: 39 for Phi-3-mini and 60 for Llama-3.2, the number of reviewed CS pairs available "
        r"for Phi-3 and the planned cap for Llama. B uses PKU-SafeRLHF pairs with exactly one safe response; C uses "
        r"the model's own reviewed CS failures paired with written refusals. B and C are each trained with three "
        r"seeds. A learning check declared before training required a final-epoch DPO loss $\le 0.60$ and reward "
        r"accuracy $\ge 0.75$ on the seed-42 B and C adapters of both models; it failed at 2 epochs and passed at 4, "
        r"which all adapters then used (Table~\ref{tab:p2-training}).",
        r"\verify{Disclose: fp16 loss scaling skipped the first 2--4 optimizer steps of every adapter (gradient "
        r"overflow while the scale calibrates), so the main adapters received 5--9 effective updates; the 2-epoch "
        r"failure is largely this. Adapters " + ", ".join(b.replace("_", r"\_") for b in bad) + r" received 0--2 "
        r"effective updates and fail the same check: their results are reported as not trained. The first full "
        r"run stalled after the learning check; the reported run repeated it on three Kaggle accounts with the "
        r"same code, versions and 4 epochs, and re-recorded the check with the same outcome.}",
        "",
        r"Before any Phase-2 output was judged we fixed the analysis: for each family, C's and B's outcomes are the "
        r"mean judge label over the three seeds; the primary test is a two-sided paired sign-flip permutation test "
        r"over families, Holm-adjusted across the two models, and a contrast is a finding only if it is "
        r"Holm-significant and its judge-corrected interval excludes zero with the same sign. Judge error is "
        r"corrected with the Phase-1 predictive values of that model and condition unless a blinded audit of "
        r"post-training responses shows that they changed; the audit (300 responses, Table~\ref{tab:p2-audit}) "
        r"changed none.",
        r"\verify{Disclose: the audit's switch rule was reworded the same day, before any output was judged "
        r"(from ``outside the Phase-1 interval'' to a Fisher exact test).}",
        "",
        r"% ---------------------------------------------------------------------",
        r"% Section 6 -- Mitigation results",
        r"% ---------------------------------------------------------------------",
        rf"For Llama-3.2, training on its own CS failures reduces CS harmful compliance more than the same number of "
        rf"external English pairs: {num(ll['diff'], True)} points raw, {num(ll['diff_corrected'], True)} after judge "
        rf"correction (95\% interval {ci(ll['diff_corrected_ci_lo'], ll['diff_corrected_ci_hi'])}, Holm $p$ = "
        rf"{pval(ll['p_holm'])}; Table~\ref{{tab:rq4}}), with the same sign in every seed. For Phi-3-mini neither "
        rf"recipe moves harmful compliance ({num(ph['diff'], True)} points, $p$ = {pval(ph['p_holm'])}): its CS rate "
        r"is already near its floor, and most of its Roman-Urdu and Urdu-script responses are non-responses in every "
        r"arm (Figure~\ref{fig:p2-composition}). The Phi-3 result is limited evidence, not evidence of equal "
        r"effectiveness.",
        "",
        r"C lowers Llama's harmful compliance mainly by refusing more often, not by producing broken output "
        r"(Figure~\ref{fig:p2-composition}); it transfers to Roman Urdu but not to Urdu script, and slightly lowers "
        r"English harmful compliance (Figure~\ref{fig:p2-effects}). Neither recipe increases refusal of harmless "
        r"prompts, and capability is retained (Table~\ref{tab:mitigation}). The safety system prompt (E) gives the "
        r"largest reductions but makes both models refuse many harmless prompts, most of all in Roman Urdu. "
        r"Withholding the D6 domain from training leaves Llama's CS reduction on D6 similar in size, with intervals "
        r"too wide to separate the two (Table~\ref{tab:p2-ncurve}).",
        r"A blinded review of 64 responses that C still produced and the judge flagged (Table~\ref{tab:p2-residual}) "
        r"confirms nearly all English and most CS flags; the confirmed harmful responses mostly comply directly or "
        r"after a warning or a fictional or educational frame, and none arise from misreading the request.",
        r"\verify{After training, only 4/16 RU and 6/16 UR judge-flagged C responses were confirmed harmful "
        r"(Table~\ref{tab:p2-residual}), while RU/UR corrections use Phase-1 error rates that were not re-audited "
        r"after training and only C's flagged responses were reviewed (no unflagged sample, no A): treat the RU/UR "
        r"transfer estimates as uncertain and say so in Limitations.}",
        r"\verify{Phi-3 UrduMMLU is near chance (0.27) in every arm; Llama C UrduMMLU retention 0.949 (6 of 300 "
        r"items). The judge's precision on flagged Phi-3 CS responses was 23/30 in the audit vs 14/14 in Phase 1 "
        r"(not significant; Exp~6 review 44/66) -- state as a caveat.}",
    ])


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def table_examples() -> str:
    """Reviewer-written one-line summaries (outputs/exp10/residual_examples.csv), only with --examples:
    include them after a human has re-read the selection."""
    rows = list(csv.DictReader((ROOT / "outputs/exp10/residual_examples.csv").open(encoding="utf-8")))
    out = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}lllp{0.58\textwidth}@{}}", r"\toprule",
           r"Pattern & Form & Model & What the response does (reviewer's sanitized summary) \\", r"\midrule"]
    for r in rows:
        out.append(f"{r['pattern'].replace('_', ' ')} & {r['condition']} & {NAME[r['model']]} & "
                   f"{tex_escape(r['sanitized_summary'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\caption{Sanitized examples of residual harmful responses after C "
            r"(the first two confirmed responses per pattern in review order; operational details removed by the "
            r"reviewer).}", r"\label{tab:p2-examples}", r"\end{table*}"]
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true",
                    help="include the reviewer's sanitized example summaries (only after a human re-read them)")
    args = ap.parse_args(argv)
    a, rates, comps, training, audit, gates, residual = load()
    cap = {(c["model"], c["arm"]): c for c in a["capability"]}
    comp_rows = composition(a)
    write_csv(AN / "response_composition.csv", comp_rows)
    doc = "\n\n".join([
        "% " + "=" * 69,
        "% Phase 2 (Exp 7-10): paste-ready text, tables and figures for the Overleaf paper.",
        "% Generated by scripts/exp10_report.py from outputs/phase2_analysis/ -- do not edit numbers by hand.",
        "% Preamble needs: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18} \\usepgfplotslibrary{groupplots}",
        "% " + "=" * 69,
        text(a, rates, comps, training, audit),
        table_mitigation(a, rates, cap), table_rq4(a), figure_forest(comps), figure_composition(comp_rows),
        "% " + "-" * 69 + "\n% Appendix\n% " + "-" * 69,
        appendix(a, rates, comps, training, audit, gates, residual, comp_rows)]
        + ([table_examples()] if args.examples else
           ["% Sanitized examples: re-read outputs/exp10/residual_examples.csv, then run with --examples."]))
    out = ROOT / "docs" / "paper_phase2.tex"
    out.write_text(doc + "\n", encoding="utf-8", newline="\n")
    print(f"[exp10] wrote {out} ({len(doc.splitlines())} lines) and {AN / 'response_composition.csv'}"
          + ("" if residual else "; residual taxonomy pending"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
