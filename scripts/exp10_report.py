#!/usr/bin/env python3
"""Exp 10: the Phase 2 results as LaTeX in the paper's own conventions (main.tex of the Overleaf project).

    python scripts/exp10_report.py            # after phase2_analysis.py and exp10_residuals.py apply

Reads outputs/phase2_analysis/{analysis.json, rates.csv, comparisons.csv, training.csv},
outputs/exp8/audit/audit_result.json, outputs/phase2_gate/*.json, outputs/exp10/residual_result.json,
the judged Exp 8 records (response types; local) and the training pairs (token counts; local).
Writes docs/paper_phase2.tex with two marked blocks -- PHASE2-SECTION (replaces the paper's Phase 2
"results pending" subsection: pairs and training, results, Table mitigation, Figure effects) and
PHASE2-APPENDIX (analysis rules and every supporting table and figure) -- plus
outputs/phase2_analysis/{response_composition.csv, pair_tokens.csv}. Every number comes from those
files; nothing is typed by hand. Figures are pgfplots (preamble: pgfplots, compat 1.18, groupplots);
colours are the dataviz reference categorical slots, validated; every plotted value is also in a table.
"""
from __future__ import annotations

import collections
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.aggregate import write_csv  # noqa: E402
from csjail.utils.io import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "outputs" / "phase2_analysis"
MODELS = ("phi3", "llama32")
NAME = {"phi3": "Phi-3-mini", "llama32": "Llama-3.2"}
B = r"B\textsubscript{ext}"
ARM = {"A": "A (untrained)", "E": "E (safety prompt)", "B_ext": B + " (external English)", "C": "C (own CS failures)"}
SHORT = {"A": "A", "E": "E", "B_ext": B, "C": "C"}
CONDS = ("EN", "CS", "RU", "UR")
KINDS = {"harmful": {"harmful_compliance"}, "refusal": {"refusal"}, "safe_helpful": {"safe_helpful"},
         "non_response": {"irrelevant", "unintelligible", "empty"}}
ARM_COLOR = {"C": "2A78D6", "B_ext": "EB6834", "E": "1BAF7A"}          # slots 1-3
ARM_MARK = {"C": "*", "B_ext": "square*", "E": "triangle*"}
MARK_SIZE = {"C": "2.5pt", "B_ext": "2.2pt", "E": "3.1pt"}     # similar visual area per shape
KIND_COLOR = {"harmful": "2A78D6", "refusal": "EB6834", "safe_helpful": "1BAF7A", "non_response": "EDA100"}
TOKENIZER = {"phi3": "microsoft--Phi-3-mini-4k-instruct", "llama32": "unsloth--Llama-3.2-3B-Instruct"}
MAX_LENGTH = 1024                                                # configs/dpo.yaml dpo.max_length


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
    residual = json.loads((ROOT / "outputs/exp10/residual_result.json").read_text(encoding="utf-8"))
    return a, rates, comps, training, audit, gates, residual


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


def pair_tokens(a) -> list[dict]:
    """Tokens of the pairs each main-budget adapter trained on (model's own tokenizer, no chat template)."""
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exp7_train_arms import take_budget

    from csjail.prefdata import load_external_english, read_pairs
    rows = []
    for t in a["primary"]:
        model, n = t["model"], t["budget"]
        tok = Tokenizer.from_file(str(next((ROOT / ".cache/tokenizers" / TOKENIZER[model]).glob("*/tokenizer.json"))))
        count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
        sets = {"C": take_budget(read_pairs(str(ROOT / f"outputs/exp6/{model}_verified/pairs_cs_all.jsonl")), str(n)),
                "B_ext": load_external_english(str(ROOT / "data/pref_pairs_en_external.jsonl"), limit=n)}
        for arm, pairs in sets.items():
            p, c, r = ([count(x[k]) for x in pairs] for k in ("prompt", "chosen", "rejected"))
            rows.append({"model": model, "arm": arm, "n_pairs": len(pairs), "prompt": statistics.fmean(p),
                         "chosen": statistics.fmean(c), "rejected": statistics.fmean(r),
                         "truncated": sum(pi + max(ci_, ri) > MAX_LENGTH for pi, ci_, ri in zip(p, c, r)) / len(pairs)})
    return rows


def table_mitigation(a, rates, cap) -> str:
    out = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{@{}llrcccrr@{}}",
           r"\toprule",
           r"Model & Arm & Pairs & CS \asr{} & EN \asr{} & Benign refusal & MMLU & UrduMMLU \\", r"\midrule"]
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
            r"\caption{Phase-2 results on the 200 held-out families (\%, 95\% intervals). \asr{} is "
            r"judge-error-corrected (raw differences in Table~\ref{tab:p2-transfer}); " + B + r" and C pool three "
            r"training seeds. Benign refusal: the 60 harmless prompts, averaged over their four forms (per form in "
            r"Table~\ref{tab:p2-orr}). MMLU: 500 items; UrduMMLU: 300 items (chance 0.25). E is a safety system "
            r"prompt, not trained.}",
            r"\label{tab:mitigation}", r"\end{table*}"]
    return "\n".join(out)


def table_rq4(a) -> str:
    out = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}lrcccccc@{}}", r"\toprule",
           r"Model & Pairs & C & " + B + r" & C$-$" + B + r" (raw) & C$-$" + B + r" (corrected) & $p_{\mathrm{Holm}}$ & "
           r"Per seed (42/43/44) \\", r"\midrule"]
    for t in a["primary"]:
        seeds = " / ".join(num(v["diff"], signed=True) for v in t["per_seed_mcnemar"].values())
        star = r"$^\ast$" if t["finding"] else ""
        out.append(f"{NAME[t['model']]} & {t['budget']} & {num(t['rate_a'])} & {num(t['rate_b'])} & "
                   f"{num(t['diff'], True)} {ci(t['diff_ci_lo'], t['diff_ci_hi'])} & "
                   f"{num(t['diff_corrected'], True)} {ci(t['diff_corrected_ci_lo'], t['diff_corrected_ci_hi'])} & "
                   f"{pval(t['p_holm'])}{star} & {seeds} \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Primary Phase-2 contrast (RQ4): CS harmful compliance (\%) after training on the model's own "
            r"CS failures (C) versus the same number of external English pairs (" + B + r"), 200 held-out families. "
            r"Each family's outcome is the mean judge label over three training seeds; $p$ from a paired sign-flip "
            r"permutation test over families (100{,}000 draws), Holm-adjusted across the two models. $^\ast$Finding: "
            r"Holm-significant and the corrected interval excludes zero with the same sign. Per-seed values are "
            r"single-seed paired differences (robustness, not the test).}",
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
           rf"  width=0.49\textwidth, height=4.6cm, xmin={lo}, xmax={hi}, ymin=0.5, ymax=4.5,",
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
            r"judge-corrected difference; bars: 95\% family-bootstrap intervals; 200 held-out families; " + B
            + r" and C pool three seeds). Values in Table~\ref{tab:p2-transfer}.}",
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
            r"prompt, B = " + B + r" external English pairs, C own CS failures (" + B + r" and C pool three seeds). "
            r"Values in Table~\ref{tab:p2-composition}.}",
            r"\label{fig:p2-composition}", r"\end{figure*}"]
    return "\n".join(out)


def section(a, rates, comps, training, audit, residual, comp_rows, tokens, cap) -> str:
    """The paper's Phase-2 subsections after the design: pairs and training, then results."""
    ll, ph = (next(t for t in a["primary"] if t["model"] == m) for m in ("llama32", "phi3"))
    bad = [r for r in training if r["learned"] != "True"]
    good = [r for r in training if r["learned"] == "True"]
    eff = sorted({int(r["effective_steps"]) for r in good})
    lost = sorted({int(r["steps"]) - int(r["effective_steps"]) for r in training})
    steps = sorted({int(r["steps"]) for r in training})
    tk = {(r["model"], r["arm"]): r for r in tokens}
    ab = {r["model"]: r for r in comps if r["kind"] == "ablation"}
    abl = {(r["model"], r["comparison"]): r for r in comps if r["kind"] in ("ablation", "ablation_reference")
           and r["condition"] == "CS"}
    nc = {(r["model"], int(r["budget"])): r for r in comps if r["kind"] == "ncurve" and r["condition"] == "CS"}
    va = {(r["model"], r["comparison"], r["condition"]): r for r in comps if r["kind"] == "vs_A"}
    cp = {(r["model"], r["arm"], r["condition"]): r for r in comp_rows}
    orr = {(m, arm): float(rate(rates, "overrefusal", m, arm, "ALL")["rate"]) for m in MODELS for arm in ("A", "E", "B_ext", "C")}
    drop = max(cap[(m, "A")][k] - cap[(m, arm)][k] for m in MODELS for arm in ("B_ext", "C") for k in ("mmlu", "urdummlu"))
    bmax = max(abs(float(va[("llama32", "B_ext-A", c)]["diff_corrected"])) for c in CONDS)
    pruru = max(float(rate(rates, "overrefusal", m, "E", "RU")["rate"]) for m in MODELS)
    cnt = residual["counts"]
    conf = {c: (sum(cnt.get(f"cell={m}|{c}", {}).get("harmful_yes", 0) for m in MODELS),
                sum(sum(cnt.get(f"cell={m}|{c}", {}).values()) for m in MODELS)) for c in CONDS}
    pats = {p: cnt["all"].get(p, 0) for p in residual["patterns"]}
    nab = {m: int(ab[m]["budget"]) for m in MODELS}
    seeds = [num(v["diff"], True) for v in ll["per_seed_mcnemar"].values()]
    t = [r"\subsection{Pairs and training}", r"\label{sec:p2training}",
         r"From the production sweep, validated CS failures on the 591 training-pool families numbered 223 for "
         r"Llama-3.2 and 66 for Phi-3. Chosen refusals were generated few-shot by a model distinct from the judge "
         r"(GPT-4.1; an Anthropic generator declined the harmful-context prompts), and every candidate passed the "
         r"automatic clean-refusal screen (judged a genuine refusal, not unsafe: 66/66 and 223/223). A blinded human "
         r"check of 156 candidate pairs (all 66 for Phi-3 and the first 90 of Llama-3.2's seeded order) kept those "
         r"whose rejected answer was confirmed harmful and whose refusal was clean: 39 and 61. The budgets are "
         rf"therefore {ph['budget']} pairs for Phi-3 and {ll['budget']} for Llama-3.2, equal for C and {B} within a "
         r"model, with nested prefixes of one seeded, domain-aware ordering at 25 pairs (both models) and 50 "
         rf"(Llama-3.2). The sources differ in length: C's rejected answers average "
         rf"{min(tk[(m, 'C')]['rejected'] for m in MODELS):.0f}--{max(tk[(m, 'C')]['rejected'] for m in MODELS):.0f} "
         rf"tokens against {min(tk[(m, 'C')]['chosen'] for m in MODELS):.0f}--"
         rf"{max(tk[(m, 'C')]['chosen'] for m in MODELS):.0f} for its refusals, while {B}'s responses average "
         rf"{min(tk[(m, 'B_ext')][k] for m in MODELS for k in ('chosen', 'rejected')):.0f}--"
         rf"{max(tk[(m, 'B_ext')][k] for m in MODELS for k in ('chosen', 'rejected')):.0f} "
         rf"(Table~\ref{{tab:p2-tokens}}); we report this rather than pad. {B} draws from 1,000 screened PKU-SafeRLHF pairs \citep{{dai2024safe}} (leakage screen "
         r"against the evaluation set: 0 hits). An unseen-domain ablation (D6, drawn by a rule declared before the "
         rf"draw) withholds the domain from all supervision in both arms (budgets {nab['phi3']} and {nab['llama32']}) "
         r"and trains fresh adapters.",
         "",
         r"All adapters use QLoRA-DPO (4-bit NF4 base, LoRA rank 16 on all attention and MLP projections, "
         r"$\beta{=}0.1$, learning rate $5{\times}10^{-5}$, effective batch 16, fp16 on T4 GPUs); " + B + r" and C at "
         r"the full budget are trained with seeds 42, 43 and 44. A learning check declared before training "
         r"(final-epoch DPO loss ${\le}0.60$ and reward accuracy ${\ge}0.75$ on the seed-42 adapters of both models) "
         r"failed at 2 epochs and passed at 4, which all adapters then used. With budgets this small, fp16 loss "
         r"scaling \citep{micikevicius2018mixed} matters: it skips each optimiser step whose gradients overflow while "
         rf"the scale calibrates, which cost every adapter {lost[0]}--{lost[-1]} of its {steps[0]}--{steps[-1]} steps "
         rf"and largely explains the 2-epoch failure. The main adapters kept {eff[0]}--{eff[-1]} effective updates and "
         r"pass the check; the two 25-pair adapters and Phi-3's two unseen-domain adapters kept "
         rf"{min(int(r['effective_steps']) for r in bad)}--{max(int(r['effective_steps']) for r in bad)}, fail it, and "
         r"are reported as untrained (Table~\ref{tab:p2-training}). The analysis rules were fixed before any Phase-2 "
         r"output was judged (Appendix~\ref{app:phase2}).",
         "",
         r"\subsection{Results}", r"\label{sec:p2results}",
         r"Table~\ref{tab:mitigation} summarises every arm. For Llama-3.2, training on its own CS failures lowers CS "
         rf"harmful compliance more than the same number of external English pairs: C {num(ll['rate_a'])}\% versus "
         rf"{B} {num(ll['rate_b'])}\% (raw, three seeds pooled), a difference of {num(ll['diff'], True)} points "
         rf"{ci(ll['diff_ci_lo'], ll['diff_ci_hi'])} and {num(ll['diff_corrected'], True)} "
         rf"{ci(ll['diff_corrected_ci_lo'], ll['diff_corrected_ci_hi'])} after judge-error correction (Holm "
         rf"$p{{=}}${pval(ll['p_holm'])}; every seed agrees in sign: {', '.join(seeds)}; Table~\ref{{tab:rq4}}). This "
         r"is the Phase-2 finding. For Phi-3, neither recipe moves CS harm (C$-$" + B + rf" "
         rf"{num(ph['diff'], True)} {ci(ph['diff_ci_lo'], ph['diff_ci_hi'])}, $p{{=}}${pval(ph['p_holm'])}) although "
         rf"its adapters fit their training pairs and changed most of its responses; with {ph['budget']} pairs and a "
         rf"{num(rate(rates, 'asr', 'phi3', 'A', 'CS')['rate'], nd=0)}\% base rate this is limited evidence, not "
         r"evidence of equal effectiveness.",
         "",
         rf"C works for Llama-3.2 by refusing more: its CS refusals rise from {100 * cp[('llama32', 'A', 'CS')]['refusal']:.0f}\% "
         rf"to {100 * cp[('llama32', 'C', 'CS')]['refusal']:.0f}\% while non-response stays at "
         rf"{100 * cp[('llama32', 'A', 'CS')]['non_response']:.0f}--{100 * cp[('llama32', 'C', 'CS')]['non_response']:.0f}\% "
         r"(Figure~\ref{fig:p2-composition}). Relative to the untrained model, its reduction carries over to Roman "
         rf"Urdu ({num(va[('llama32', 'C-A', 'RU')]['diff_corrected'], True)} points) but not to Urdu script "
         rf"({num(va[('llama32', 'C-A', 'UR')]['diff_corrected'], True)}), with a small English gain "
         rf"({num(va[('llama32', 'C-A', 'EN')]['diff_corrected'], True)}; Figure~\ref{{fig:p2-effects}}); {B} changes "
         rf"Llama-3.2's harm by at most {100 * bmax:.0f} points in any form. Neither trained recipe raises refusal of "
         rf"harmless prompts ({100 * min(orr[(m, a_)] for m in MODELS for a_ in ('B_ext', 'C')):.1f}--"
         rf"{100 * max(orr[(m, a_)] for m in MODELS for a_ in ('B_ext', 'C')):.1f}\% versus "
         rf"{100 * min(orr[(m, 'A')] for m in MODELS):.1f}\% untrained) or costs more than {100 * drop:.0f} points of "
         r"MMLU or UrduMMLU. The safety prompt E gives the largest reductions (Llama-3.2 CS "
         rf"{num(va[('llama32', 'E-A', 'CS')]['diff_corrected'], True)}) but makes the models refuse "
         rf"{100 * orr[('llama32', 'E')]:.0f}--{100 * orr[('phi3', 'E')]:.0f}\% of harmless prompts (up to "
         rf"{100 * pruru:.0f}\% in Roman Urdu). Fifty pairs already give Llama-3.2 the full-budget effect "
         rf"({num(nc[('llama32', 50)]['diff_corrected'], True)} versus {num(nc[('llama32', ll['budget'])]['diff_corrected'], True)} "
         r"at 60, seed 42). With D6 withheld from training, Llama-3.2's CS reduction on that domain is similar in size "
         rf"({num(abl[('llama32', 'C-A')]['diff_corrected'], True)} versus "
         rf"{num(abl[('llama32', 'C(main, trained with it)-A')]['diff_corrected'], True)} with D6 in training; "
         rf"{abl[('llama32', 'C-A')]['n_families']} families, wide intervals). A blinded review of 64 harmful responses "
         rf"that C still produced (Table~\ref{{tab:p2-residual}}) confirmed {conf['EN'][0]}/{conf['EN'][1]} English and "
         rf"{conf['CS'][0]}/{conf['CS'][1]} CS judge flags but only {conf['RU'][0]}/{conf['RU'][1]} Roman-Urdu and "
         rf"{conf['UR'][0]}/{conf['UR'][1]} Urdu-script ones; the confirmed residuals comply directly ({pats['direct']}), "
         rf"partially ({pats['partial']}), inside a fictional or educational frame ({pats['reframed']}) or after a "
         rf"warning ({pats['warning_then_comply']}), and none from misreading the request.",
         "",
         table_mitigation(a, rates, cap), "", figure_forest(comps)]
    return "\n".join(t)


def appendix(a, rates, comps, training, audit, gates, residual, comp_rows, tokens) -> str:
    t = [r"\section{Phase 2 details}", r"\label{app:phase2}",
         r"\paragraph{Analysis rules.} Before any Phase-2 output was judged we fixed the analysis: for each family, "
         r"C's and " + B + r"'s outcomes are the mean judge label over the three seeds; the primary test (RQ4, C "
         r"versus " + B + r" on CS) is a two-sided paired sign-flip permutation test over families, Holm-adjusted "
         r"across the two models, and a contrast is a finding only if it is Holm-significant and its "
         r"judge-corrected interval excludes zero with the same sign (the Phase-1 rule). Judge error is corrected "
         r"with the Phase-1 predictive values of that model and form unless a blinded audit of post-training "
         r"responses shows that they changed (two-sided Fisher exact test, $p{<}.05$, for either predictive value; "
         r"the wording of this rule was tightened the same day, still before any output was judged). The audit "
         r"(Table~\ref{tab:p2-audit}) changed none. Comparisons with the untrained model, data efficiency, transfer, "
         r"over-refusal and the unseen domain are secondary and reported with unadjusted intervals. The first full "
         r"training run stopped after the learning check; the reported run repeated it with the same code, software "
         r"versions and settings and re-recorded the check with the same outcome.",
         "",
         table_rq4(a), "", figure_composition(comp_rows), ""]
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
          r"(points; judge-corrected with 95\% interval, raw below). Values of Figure~\ref{fig:p2-effects}.}",
          r"\label{tab:p2-transfer}", r"\end{table*}", ""]
    # over-refusal per form
    t += [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
          r"Model & Arm & EN & CS & RU & UR & All \\", r"\midrule"]
    for m in MODELS:
        for i, arm in enumerate(("A", "E", "B_ext", "C")):
            t.append(f"{NAME[m] if i == 0 else ''} & {SHORT[arm]} & "
                     + " & ".join(num(rate(rates, "overrefusal", m, arm, c)["rate"]) for c in CONDS + ("ALL",)) + r" \\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    agree = ", ".join(f"{NAME[k.split('|')[0]]} {k.split('|')[1]} {v['agreement']:.2f}" for k, v in audit["benign"].items())
    t += [r"\end{tabular}", r"\caption{Refusal of the 60 harmless prompts per form (\%, uncorrected). On RU/UR "
          r"responses the harmless-prompt judge agreed with a blinded human check at " + agree + r" ($n{=}15$ each).}",
          r"\label{tab:p2-orr}", r"\end{table}", ""]
    # n-curve and unseen domain
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}llcl@{}}", r"\toprule",
          r"Model & Comparison & CS, C $-$ A (corrected) & Trained \\", r"\midrule"]
    for r in comps:
        if r["kind"] == "ncurve" and r["condition"] == "CS":
            t.append(f"{NAME[r['model']]} & C, {r['budget']} pairs (seed 42) & {num(r['diff_corrected'], True)} "
                     f"{ci(r['diff_corrected_ci_lo'], r['diff_corrected_ci_hi'])} & "
                     f"{'yes' if r['adapter_learned'] == 'True' else 'no'} \\\\")
    t.append(r"\midrule")
    for r in comps:
        if r["kind"] in ("ablation", "ablation_reference") and r["condition"] == "CS":
            lab = {"C-A": "D6 withheld: C $-$ A", "B_ext-A": "D6 withheld: " + B + " $-$ A",
                   "C-B_ext": "D6 withheld: C $-$ " + B, "C(main, trained with it)-A": "D6 in training: C $-$ A"}[r["comparison"]]
            learned = "yes" if r["kind"] == "ablation_reference" or r.get("adapter_learned") == "True" else "no"
            t.append(f"{NAME[r['model']]} & {lab} ($n{{=}}{r['n_families']}$) & {num(r['diff_corrected'], True)} "
                     f"{ci(r['diff_corrected_ci_lo'], r['diff_corrected_ci_hi'])} & {learned} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Data efficiency (C at each pair budget) and the unseen "
          r"domain (D6, drawn by a rule fixed before the draw; its held-out families only). ``Trained'' applies the "
          r"declared learning check to that adapter's own training log (Table~\ref{tab:p2-training}); results of "
          r"untrained adapters are not evidence. Phi-3 gave no harmful D6 response in any arm.}",
          r"\label{tab:p2-ncurve}", r"\end{table*}", ""]
    # training
    t += [r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrrl@{}}", r"\toprule",
          r"Adapter & Pairs & Steps & Effective & Final-epoch loss & Reward acc. & Same as A & Learned \\",
          r"\midrule"]
    for r in training:
        t.append(f"\\texttt{{{r['adapter'].replace('_', chr(92) + '_')}}} & {r['n_pairs']} & {r['steps']} & "
                 f"{r['effective_steps']} & {float(r['final_epoch_loss']):.3f} & {float(r['final_epoch_reward_accuracy']):.2f} & "
                 f"{num(r['identical_to_A'])}\\% & {'yes' if r['learned'] == 'True' else 'no'} \\\\")
    gtxt = "; ".join(f"{NAME[m]} {e} epochs: " + ", ".join(
        f"{'C' if a_.startswith('C_') else B} {r['final_epoch_loss']:.2f}/{r['final_epoch_reward_accuracy']:.2f}"
        for a_, r in gates[f"{m}_e{e}"]["adapters"].items()) for m in MODELS for e in (2, 4) if f"{m}_e{e}" in gates)
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Every adapter (4 epochs; seeds in the name, 42 when absent). "
          r"Effective steps: optimiser steps with a finite gradient and a non-zero learning rate (fp16 loss scaling "
          r"skips the first steps whose gradients overflow). Learned: the declared learning check (final-epoch loss "
          r"${\le}0.60$ and reward accuracy ${\ge}0.75$). Same as A: share of responses byte-identical to the untrained "
          r"model's. Learning check on the seed-42 adapters (loss/accuracy): " + gtxt + r".}",
          r"\label{tab:p2-training}", r"\end{table*}", ""]
    # tokens
    t += [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrr@{}}",
          r"\toprule", r"Model & Arm & Prompt & Chosen & Rejected & ${>}1{,}024$ \\", r"\midrule"]
    for m in MODELS:
        for i, arm in enumerate(("B_ext", "C")):
            r = next(x for x in tokens if x["model"] == m and x["arm"] == arm)
            t.append(f"{NAME[m] if i == 0 else ''} & {SHORT[arm]} & {r['prompt']:.0f} & "
                     f"{r['chosen']:.0f} & {r['rejected']:.0f} & {num(r['truncated'], nd=0)}\\% \\\\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    t += [r"\end{tabular}", r"\caption{Mean tokens per training pair (each model's own tokenizer, no chat template) "
          r"and the share of pairs whose prompt plus longer response exceeds the 1{,}024-token training length "
          r"(truncated in training).}", r"\label{tab:p2-tokens}", r"\end{table}", ""]
    # audit
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"Cell & Audit TP/FP/FN/TN & Phase 1 TP/FP/FN/TN & Fisher $p$ (PPV, FOR) & Switch \\", r"\midrule"]
    for k, v in audit["cells"].items():
        au, p1, fp = v["audit"], v["phase1"], v["fisher_p"]
        t.append(f"{NAME[k.split('|')[0]]} {k.split('|')[1]} & {au['tp']}/{au['fp']}/{au['fn']}/{au['tn']} & "
                 f"{p1['tp']}/{p1['fp']}/{p1['fn']}/{p1['tn']} & {fp['ppv']:.3f}, {fp['for']:.3f} & "
                 f"{'yes' if v['switch'] else 'no'} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Post-training judge audit: blinded human labels for 30 "
          r"judge-flagged and 30 unflagged responses per cell, drawn equally from A, " + B + r" and C (all seeds). The "
          r"declared rule switches a cell to the audit's counts when either predictive value differs from Phase 1 "
          r"($p{<}.05$); none did.}", r"\label{tab:p2-audit}", r"\end{table*}", ""]
    # composition (table view of Figure p2-composition)
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}ll" + "c" * 4 + r"@{}}", r"\toprule",
          r"Model & Arm & EN & CS & RU & UR \\", r"\midrule"]
    for m in MODELS:
        for i, arm in enumerate(("A", "E", "B_ext", "C")):
            cells = [" / ".join(f"{100 * r[k]:.0f}" for k in KINDS) for cond in CONDS
                     for r in comp_rows if r["model"] == m and r["arm"] == arm and r["condition"] == cond]
            t.append(f"{NAME[m] if i == 0 else ''} & {SHORT[arm]} & " + " & ".join(cells) + r" \\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    t += [r"\end{tabular}", r"\caption{Response types (\% harmful / refusal / safe-helpful / non-response) on "
          r"the held-out harmful prompts; values of Figure~\ref{fig:p2-composition}.}",
          r"\label{tab:p2-composition}", r"\end{table*}", ""]
    # residual taxonomy
    cnt = residual["counts"]
    pats = [p for p in residual["patterns"] if p != "not_harmful"] + ["not_harmful"]
    t += [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}l" + "r" * 5 + r"@{}}",
          r"\toprule", r"Pattern & " + " & ".join(CONDS) + r" & All \\", r"\midrule"]
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
          + r" judge-flagged responses (8 per model and form, all seeds, sampled by a recorded rule), labelled blind: "
          r"how the confirmed harmful responses comply, and how often the judge's flag was confirmed (" + by_model
          + r"). After training, most RU/UR flags were not harmful, so post-training RU/UR rates, corrected with "
          r"Phase-1 error rates, are likely overstated.}", r"\label{tab:p2-residual}", r"\end{table}", ""]
    return "\n".join(t)


def main(argv=None) -> int:
    a, rates, comps, training, audit, gates, residual = load()
    cap = {(c["model"], c["arm"]): c for c in a["capability"]}
    comp_rows = composition(a)
    tokens = pair_tokens(a)
    write_csv(AN / "response_composition.csv", comp_rows)
    write_csv(AN / "pair_tokens.csv", tokens)
    doc = "\n".join([
        "% " + "=" * 69,
        "% Phase 2 (Exp 7-10) in the paper's conventions. Generated by scripts/exp10_report.py from",
        "% outputs/phase2_analysis/ -- do not edit numbers by hand. Preamble: \\usepackage{pgfplots}",
        "% \\pgfplotsset{compat=1.18} \\usepgfplotslibrary{groupplots}; cites dai2024safe, micikevicius2018mixed.",
        "% " + "=" * 69,
        "%%% BEGIN PHASE2-SECTION",
        section(a, rates, comps, training, audit, residual, comp_rows, tokens, cap),
        "%%% END PHASE2-SECTION",
        "",
        "%%% BEGIN PHASE2-APPENDIX",
        appendix(a, rates, comps, training, audit, gates, residual, comp_rows, tokens),
        "%%% END PHASE2-APPENDIX"])
    out = ROOT / "docs" / "paper_phase2.tex"
    out.write_text(doc + "\n", encoding="utf-8", newline="\n")
    print(f"[exp10] wrote {out} ({len(doc.splitlines())} lines), response_composition.csv, pair_tokens.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
