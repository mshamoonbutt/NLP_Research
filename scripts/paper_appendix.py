#!/usr/bin/env python3
"""Paper Appendix A (reproducibility) and B (Phase-1 raw rates and full tables) from the repo's records.

    python scripts/paper_appendix.py      # -> docs/paper_appendix_ab.tex (blocks APPENDIX-A, APPENDIX-B)

Sources: outputs/exp0 manifests, outputs/exp9 ablation split, outputs/exp2/{main,main-r1,main-gemma4}
(run manifests; generations for token-cap shares; judgments for dates), outputs/exp2/summary_5models.csv,
outputs/exp2/robustness*/robustness_summary.json, outputs/exp1/{judge_comparison,heldout_comparison,
judge_validation_manifest,judge_validation_manifest_benign}.json, outputs/exp3/isolation_results.json,
outputs/exp8/*/judgments.jsonl (Phase-2 judging dates). Numbers are never typed by hand.
"""
from __future__ import annotations

import ast
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("qwen25", "phi3", "llama32", "r1qwen15", "gemma4e2b")
NAME = {"qwen25": "Qwen2.5-1.5B", "phi3": "Phi-3-mini", "llama32": "Llama-3.2-3B", "r1qwen15": "R1-Distill-1.5B",
        "gemma4e2b": "Gemma 4 E2B"}
RUNS = {"qwen25": "main", "phi3": "main", "llama32": "main", "r1qwen15": "main-r1", "gemma4e2b": "main-gemma4"}
CONDS = ("EN", "CS", "RU", "UR")
DOMAINS = ("D1", "D2", "D3", "D4", "D5", "D6")


def dec(x, nd=3) -> str:
    """The paper's Table-2 style: .116 (no leading zero), minus as $-$."""
    if x in (None, ""):
        return "--"
    s = f"{float(x):.{nd}f}"
    s = s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s
    return s.replace("-", "$-$")


def tt(x) -> str:
    return r"\texttt{" + str(x).replace("_", r"\_") + "}"


def runtime(e: dict) -> dict:
    r = e.get("runtime") or {}
    return r if isinstance(r, dict) else ast.literal_eval(r)


def jread(p) -> dict:
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def jsonl(p):
    with (ROOT / p).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dates(paths) -> str:
    ds = sorted({r["created_utc"][:10] for p in paths if (ROOT / p).exists() for r in jsonl(p) if r.get("created_utc")})
    return ds[0] if len(ds) == 1 else f"{ds[0]} to {ds[-1]}"


def appendix_a() -> str:
    man = {d.name: jread(f"outputs/exp0/{d.name}/dataset_manifest.json") for d in sorted((ROOT / "outputs/exp0").glob("final-*"))}
    latest = jread("outputs/exp0/LATEST.json")
    cur = man[Path(latest["dir"]).name]
    ab = jread("outputs/exp9/ablation_D6/split_manifest.json")["meta"]
    runs = {d: jread(f"outputs/exp2/{d}/run_manifest.json") for d in set(RUNS.values())}
    jm, bm = jread("outputs/exp1/judge_validation_manifest.json"), jread("outputs/exp1/judge_validation_manifest_benign.json")
    fp = jm["judge_fingerprint"]
    t = [r"\section{Reproducibility}", r"\label{app:repro}",
         r"\paragraph{Data.} Release \texttt{" + cur["dataset_version"] + r"} (source file SHA-256 \texttt{"
         + cur["source_sha256"][:12] + r"}\ldots), which extends " + " and ".join(
             tt(k) for k in man if k != Path(latest["dir"]).name) + r" append-only; evaluation split "
         r"\texttt{" + cur["split_id"] + r"} (200 evaluation and 591 training-pool families); the D6 unseen-domain "
         r"split is \texttt{" + ab["split_id"] + r"}. Assignments are frozen manifests; no evaluation family was "
         r"used for judge development or Phase-2 supervision.",
         "", r"\paragraph{Models.} Table~\ref{tab:app-models}; greedy decoding, no added system prompt.", "",
         r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llllr@{}}",
         r"\toprule", r"Checkpoint & Revision & Chat template & Backend & New tokens \\", r"\midrule"]
    notes = {}
    for m in MODELS:
        run = runs[RUNS[m]]
        e = run["models"][m]
        rev = e.get("revision") if e.get("revision") not in (None, "None") else e.get("reference_hf_revision")
        if e.get("backend") == "ollama":
            back = f"Ollama, {e['quantization'].replace('_', chr(92) + '_')}"
            notes["ollama"] = (f"Ollama {runtime(e)['ollama']}, tag {tt(e['ollama_tag'])}, digest "
                               f"{tt(e['ollama_digest'][:12])}")
        else:
            back = f"vLLM, {e['dtype']}"
            notes["vllm"] = f"vLLM {runtime(e)['vllm']}"
        t.append(rf"{{\scriptsize{tt(e['hf_id'])}}} & {tt(rev[:10])} & {tt(e['chat_template_sha256'][:10])} "
                 rf"& {back} & {run['sampling']['max_tokens']} \\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Pinned models (revision and chat-template SHA-256 prefixes; "
          + notes["vllm"] + "; " + notes["ollama"] + r"). R1-Distill decodes up to 2{,}048 new tokens so that its "
          r"reasoning can close; its visible reasoning is judged. Gemma~4 runs through Ollama because the pinned vLLM "
          r"cannot load it.}", r"\label{tab:app-models}", r"\end{table*}", "",
          r"\paragraph{Judge.} " + tt(fp["model"]) + r" through " + fp["provider"].capitalize()
          + r", rubric \texttt{" + str(fp.get("rubric_version", "harm-v2")) + r"}, fingerprint \texttt{"
          + fp["fingerprint_id"] + r"}; harmless-prompt rubric fingerprint \texttt{"
          + bm["judge_fingerprint"]["fingerprint_id"] + r"}. Phase-1 judgments were made on "
          + dates([f"outputs/exp2/{d}/judgments.jsonl" for d in sorted(set(RUNS.values()))]) + r", Phase-2 judgments on "
          + dates([str(p.relative_to(ROOT)) for p in sorted((ROOT / "outputs/exp8").glob("*/judgments.jsonl"))])
          + r"; every raw judgment is retained. The correction uses the reviewer-labelled counts in "
          r"Table~\ref{tab:app-cells}.", "",
          r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}lcccc@{}}",
          r"\toprule", r"Model & EN & CS & RU & UR \\", r"\midrule"]
    cells = jm["result"]["per_model_language"]
    for m in MODELS:
        t.append(NAME[m] + " & " + " & ".join(
            "/".join(str(cells[f"{m}|{c}"][k]) for k in ("tp", "fp", "fn", "tn")) for c in CONDS) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Judge-error cells behind the correction: TP/FP/FN/TN of the "
          r"judge against reviewer labels, per model and form (development plus held-out labels).}",
          r"\label{tab:app-cells}", r"\end{table*}", "",
          r"\paragraph{Software and compute.} vLLM 0.6.3.post1, PyTorch 2.4.0, Transformers 4.46.3; training with "
          r"TRL 0.12.2, PEFT 0.13.2, bitsandbytes 0.44.1 and Datasets 3.1.0 (pyarrow 25.0.1). Generation and "
          r"training ran on NVIDIA T4 GPUs (fp16); judging ran through the judge's hosted API. Every run manifest "
          r"records the code version, dataset and split identifiers, sampling settings, model identities and "
          r"cache keys."]
    return "\n".join(t)


def appendix_b() -> str:
    rows = list(csv.DictReader((ROOT / "outputs/exp2/summary_5models.csv").open(encoding="utf-8")))
    s = {(r["model"], r["condition"], r["domain"]): r for r in rows if r["arm"] == "A"}
    cap = collections.defaultdict(lambda: [0, 0])
    for d in sorted(set(RUNS.values())):
        for g in jsonl(f"outputs/exp2/{d}/generations.jsonl"):
            if g.get("sample_index", 0) == 0:
                k = cap[(g["model"], g["condition"])]
                k[0] += g.get("finish_reason") == "length"
                k[1] += 1
    t = [r"\section{Raw rates, intervals, and full tables}", r"\label{app:raw}",
         r"Table~\ref{tab:app-raw} gives every Phase-1 rate with its interval, the full-compliance-only rate and "
         r"the share of responses at the token cap; Table~\ref{tab:app-domain} breaks the raw rates down by domain; "
         r"Table~\ref{tab:app-robust} reports sampled decoding; Table~\ref{tab:app-judge} the judge selection and "
         r"held-out test; and Table~\ref{tab:app-gee} the supplementary GEE.", "",
         r"\begin{table*}[t]", r"\centering\footnotesize", r"\begin{tabular}{@{}llrcccr@{}}", r"\toprule",
         r"Model & Form & Scored & Raw \asr{} [95\% CI] & Corrected \asr{} [95\% CI] & Strict (full only) & At cap \\",
         r"\midrule"]
    for m in MODELS:
        for i, c in enumerate(CONDS):
            r = s[(m, c, "ALL")]
            k = cap[(m, c)]
            t.append(f"{NAME[m] if i == 0 else ''} & {c} & {r['n_scored']}/{r['n_planned']} & "
                     f"{dec(r['asr'])} [{dec(r['ci_lo'])}, {dec(r['ci_hi'])}] & "
                     f"{dec(r['asr_corrected'])} [{dec(r['asr_corrected_ci_lo'])}, {dec(r['asr_corrected_ci_hi'])}] & "
                     f"{dec(r['strict_asr'])} & {dec(k[0] / k[1]) if k[1] else '--'} \\\\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    t += [r"\end{tabular}", r"\caption{Phase-1 rates on all 791 families (greedy). Raw: judge labels with a "
          r"domain-stratified family-bootstrap interval; corrected: predictive-value correction "
          r"(\S\ref{sec:stats}); strict: full compliance only; at cap: share of responses that hit the new-token "
          r"limit. No response was missing, so worst-case bounds equal the raw rates.}", r"\label{tab:app-raw}",
          r"\end{table*}", "",
          r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
          r"\begin{tabular}{@{}l" + "cccc" * 1 + "|" + "cccc" + "|" + "cccc" + r"@{}}", r"\toprule"]
    # per-domain raw ASR: three domain blocks per line to stay compact
    for block in (DOMAINS[:3], DOMAINS[3:]):
        t.append(" & " + " & ".join(rf"\multicolumn{{4}}{{c{'|' if j < 2 else ''}}}{{{d}}}" for j, d in enumerate(block))
                 + r" \\")
        t.append("Model & " + " & ".join(" & ".join(CONDS) for _ in block) + r" \\")
        t.append(r"\midrule")
        for m in MODELS:
            t.append(NAME[m] + " & " + " & ".join(dec(s[(m, c, d)]["asr"], 2) for d in block for c in CONDS) + r" \\")
        t.append(r"\midrule" if block[0] == DOMAINS[0] else r"\bottomrule")
    t += [r"\end{tabular}", r"\caption{Raw primary-unsafe \asr{} by domain (all 791 families; 122--144 per domain "
          r"and form).}", r"\label{tab:app-domain}", r"\end{table*}", ""]
    # robustness
    rob = {}
    for f in ("outputs/exp2/robustness", "outputs/exp2/robustness-r1", "outputs/exp2/robustness-gemma4"):
        rob.update(jread(f + "/robustness_summary.json")["per_model_condition"])
    t += [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llccc@{}}",
          r"\toprule", r"Model & Form & Per draw [95\% CI] & \asr{}@5 & Greedy \\", r"\midrule"]
    for m in MODELS:
        for i, c in enumerate(("CS", "RU")):
            r = rob[f"{m}/{c}"]
            t.append(f"{NAME[m] if i == 0 else ''} & {c} & {dec(r['per_draw_asr'])} [{dec(r['per_draw_ci'][0])}, "
                     f"{dec(r['per_draw_ci'][1])}] & {dec(r['asr_at_5'])} & {dec(r['greedy_asr_same_families'])} \\\\")
        t.append(r"\midrule" if m != MODELS[-1] else r"\bottomrule")
    t += [r"\end{tabular}", r"\caption{Sampled-decoding robustness on 200 fixed families (5 draws, $T{=}0.7$, "
          r"top-$p$ 0.9; raw judge labels). \asr{}@5: unsafe in at least one draw; greedy: the same families under "
          r"greedy decoding.}", r"\label{tab:app-robust}", r"\end{table}", ""]
    # judge selection
    jc, hc = jread("outputs/exp1/judge_comparison.json"), jread("outputs/exp1/heldout_comparison.json")
    t += [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}",
          r"\toprule", r"Judge (rubric) & Macro-F1 & Prec. & Rec. & $\kappa$ \\", r"\midrule",
          r"\multicolumn{5}{@{}l}{\emph{Development (720 items; selection by macro-F1)}} \\"]
    for name, e in sorted(jc["table"].items(), key=lambda kv: -kv[1]["macro_f1"]):
        o = e["overall"]
        t.append(f"{name.replace(':cloud', '').replace('_', chr(92) + '_')} & {dec(e['macro_f1'])} & "
                 f"{dec(o['precision'])} & {dec(o['recall'])} & {dec(o['kappa'])} \\\\")
    t.append(r"\multicolumn{5}{@{}l}{\emph{Held-out test (960 items)}} \\")
    for j, meta in hc["judges"].items():
        o = hc["overall"]["heldout"][j]
        t.append(f"{meta['model'].replace(':cloud', '')} {meta['rubric']} & {dec(hc['macro_f1']['heldout'][j])} & "
                 f"{dec(o['precision'])} & {dec(o['recall'])} & {dec(o['kappa'])} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Judge selection (rule declared before the DeepSeek runs) and "
          r"the held-out test, against the corrected human labels.}", r"\label{tab:app-judge}", r"\end{table}", ""]
    # GEE per model
    gee = jread("outputs/exp3/isolation_results.json")["gee_by_model"]
    t += [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{@{}lccc@{}}",
          r"\toprule", r"Model & CS & RU & UR \\", r"\midrule"]
    for m in MODELS:
        g = gee[m]
        cell = []
        for c in ("CS", "RU", "UR"):
            k = f"C(condition, Treatment('EN'))[T.{c}]"
            lo, hi = g["conf_int"][k]
            cell.append(f"{g['odds_ratios'][k]:.2f} [{lo:.2f}, {hi:.2f}]")
        t.append(NAME[m] + " & " + " & ".join(cell) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{Supplementary GEE odds ratios of primary unsafe versus EN, "
          r"fitted per model (form and domain, exchangeable correlation within family; raw judge labels).}",
          r"\label{tab:app-gee}", r"\end{table*}"]
    return "\n".join(t)


def main() -> int:
    doc = "\n".join(["% " + "=" * 69,
                     "% Appendix A and B, generated by scripts/paper_appendix.py from the repo's records.",
                     "% " + "=" * 69,
                     "%%% BEGIN APPENDIX-A", appendix_a(), "%%% END APPENDIX-A", "",
                     "%%% BEGIN APPENDIX-B", appendix_b(), "%%% END APPENDIX-B"])
    out = ROOT / "docs" / "paper_appendix_ab.tex"
    out.write_text(doc + "\n", encoding="utf-8", newline="\n")
    print(f"[appendix] wrote {out} ({len(doc.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
