#!/usr/bin/env python3
"""Benign Urdu-script sanity check of a generation setup (no harmful content).

Runs six trivial Urdu-script prompts and their English translations through
each model with the production greedy settings and saves the outputs, so a
human can judge whether Urdu-script degeneration is a model property or a
setup error (checkpoint, chat template, tokenizer, generation settings).

    python scripts/urdu_sanity_check.py --backend vllm --out outputs/checks/urdu_sanity_vllm.json
    python scripts/urdu_sanity_check.py --backend ollama --out outputs/checks/urdu_sanity_ollama.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

UR = ["پاکستان کا دارالحکومت کون سا شہر ہے؟",
      "چائے بنانے کا آسان طریقہ بتائیں۔",
      "بارش کے موسم میں گھر کے پودوں کی دیکھ بھال کیسے کریں؟",
      "دوستی کے بارے میں ایک مختصر نظم لکھیں۔",
      "اس جملے کا انگریزی میں ترجمہ کریں: میں کل بازار جاؤں گا۔",
      "صحت مند رہنے کے تین مشورے دیں۔"]
EN = ["Which city is the capital of Pakistan?",
      "Tell me an easy way to make tea.",
      "How should I care for house plants in the rainy season?",
      "Write a short poem about friendship.",
      "Translate this sentence into English: I will go to the market tomorrow. "
      "(Urdu: میں کل بازار جاؤں گا)",
      "Give three tips for staying healthy."]


def repeat_share(text: str) -> float:
    w = text.split()
    return round(1 - len(set(w)) / len(w), 2) if w else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "ollama"], required=True)
    ap.add_argument("--models", nargs="+", default=["qwen25", "phi3"])
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from csjail.models import SamplingConfig, resolve

    root = Path(__file__).resolve().parent.parent
    s = dict(yaml.safe_load((root / "configs" / "eval.yaml").read_text(encoding="utf-8"))["sampling"])
    s["max_tokens"] = args.max_tokens
    sampling = SamplingConfig(**s)
    out = {"backend": args.backend, "sampling": sampling.as_dict(), "results": []}
    for key in args.models:
        if args.backend == "ollama":
            from csjail.ollama_backend import OllamaRunner
            runner = OllamaRunner(resolve(key))
        else:
            from csjail.models import SLMRunner
            runner = SLMRunner(resolve(key), require_pinned=True)
        try:
            prov = runner.provenance()
            for lang, prompts in (("UR", UR), ("EN", EN)):
                for i, (p, g) in enumerate(zip(prompts, runner.generate(prompts, sampling))):
                    g = g[0]
                    out["results"].append({
                        "model": key, "lang": lang, "i": i, "prompt": p, "text": g.text,
                        "finish_reason": g.finish_reason, "prompt_tokens": g.n_prompt_tokens,
                        "completion_tokens": g.n_completion_tokens,
                        "repeat_word_share": repeat_share(g.text)})
            out.setdefault("provenance", {})[key] = {k: prov.get(k) for k in (
                "hf_id", "revision", "ollama_tag", "ollama_digest", "quantization",
                "chat_template_sha256", "default_system", "runtime")}
        finally:
            runner.shutdown()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for key in args.models:
        for lang in ("UR", "EN"):
            rs = [r for r in out["results"] if r["model"] == key and r["lang"] == lang]
            print(f"{key} {lang}: mean repeated-word share "
                  f"{sum(r['repeat_word_share'] for r in rs) / len(rs):.2f}, "
                  f"hit length cap {sum(r['finish_reason'] == 'length' for r in rs)}/{len(rs)}")
    print(f"outputs -> {args.out}  (read the texts; the numbers are only a hint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
