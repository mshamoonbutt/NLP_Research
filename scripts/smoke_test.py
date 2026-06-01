"""End-to-end smoke test for the inference harness.

Runs the smallest SLM (Qwen2.5-1.5B) over 10 fixture prompts at T=0 and
verifies every output is non-empty. Target wall time: < 90s on a single A100.

Usage (on the A100 host):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --model phi3   # try another SLM
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow `python scripts/smoke_test.py` without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from csjail.data import filter_prompts, load_dataset
from csjail.models import SLMRunner, SamplingConfig, resolve

FIXTURE = Path(__file__).parent.parent / "data" / "csjail_fixture.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25", help="model key from configs/models.yaml")
    ap.add_argument("--n", type=int, default=10, help="how many prompts to run")
    args = ap.parse_args()

    print(f"[smoke] loading fixture {FIXTURE}")
    rows = load_dataset(FIXTURE)
    sample = (
        filter_prompts(rows, condition="EN")[: args.n // 2]
        + filter_prompts(rows, condition="CS")[: args.n // 2]
    )[: args.n]
    print(f"[smoke] sampled {len(sample)} prompts ({sum(1 for r in sample if r.condition=='EN')} EN, "
          f"{sum(1 for r in sample if r.condition=='CS')} CS)")

    spec = resolve(args.model)
    print(f"[smoke] loading model {spec.hf_id} ...")
    t0 = time.time()
    runner = SLMRunner(spec)
    print(f"[smoke] load took {time.time() - t0:.1f}s")

    t0 = time.time()
    outs = runner.generate(
        [r.prompt for r in sample],
        SamplingConfig(temperature=0.0, max_tokens=128, seed=0),
        show_progress=False,
    )
    dt = time.time() - t0
    print(f"[smoke] generate took {dt:.1f}s ({dt / max(len(outs), 1):.2f}s/prompt)")

    bad = 0
    for r, o in zip(sample, outs, strict=True):
        if not o or not o.strip():
            print(f"[smoke] EMPTY OUTPUT for id={r.id}", file=sys.stderr)
            bad += 1
        else:
            preview = o.strip().replace("\n", " ")[:120]
            print(f"[smoke] {r.id:<18} [{r.condition}] -> {preview}")

    runner.shutdown()
    if bad:
        print(f"[smoke] FAIL: {bad} empty outputs", file=sys.stderr)
        return 1
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
