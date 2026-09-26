"""GPU smoke: load one SLM and generate for a few harmless fixture prompts.

    python scripts/smoke_test.py [--model qwen25]

Checks that the pinned model loads, that its official chat template exists,
prints the template probe (so any default system text is visible), and that
outputs carry finish reasons and token counts. It does not judge anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from csjail.data import load_dataset  # noqa: E402
from csjail.models import SamplingConfig, SLMRunner, resolve  # noqa: E402

FIXTURE = Path(__file__).parent.parent / "data" / "csjail_fixture.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    rows = load_dataset(FIXTURE)[: args.n]
    runner = SLMRunner(resolve(args.model), require_pinned=True)
    print(f"[smoke] template sha256 {runner.chat_template_sha256[:12]}; probe:\n"
          f"{runner.template_probe}")
    outs = runner.generate([r.prompt for r in rows], SamplingConfig(max_tokens=64),
                           show_progress=False)
    bad = 0
    for r, o in zip(rows, outs, strict=True):
        g = o[0]
        print(f"[smoke] {r.id:<22} finish={g.finish_reason} tokens={g.n_completion_tokens} "
              f"-> {g.text.strip()[:80]!r}")
        bad += not g.text.strip()
    runner.shutdown()
    print("[smoke] OK" if not bad else f"[smoke] {bad} empty outputs (record, not fatal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
