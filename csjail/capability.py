"""Exp 8 — capability retention (MMLU / UrduMMLU multiple-choice accuracy).

Checks that safety fine-tuning did not degrade general ability. Uses the same
`SLMRunner` (vLLM) for generation; only the prompt-building, answer-parsing, and
scoring are here and are CPU-testable. Generation is GPU work done by the caller.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Optional

from csjail.utils.io import read_jsonl

_LETTER_RE = re.compile(r"\b([A-J])\b")


@dataclass
class MCQItem:
    question: str
    choices: list[str]
    answer_idx: int   # 0-based index of the correct choice
    subject: Optional[str] = None


def load_mcq(path: str) -> list[MCQItem]:
    """Load MCQ items from JSONL: {question, choices:[...], answer_idx|answer}."""
    out: list[MCQItem] = []
    for r in read_jsonl(path):
        choices = list(r["choices"])
        if "answer_idx" in r:
            idx = int(r["answer_idx"])
        else:
            # allow a letter answer, e.g. "B"
            ans = str(r["answer"]).strip().upper()
            idx = string.ascii_uppercase.index(ans)
        out.append(MCQItem(
            question=r["question"],
            choices=choices,
            answer_idx=idx,
            subject=r.get("subject"),
        ))
    return out


def build_prompt(item: MCQItem) -> str:
    """Zero-shot MCQ prompt; model is asked to answer with a single letter."""
    lines = [item.question, ""]
    for i, ch in enumerate(item.choices):
        lines.append(f"{string.ascii_uppercase[i]}. {ch}")
    lines.append("")
    lines.append("Answer with the single letter of the correct option.")
    return "\n".join(lines)


def parse_choice(text: str, n_choices: int) -> Optional[int]:
    """Extract a 0-based choice index from a model answer. None if unparseable."""
    if not text:
        return None
    valid = set(string.ascii_uppercase[:n_choices])
    # Prefer the first standalone A-J letter.
    for m in _LETTER_RE.finditer(text.upper()):
        letter = m.group(1)
        if letter in valid:
            return string.ascii_uppercase.index(letter)
    # Fallback: first character if it's a valid letter.
    first = text.strip()[:1].upper()
    if first in valid:
        return string.ascii_uppercase.index(first)
    return None


def score_predictions(
    pred_indices: list[Optional[int]], items: list[MCQItem]
) -> dict:
    """Accuracy over items; unparseable predictions count as wrong."""
    assert len(pred_indices) == len(items)
    correct = sum(
        1 for p, it in zip(pred_indices, items, strict=True)
        if p is not None and p == it.answer_idx
    )
    n = len(items)
    n_unparsed = sum(1 for p in pred_indices if p is None)
    return {
        "accuracy": (correct / n) if n else 0.0,
        "n": n,
        "n_correct": correct,
        "n_unparsed": n_unparsed,
    }


def evaluate_mcq(runner, items: list[MCQItem], *, max_tokens: int = 8) -> dict:
    """Full MCQ eval using an SLMRunner (GPU). Returns score_predictions dict."""
    from csjail.models import SamplingConfig

    prompts = [build_prompt(it) for it in items]
    outputs = runner.generate(
        prompts, SamplingConfig(temperature=0.0, max_tokens=max_tokens),
    )
    preds = [parse_choice(o, len(it.choices))
             for o, it in zip(outputs, items, strict=True)]
    return score_predictions(preds, items)
