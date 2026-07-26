"""Exp 6 — build DPO preference pairs.

A preference pair is {"prompt", "chosen", "rejected"}:
  - `rejected` = the SLM's actual harmful completion (mined from Exp 2 judgments),
  - `chosen`   = a natural code-switched refusal (generated on GPU/API elsewhere;
                 this module only consumes a base_id -> chosen-text map).

Pure-Python: mining, dedup, length-balancing, subsampling, and IO are all
CPU-testable. `chosen` generation itself is a thin LLM call done in the
experiment script, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from csjail.utils.io import read_jsonl, write_jsonl


@dataclass
class PrefBuildReport:
    n_mined: int
    n_with_chosen: int
    n_after_dedup: int
    n_final: int
    median_chosen_len: int
    median_rejected_len: int
    length_ratio: float          # median_chosen / median_rejected (char-based)
    within_tolerance: bool


def mine_rejected(
    judgment_rows: list[dict],
    assignments: dict[str, dict],
    *,
    condition: str = "CS",
    only_training_pool: bool = True,
    exclude_held_out_categories: bool = True,
    harmful_label: int = 1,
) -> list[dict]:
    """Select harmful completions to use as `rejected`.

    `judgment_rows`: dicts with at least base_id, condition, prompt, response,
    label (and optionally harm_category). `assignments`: from `splits.make_splits`.
    Keeps rows in the requested condition whose judge label == harmful_label and
    (optionally) are in the training pool and not in a held-out category.
    """
    out: list[dict] = []
    for r in judgment_rows:
        if r.get("condition") != condition:
            continue
        if r.get("label") != harmful_label:
            continue
        bid = r.get("base_id")
        assign = assignments.get(bid)
        if only_training_pool and (assign is None or assign.get("split") != "train"):
            continue
        if exclude_held_out_categories and assign and assign.get("held_out_category"):
            continue
        resp = (r.get("response") or "").strip()
        if not resp:
            continue
        out.append({
            "base_id": bid,
            "condition": condition,
            "harm_category": r.get("harm_category"),
            "prompt": r.get("prompt"),
            "rejected": resp,
        })
    return out


def _median(xs: list[int]) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) // 2


def assemble_pairs(
    mined: list[dict],
    chosen_by_base_id: dict[str, str],
    *,
    target_pairs: int = 800,
    length_tolerance: float = 0.15,
) -> tuple[list[dict], PrefBuildReport]:
    """Join mined `rejected` with `chosen` refusals; dedup; cap at target.

    Dedup is on (prompt, rejected). Length balancing is reported (median
    chosen/rejected char-length ratio); `within_tolerance` is True when the
    ratio is within +/- length_tolerance of 1.0. We report rather than fabricate
    text — if imbalanced, regenerate `chosen` at a matched length upstream.
    """
    n_mined = len(mined)
    joined: list[dict] = []
    for m in mined:
        chosen = chosen_by_base_id.get(m["base_id"])
        if not chosen:
            continue
        joined.append({
            "base_id": m["base_id"],
            "prompt": m["prompt"],
            "chosen": chosen.strip(),
            "rejected": m["rejected"],
            "harm_category": m.get("harm_category"),
            "condition": m.get("condition"),
        })
    n_with_chosen = len(joined)

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for p in joined:
        key = (p["prompt"] or "", p["rejected"] or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    n_after_dedup = len(deduped)

    final = deduped[:target_pairs]

    ch_lens = [len(p["chosen"]) for p in final]
    rej_lens = [len(p["rejected"]) for p in final]
    m_ch, m_rej = _median(ch_lens), _median(rej_lens)
    ratio = (m_ch / m_rej) if m_rej else 0.0
    within = abs(ratio - 1.0) <= length_tolerance if m_rej else False

    report = PrefBuildReport(
        n_mined=n_mined,
        n_with_chosen=n_with_chosen,
        n_after_dedup=n_after_dedup,
        n_final=len(final),
        median_chosen_len=m_ch,
        median_rejected_len=m_rej,
        length_ratio=round(ratio, 3),
        within_tolerance=within,
    )
    return final, report


def subsample(pairs: list[dict], n: int, *, seed: int = 42) -> list[dict]:
    """Deterministic subsample for the data-efficiency (N) ablation."""
    import numpy as np
    if n >= len(pairs):
        return list(pairs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[i] for i in sorted(idx.tolist())]


def load_english_pairs(path: str, *, limit: Optional[int] = None) -> list[dict]:
    """Load an off-the-shelf English refusal set already in {prompt,chosen,rejected}.

    Rows missing any of the three fields are skipped. Used for Arm B (English-only
    DPO), which needs none of our data.
    """
    rows = read_jsonl(path)
    out: list[dict] = []
    for r in rows:
        if all(k in r and r[k] for k in ("prompt", "chosen", "rejected")):
            out.append({
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            })
        if limit and len(out) >= limit:
            break
    return out


def write_pairs(path: str, pairs: list[dict]) -> None:
    write_jsonl(path, pairs)
