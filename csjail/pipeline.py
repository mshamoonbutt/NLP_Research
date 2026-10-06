"""Persist-first, resumable generation and judging (docs/PROTOCOL.md §6.2).

Stage 1 (generate) appends each chunk to `generations.jsonl` as soon as it is
produced, BEFORE any judging. Stage 2 (judge) appends to `judgments.jsonl`.
An interrupted run is resumed by re-running the same command: successful
cached generations/judgments are reused, failed ones retried.

Cache identity
  generation key = dataset version + row id + prompt hash + model identity
                   (model, revision, chat template, arm, adapter hash, system
                   prompt hash) + sampling config (+ sample index per output)
  judgment key   = generation key + response hash + judge fingerprint
So a rubric edit re-uses generations but invalidates judgments; a changed
prompt/model/template/adapter invalidates the affected generations.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from csjail.artifacts import sha256_json, sha256_text
from csjail.data import Prompt
from csjail.outcomes import STATUS_OK

GEN_FIELDS_FROM_ROW = ("base_id", "domain_id", "domain_name", "condition", "group_id",
                       "dataset_version")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlCache:
    """Append-only JSONL keyed by `key_field`; later records win."""

    def __init__(self, path: str | Path, key_field: str):
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            data = self.path.read_bytes()
            end = data.rfind(b"\n") + 1
            if end < len(data):
                # A run killed mid-write (time limit, lost session) leaves a partial
                # final line; appending after it would corrupt the next record.
                print(f"WARN: {self.path}: dropping an incomplete final line "
                      f"({len(data) - end} bytes) left by an interrupted run", file=sys.stderr)
                with self.path.open("r+b") as f:
                    f.truncate(end)
            for line in data[:end].decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    r = json.loads(line)
                    self.records[r[self.key_field]] = r

    def append(self, records: Iterable[dict]) -> None:
        recs = list(records)
        if not recs:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        for r in recs:
            self.records[r[self.key_field]] = r


def generation_key(row: Prompt, identity: dict, sampling: dict) -> str:
    return sha256_json({"dataset_version": row.dataset_version, "row_id": row.id,
                        "prompt_sha256": sha256_text(row.prompt),
                        "identity": identity, "sampling": sampling})


def run_generation(generate_fn: Callable[[list[str]], list[list[Any]]], rows: list[Prompt], *,
                   identity: dict, sampling: dict, cache: JsonlCache,
                   split_lookup: Optional[dict[str, dict]] = None, split_id: Optional[str] = None,
                   chunk_size: int = 64, log: Callable[[str], None] = print) -> list[dict]:
    """`generate_fn(prompts) -> [[GenOutput x n] per prompt]`. Returns one
    record per (row, sample_index), from cache when possible."""
    n = int(sampling.get("n", 1))
    todo, keys = [], {}
    for r in rows:
        k = generation_key(r, identity, sampling)
        keys[r.id] = k
        cached = [cache.records.get(f"{k}#{i}") for i in range(n)]
        if not all(c is not None and c.get("generation_status") == "ok" for c in cached):
            todo.append(r)
    if todo:
        log(f"[gen] {identity.get('model_key')}/{identity.get('arm')}: "
            f"{len(rows) - len(todo)} cached, {len(todo)} to generate")
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        try:
            outs = generate_fn([r.prompt for r in chunk])
            err = None
        except Exception as e:  # infrastructure failure: record, retry on resume
            outs, err = [[None] * n for _ in chunk], f"{type(e).__name__}: {e}"
        recs = []
        for r, samples in zip(chunk, outs, strict=True):
            for i in range(n):
                g = samples[i] if i < len(samples) else None
                base = {
                    "kind": "generation", "gen_key": f"{keys[r.id]}#{i}",
                    "gen_group_key": keys[r.id], "row_id": r.id,
                    **{f: getattr(r, f) for f in GEN_FIELDS_FROM_ROW},
                    "model": identity.get("model_key"), "arm": identity.get("arm"),
                    "model_identity": identity, "sampling": sampling, "sample_index": i,
                    "split": (split_lookup or {}).get(r.base_id, {}).get("split"),
                    "split_id": split_id, "prompt_sha256": sha256_text(r.prompt),
                    "created_utc": _now(),
                }
                if g is None:
                    recs.append({**base, "generation_status": "failed",
                                 "generation_error": err or "no output returned",
                                 "response": None, "response_sha256": None})
                else:
                    recs.append({**base, "generation_status": "ok", "response": g.text,
                                 "response_sha256": sha256_text(g.text),
                                 "finish_reason": g.finish_reason,
                                 "n_prompt_tokens": g.n_prompt_tokens,
                                 "n_completion_tokens": g.n_completion_tokens})
        cache.append(recs)
    return [cache.records[f"{keys[r.id]}#{i}"] for r in rows for i in range(n)]


def judgment_key(gen: dict, fingerprint: dict) -> str:
    return sha256_json({"gen_key": gen["gen_key"], "response_sha256": gen["response_sha256"],
                        "judge": fingerprint["fingerprint_id"]})


def run_judging(judge, gens: list[dict], prompts_by_row: dict[str, str], *, cache: JsonlCache,
                chunk_size: int = 256, log: Callable[[str], None] = print) -> dict[str, dict]:
    """Judge every successful generation not already judged OK under this
    judge fingerprint. Returns {gen_key: judgment record}."""
    fp = judge.fingerprint
    todo = [g for g in gens if g["generation_status"] == "ok"
            and cache.records.get(judgment_key(g, fp), {}).get("judge_status") != STATUS_OK]
    if todo:
        log(f"[judge] {len(todo)} to judge ({fp['rubric_kind']}, {fp['fingerprint_id']})")
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        js = judge.score_sync([(prompts_by_row[g["row_id"]], g["response"]) for g in chunk],
                              show_progress=False)
        cache.append({"kind": "judgment", "judge_key": judgment_key(g, fp), "gen_key": g["gen_key"],
                      "judge_fingerprint_id": fp["fingerprint_id"], "created_utc": _now(),
                      **j.as_record_fields()} for g, j in zip(chunk, js, strict=True))
    out = {}
    for g in gens:
        if g["generation_status"] == "ok":
            j = cache.records.get(judgment_key(g, fp))
            if j is not None:
                out[g["gen_key"]] = j
    return out


def merge_results(gens: list[dict], judgments: dict[str, dict], *,
                  judge_fingerprint_id: Optional[str]) -> list[dict]:
    """One result record per generation; unjudged/failed stay explicit."""
    out = []
    for g in gens:
        rec = {k: v for k, v in g.items() if k != "kind"}
        rec["kind"] = "result"
        j = judgments.get(g["gen_key"])
        if g["generation_status"] != "ok":
            rec["judge_status"] = "not_run_generation_failed"
        elif j is None:
            rec["judge_status"] = "not_run"
        else:
            rec.update({k: v for k, v in j.items()
                        if k.startswith("judge_") and k != "judge_key"})
        rec["judge_fingerprint_id"] = judge_fingerprint_id
        out.append(rec)
    return out
