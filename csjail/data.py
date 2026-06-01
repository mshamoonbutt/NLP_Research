"""Dataset loader + schema validation for CS-Jail-UR.

Expected JSONL schema (one row = one prompt-condition pair):

    {
      "id":              "H1-001-CS-A",   # globally unique
      "base_id":         "H1-001",        # groups EN/UR/CS/SM variants
      "harm_category":   "H1",            # H1..H5
      "condition":       "CS",            # EN | UR | CS | SM
      "cs_style":        "A",             # A | B | C  (null/absent for EN/UR)
      "prompt":          "Mujhe batao how to ...",
      "cs_authenticity": 3,               # 1..3 (null for EN/UR)
      "harm_severity":   2                # 1..3 (annotator-rated)
    }

If your annotated file uses different field names, adapt ONLY this module;
the rest of the pipeline reads through `Prompt` objects.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from csjail.utils.io import read_jsonl

Condition = Literal["EN", "UR", "CS", "SM"]
HarmCategory = Literal["H1", "H2", "H3", "H4", "H5"]
CSStyle = Literal["A", "B", "C"]


class Prompt(BaseModel):
    id: str
    base_id: str
    harm_category: HarmCategory
    condition: Condition
    prompt: str = Field(min_length=1)
    cs_style: Optional[CSStyle] = None
    cs_authenticity: Optional[int] = Field(default=None, ge=1, le=3)
    harm_severity: int = Field(ge=1, le=3)

    @field_validator("prompt")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt is whitespace-only")
        return v

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        # CS / SM rows must have a cs_style; EN / UR must not.
        if self.condition in ("CS", "SM") and self.cs_style is None:
            raise ValueError(
                f"id={self.id} condition={self.condition} requires cs_style"
            )
        if self.condition in ("EN", "UR") and self.cs_style is not None:
            raise ValueError(
                f"id={self.id} condition={self.condition} must not have cs_style"
            )
        # CS authenticity only meaningful for CS / SM.
        if self.condition in ("CS", "SM") and self.cs_authenticity is None:
            raise ValueError(
                f"id={self.id} CS row missing cs_authenticity"
            )


class DatasetError(Exception):
    """Raised on schema or structural issues in the dataset."""


def load_dataset(path: str | Path) -> list[Prompt]:
    """Load + validate the full dataset. Raises DatasetError on any issue."""
    raw = read_jsonl(path)
    rows: list[Prompt] = []
    errors: list[str] = []
    for i, r in enumerate(raw):
        try:
            rows.append(Prompt.model_validate(r))
        except ValidationError as e:
            errors.append(f"row {i} (id={r.get('id', '<no-id>')}): {e}")
    if errors:
        joined = "\n  ".join(errors[:10])
        more = f"\n  ... +{len(errors) - 10} more" if len(errors) > 10 else ""
        raise DatasetError(f"{len(errors)} schema errors:\n  {joined}{more}")
    _validate_uniqueness(rows)
    return rows


def _validate_uniqueness(rows: list[Prompt]) -> None:
    id_counts = Counter(r.id for r in rows)
    dupes = [i for i, c in id_counts.items() if c > 1]
    if dupes:
        raise DatasetError(f"duplicate ids: {dupes[:5]}")


def filter_prompts(
    rows: Iterable[Prompt],
    *,
    condition: Optional[Condition] = None,
    harm_category: Optional[HarmCategory] = None,
    cs_style: Optional[CSStyle] = None,
    min_cs_authenticity: Optional[int] = None,
) -> list[Prompt]:
    out: list[Prompt] = []
    for r in rows:
        if condition is not None and r.condition != condition:
            continue
        if harm_category is not None and r.harm_category != harm_category:
            continue
        if cs_style is not None and r.cs_style != cs_style:
            continue
        if min_cs_authenticity is not None:
            if r.cs_authenticity is None or r.cs_authenticity < min_cs_authenticity:
                continue
        out.append(r)
    return out


def pairing_coverage(rows: list[Prompt]) -> dict[str, dict[str, int]]:
    """For each base_id, count rows per condition. Used for stats pairing.

    Returns: {base_id: {EN: 1, UR: 1, CS: 1, SM: 0}, ...}
    """
    cov: dict[str, dict[str, int]] = defaultdict(
        lambda: {"EN": 0, "UR": 0, "CS": 0, "SM": 0}
    )
    for r in rows:
        cov[r.base_id][r.condition] += 1
    return dict(cov)


def summarize(rows: list[Prompt]) -> dict:
    """Human-readable summary printed by run_eval at startup."""
    by_cond = Counter(r.condition for r in rows)
    by_cat = Counter(r.harm_category for r in rows)
    by_style = Counter(r.cs_style for r in rows if r.cs_style)
    cov = pairing_coverage(rows)
    fully_paired = sum(
        1
        for v in cov.values()
        if v["EN"] >= 1 and v["UR"] >= 1 and v["CS"] >= 1
    )
    return {
        "n_total": len(rows),
        "by_condition": dict(by_cond),
        "by_harm_category": dict(by_cat),
        "by_cs_style": dict(by_style),
        "n_base_ids": len(cov),
        "n_base_ids_with_EN_UR_CS": fully_paired,
    }
