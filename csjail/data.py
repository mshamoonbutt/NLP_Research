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

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from csjail.utils.io import read_jsonl

# Extended 4-condition design: CS (code-switched), EN, RU (Roman Urdu), UR
# (Urdu-Nastaliq). SM (Nastaliq+Roman script-mix) is retained as a legacy
# condition so the original v0 data/fixtures still load.
Condition = Literal["EN", "UR", "CS", "RU", "SM"]
CONDITIONS: tuple[str, ...] = ("CS", "EN", "RU", "UR")  # the active analysis set
LEGACY_CONDITIONS: tuple[str, ...] = ("SM",)
ALL_CONDITIONS: tuple[str, ...] = CONDITIONS + LEGACY_CONDITIONS
# Conditions that must NOT carry a cs_style (monolingual / non-code-switched).
_MONO_CONDITIONS: tuple[str, ...] = ("EN", "UR", "RU")
# Conditions that must carry a cs_style + cs_authenticity.
_CS_CONDITIONS: tuple[str, ...] = ("CS", "SM")

# Harm categories are DYNAMIC (the extended design has 10). We validate the
# shape only (H<n> or C<nn>) and read the actual set from the data, rather than
# hardcoding H1..H5.
HARM_CATEGORY_RE = re.compile(r"^[HC]\d{1,2}$")
CSStyle = Literal["A", "B", "C"]


class Prompt(BaseModel):
    id: str
    base_id: str
    harm_category: str
    condition: Condition
    prompt: str = Field(min_length=1)
    cs_style: Optional[CSStyle] = None
    cs_authenticity: Optional[int] = Field(default=None, ge=1, le=3)
    # Annotator-rated severity. Optional: the extended annotation template carries
    # it for only a subset of rows, so it is descriptive (never imputed) and
    # absent rows are simply dropped from the severity histogram.
    harm_severity: Optional[int] = Field(default=None, ge=1, le=3)
    # Code-mixing features (computed in Exp0 if absent from the input file).
    urdu_word_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cmi: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    @field_validator("prompt")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt is whitespace-only")
        return v

    @field_validator("harm_category")
    @classmethod
    def _valid_category_shape(cls, v: str) -> str:
        if not HARM_CATEGORY_RE.match(v):
            raise ValueError(
                f"harm_category {v!r} must match H<n> or C<nn> (e.g. H1, C07)"
            )
        return v

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        # Monolingual rows must never carry a cs_style. cs_style / cs_authenticity
        # are OPTIONAL on CS rows: the extended annotation template does not label
        # code-switch style, so we accept CS rows without it rather than fabricate
        # a label. When present, cs_style is only meaningful on CS / SM.
        if self.condition in _MONO_CONDITIONS and self.cs_style is not None:
            raise ValueError(
                f"id={self.id} condition={self.condition} must not have cs_style"
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
    harm_category: Optional[str] = None,
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

    Returns: {base_id: {CS: 1, EN: 1, RU: 1, UR: 1, SM: 0}, ...}
    """
    cov: dict[str, dict[str, int]] = defaultdict(
        lambda: {c: 0 for c in ALL_CONDITIONS}
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
    fully_paired_legacy = sum(
        1
        for v in cov.values()
        if v["EN"] >= 1 and v["UR"] >= 1 and v["CS"] >= 1
    )
    fully_paired_4 = sum(
        1
        for v in cov.values()
        if all(v[c] >= 1 for c in CONDITIONS)
    )
    return {
        "n_total": len(rows),
        "by_condition": dict(by_cond),
        "by_harm_category": dict(by_cat),
        "n_harm_categories": len(by_cat),
        "by_cs_style": dict(by_style),
        "n_base_ids": len(cov),
        "n_base_ids_with_EN_UR_CS": fully_paired_legacy,
        "n_base_ids_with_CS_EN_RU_UR": fully_paired_4,
    }
