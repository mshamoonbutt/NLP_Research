"""Dataset schema, loader and structural validation for CS-Jail-UR.

Active long-format schema (one row = one family x language condition), as
produced by `csjail.convert_final` from the seven-column final CSV:

    {
      "id":           "CSJUR-V3-0001::CS",   # make_row_id(base_id, condition)
      "base_id":      "CSJUR-V3-0001",       # opaque family ID (string, never int-cast)
      "domain_id":    "D6",                  # validated against configs/domains.yaml
      "domain_name":  "Cyber abuse and unauthorized access",
      "harm_category":"D6",                  # compatibility alias == domain_id
      "condition":    "CS",                  # CS | EN | RU | UR
      "prompt":       "...",                 # the text sent to the model
      "dataset_version": "final-692-<hash>",
      "provenance":   {...},                 # source row, ID namespace, normalization
      "group_id":     null,                  # duplicate/relative group (set in Exp 0)
      ...optional descriptive metadata (never required, never invented)...
    }

Legacy v0/v1 rows (H1..H5 / C01..C10 categories, the SM condition, a `_meta`
blob) still load with `allow_legacy=True` so historical runs can be reproduced,
but they are rejected by default: legacy SM must not enter active analysis.

Unknown fields are rejected (`extra="forbid"`) instead of being silently dropped;
metadata lives in the explicit `provenance` / `qa` fields.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from csjail.utils.io import read_jsonl

SCHEMA_VERSION = "csjail-longform-v2"

Condition = Literal["EN", "UR", "CS", "RU", "SM"]
CONDITIONS: tuple[str, ...] = ("CS", "EN", "RU", "UR")  # the active analysis set
LEGACY_CONDITIONS: tuple[str, ...] = ("SM",)
ALL_CONDITIONS: tuple[str, ...] = CONDITIONS + LEGACY_CONDITIONS
_MONO_CONDITIONS: tuple[str, ...] = ("EN", "UR", "RU")

# Row identity: a collision-safe encoding of (family ID, condition). Family IDs
# may not contain the separator, so the encoding is reversible.
ROW_ID_SEP = "::"

DOMAIN_ID_RE = re.compile(r"^D\d{1,2}$")
LEGACY_CATEGORY_RE = re.compile(r"^[HC]\d{1,2}$")
CSStyle = Literal["A", "B", "C"]


def make_row_id(base_id: str, condition: str) -> str:
    if ROW_ID_SEP in base_id:
        raise ValueError(f"family id {base_id!r} must not contain {ROW_ID_SEP!r}")
    return f"{base_id}{ROW_ID_SEP}{condition}"


def split_row_id(row_id: str) -> tuple[str, str]:
    base_id, sep, condition = row_id.rpartition(ROW_ID_SEP)
    if not sep:
        raise ValueError(f"row id {row_id!r} is not a {ROW_ID_SEP!r}-encoded id")
    return base_id, condition


class Provenance(BaseModel):
    """Where a row came from. Unknown values stay None -- never invented."""

    model_config = ConfigDict(extra="forbid")

    source_file: Optional[str] = None
    source_sha256: Optional[str] = None
    source_row: Optional[int] = None          # 1-based data row in the source CSV
    source_column: Optional[str] = None       # e.g. "CS"
    id_namespace: Optional[str] = None        # descriptive, derived from the ID pattern
    lineage_hint: Optional[str] = None        # e.g. "replacement-of:0246" (from the ID only)
    author: Optional[str] = None              # None = unknown (the final CSV has no ledger)
    model_assistance: Optional[str] = None    # None = unknown
    text_normalization: Optional[str] = None  # exact transformation applied, if any
    source_text: Optional[str] = None         # original text when normalization changed it
    legacy: Optional[dict[str, Any]] = None   # the old converters' `_meta` blob


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    base_id: str
    condition: Condition
    prompt: str = Field(min_length=1)
    harm_category: str
    domain_id: Optional[str] = None
    domain_name: Optional[str] = None
    group_id: Optional[str] = None
    dataset_version: Optional[str] = None
    provenance: Optional[Provenance] = None
    qa: Optional[dict[str, Any]] = None
    # Optional descriptive metadata -- absent in the final seven-column CSV.
    cs_style: Optional[CSStyle] = None
    cs_authenticity: Optional[int] = Field(default=None, ge=1, le=3)
    harm_severity: Optional[int] = Field(default=None, ge=1, le=3)
    # Validated code-mixing features (null until a validated tagger exists).
    urdu_word_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cmi: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    # Heuristic diagnostics -- explicitly labelled, never used as a gate.
    urdu_word_ratio_heuristic: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cmi_heuristic: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    feature_method: Optional[str] = None
    feature_validation_status: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _compat(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        meta = d.pop("_meta", None)
        if meta is not None:
            prov = dict(d.get("provenance") or {})
            prov.setdefault("legacy", meta)
            d["provenance"] = prov
        if d.get("harm_category") is None and d.get("domain_id") is not None:
            d["harm_category"] = d["domain_id"]
        if d.get("domain_id") is None and isinstance(d.get("harm_category"), str) \
                and DOMAIN_ID_RE.match(d["harm_category"]):
            d["domain_id"] = d["harm_category"]
        return d

    @field_validator("prompt")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt is whitespace-only")
        return v

    @field_validator("harm_category")
    @classmethod
    def _valid_category_shape(cls, v: str) -> str:
        if not (DOMAIN_ID_RE.match(v) or LEGACY_CATEGORY_RE.match(v)):
            raise ValueError(f"harm_category {v!r} must match D<n> (or legacy H<n>/C<nn>)")
        return v

    @field_validator("domain_id")
    @classmethod
    def _valid_domain_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not DOMAIN_ID_RE.match(v):
            raise ValueError(f"domain_id {v!r} must match D<n>")
        return v

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        if self.condition in _MONO_CONDITIONS and self.cs_style is not None:
            raise ValueError(f"id={self.id} condition={self.condition} must not have cs_style")
        if self.domain_id is not None and self.harm_category != self.domain_id:
            raise ValueError(
                f"id={self.id}: harm_category alias {self.harm_category!r} "
                f"!= domain_id {self.domain_id!r}")

    @property
    def is_legacy(self) -> bool:
        return self.domain_id is None or self.condition in LEGACY_CONDITIONS


class DatasetError(Exception):
    """Raised on schema or structural issues in the dataset."""


def load_dataset(path: str | Path, *, allow_legacy: bool = False) -> list[Prompt]:
    """Load + validate rows. Raises DatasetError on any issue.

    Structural (family-level) validation is separate: call `validate_structure`.
    """
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
    if not allow_legacy:
        legacy = [r.id for r in rows if r.is_legacy]
        if legacy:
            raise DatasetError(
                f"{len(legacy)} legacy rows (SM condition or no domain_id), e.g. "
                f"{legacy[:3]}; pass allow_legacy=True only to reproduce historical runs")
    return rows


def _validate_uniqueness(rows: list[Prompt]) -> None:
    id_counts = Counter(r.id for r in rows)
    dupes = [i for i, c in id_counts.items() if c > 1]
    if dupes:
        raise DatasetError(f"duplicate ids: {dupes[:5]}")


@dataclass
class StructureReport:
    n_families: int
    n_rows: int
    n_complete: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.n_complete == self.n_families


def check_structure(rows: list[Prompt]) -> StructureReport:
    """Exact family contract: every family has EXACTLY one row per active
    condition, no duplicate (family, condition), no legacy conditions, and one
    shared domain / group / dataset version across its four rows."""
    errors: list[str] = []
    by_family: dict[str, list[Prompt]] = defaultdict(list)
    for r in rows:
        by_family[r.base_id].append(r)
    n_complete = 0
    for bid in sorted(by_family):
        fam = by_family[bid]
        counts = Counter(r.condition for r in fam)
        bad = False
        for c in CONDITIONS:
            if counts.get(c, 0) != 1:
                errors.append(f"{bid}: expected exactly 1 {c} row, found {counts.get(c, 0)}")
                bad = True
        extra = sorted(set(counts) - set(CONDITIONS))
        if extra:
            errors.append(f"{bid}: non-active conditions present {extra}")
            bad = True
        for attr in ("domain_id", "domain_name", "harm_category", "group_id", "dataset_version"):
            vals = {getattr(r, attr) for r in fam}
            if len(vals) > 1:
                errors.append(f"{bid}: inconsistent {attr} across conditions {sorted(map(str, vals))}")
                bad = True
        for r in fam:
            if r.id != make_row_id(bid, r.condition):
                errors.append(f"{bid}: row id {r.id!r} != {make_row_id(bid, r.condition)!r}")
                bad = True
        if not bad:
            n_complete += 1
    return StructureReport(len(by_family), len(rows), n_complete, errors)


def validate_structure(rows: list[Prompt]) -> StructureReport:
    rep = check_structure(rows)
    if not rep.ok:
        joined = "\n  ".join(rep.errors[:10])
        more = f"\n  ... +{len(rep.errors) - 10} more" if len(rep.errors) > 10 else ""
        raise DatasetError(f"structural validation failed ({len(rep.errors)} errors):\n  {joined}{more}")
    return rep


def filter_prompts(
    rows: Iterable[Prompt],
    *,
    condition: Optional[str] = None,
    harm_category: Optional[str] = None,
    domain_id: Optional[str] = None,
    base_ids: Optional[set[str]] = None,
    cs_style: Optional[CSStyle] = None,
    min_cs_authenticity: Optional[int] = None,
) -> list[Prompt]:
    out: list[Prompt] = []
    for r in rows:
        if condition is not None and r.condition != condition:
            continue
        if harm_category is not None and r.harm_category != harm_category:
            continue
        if domain_id is not None and r.domain_id != domain_id:
            continue
        if base_ids is not None and r.base_id not in base_ids:
            continue
        if cs_style is not None and r.cs_style != cs_style:
            continue
        if min_cs_authenticity is not None:
            if r.cs_authenticity is None or r.cs_authenticity < min_cs_authenticity:
                continue
        out.append(r)
    return out


def pairing_coverage(rows: list[Prompt]) -> dict[str, dict[str, int]]:
    """{base_id: {CS: n, EN: n, RU: n, UR: n, SM: n}} -- raw counts per condition."""
    cov: dict[str, dict[str, int]] = defaultdict(lambda: {c: 0 for c in ALL_CONDITIONS})
    for r in rows:
        cov[r.base_id][r.condition] += 1
    return dict(cov)


def family_domains(rows: list[Prompt]) -> dict[str, str]:
    """base_id -> domain_id (harm_category alias for legacy rows); asserts consistency."""
    out: dict[str, str] = {}
    for r in rows:
        dom = r.domain_id or r.harm_category
        prev = out.get(r.base_id)
        if prev is not None and prev != dom:
            raise DatasetError(f"family {r.base_id} has inconsistent domain ({prev} vs {dom})")
        out[r.base_id] = dom
    return out


def summarize(rows: list[Prompt]) -> dict:
    by_cond = Counter(r.condition for r in rows)
    by_dom = Counter(r.domain_id or r.harm_category for r in rows)
    cov = pairing_coverage(rows)
    exactly_4 = sum(1 for v in cov.values() if all(v[c] == 1 for c in CONDITIONS)
                    and all(v[c] == 0 for c in LEGACY_CONDITIONS))
    fam_dom = family_domains(rows)
    return {
        "n_rows": len(rows),
        "n_families": len(cov),
        "n_families_exactly_4_conditions": exactly_4,
        "rows_by_condition": dict(sorted(by_cond.items())),
        "rows_by_domain": dict(sorted(by_dom.items())),
        "families_by_domain": dict(sorted(Counter(fam_dom.values()).items())),
        "n_domains": len(set(fam_dom.values())),
    }
