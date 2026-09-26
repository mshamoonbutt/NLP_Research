"""Exp 0 — duplicate grouping, frozen split manifests, and descriptive agreement.

Split contract (see docs/PROTOCOL.md §4):
- Assignment happens at the FAMILY/GROUP level (all four language rows of a
  family, and all families in one duplicate/relative group, always share a
  split), stratified by domain, with every seeded operation applied to sorted
  input so the result is identical across processes.
- `scheme="main"` (primary): `eval_size` held-out families across all six
  domains -> `eval_main`; everything else -> `train_pool` (a CANDIDATE pool,
  not a promise of preference pairs). The main B/C comparison trains on all
  domains.
- `scheme="domain_holdout"` (explicit alternative): the caller NAMES the
  held-out domain(s); they go to `domain_holdout`, `eval_main` is drawn from
  the remaining domains, and the three sets are mutually exclusive. Nothing is
  chosen automatically by size.
- The unseen-domain ablation (Exp 9) is a DERIVED manifest
  (`make_domain_ablation`) that requires fresh training runs.
- Manifests are frozen: `extend_split` assigns only families that are new,
  never moving an existing family.

Near-duplicates use token Jaccard (csjail.qa), not embedding cosine.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from csjail.data import CONDITIONS, Prompt, family_domains
from csjail.features import heuristic_features

SPLIT_METHOD = ("group-level, domain-stratified (largest-remainder quotas), "
                "numpy default_rng(seed).permutation over sorted groups per sorted domain")
TRAIN = "train_pool"
EVAL = "eval_main"
DOMAIN_HOLDOUT = "domain_holdout"
DOMAIN_SUPPLEMENTARY = "domain_supplementary"
QUARANTINE = "excluded_group_relative"
TRAINABLE_SPLITS = frozenset({TRAIN})


# ---------------------------------------------------------------- features --

def attach_heuristic_features(rows: list[Prompt]) -> list[Prompt]:
    """Fill the explicitly-heuristic diagnostic fields. The validated
    `urdu_word_ratio` / `cmi` fields are left untouched (null)."""
    out = []
    for r in rows:
        if r.cmi_heuristic is None or r.urdu_word_ratio_heuristic is None:
            out.append(r.model_copy(update=heuristic_features(r.prompt)))
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------- groups ---

class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.p = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.p[hi] = lo


def load_duplicate_decisions(path: Optional[str | Path]) -> dict[tuple[str, str], str]:
    """Reviewer decisions for near-duplicate candidates: CSV with columns
    family_a, family_b, decision (duplicate|related|distinct). Missing file ->
    no decisions (all candidates stay unresolved)."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("decision") or "").strip().lower()
            if d:
                out[tuple(sorted((r["family_a"], r["family_b"])))] = d
    return out


def build_groups(
    rows: list[Prompt],
    *,
    exact_duplicates: list[dict],
    near_duplicate_candidates: list[dict],
    decisions: Optional[dict[tuple[str, str], str]] = None,
) -> tuple[dict[str, str], dict]:
    """Return (family -> group_id, report).

    Merged into one group: exact duplicates (always), near-duplicate candidates
    that a reviewer marked duplicate/related, and -- conservatively --
    candidates with NO decision yet (so an unresolved candidate can never
    straddle train/eval). Only an explicit `distinct` decision keeps a
    candidate pair apart. Pre-existing `group_id`s on rows are honoured.
    Singletons get group_id == family id.
    """
    decisions = decisions or {}
    fams = sorted({r.base_id for r in rows})
    uf = _UnionFind(fams)
    by_gid: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.group_id:
            by_gid[r.group_id].append(r.base_id)
    for members in by_gid.values():
        for m in members[1:]:
            uf.union(members[0], m)
    for g in exact_duplicates:
        for m in g["families"][1:]:
            uf.union(g["families"][0], m)
    unresolved = merged = distinct = 0
    for c in near_duplicate_candidates:
        key = tuple(sorted((c["family_a"], c["family_b"])))
        dec = decisions.get(key)
        if dec == "distinct":
            distinct += 1
            continue
        if dec is None:
            unresolved += 1
        else:
            merged += 1
        uf.union(*key)
    comps: dict[str, list[str]] = defaultdict(list)
    for fam in fams:
        comps[uf.find(fam)].append(fam)
    fam_to_group = {}
    for root, members in comps.items():
        gid = members[0] if len(members) == 1 else f"grp:{sorted(members)[0]}"
        for m in members:
            fam_to_group[m] = gid
    report = {
        "n_groups": len(comps),
        "n_multi_family_groups": sum(1 for m in comps.values() if len(m) > 1),
        "n_families_in_multi_groups": sum(len(m) for m in comps.values() if len(m) > 1),
        "near_duplicate_candidates": len(near_duplicate_candidates),
        "candidates_unresolved_grouped_conservatively": unresolved,
        "candidates_confirmed_grouped": merged,
        "candidates_marked_distinct": distinct,
        "exact_duplicate_sets": len(exact_duplicates),
    }
    return fam_to_group, report


# ---------------------------------------------------------------- splits ---

def _largest_remainder(total: int, sizes: dict[str, int]) -> dict[str, int]:
    n = sum(sizes.values())
    raw = {k: total * v / n for k, v in sizes.items()}
    base = {k: int(np.floor(x)) for k, x in raw.items()}
    rest = total - sum(base.values())
    order = sorted(sizes, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rest]:
        base[k] += 1
    return base


def _canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def _group_domains(fam_dom: dict[str, str], fam_group: dict[str, str]) -> dict[str, str]:
    by_g: dict[str, Counter] = defaultdict(Counter)
    for fam, g in fam_group.items():
        by_g[g][fam_dom[fam]] += 1
    # A group spanning domains is stratified under its most frequent domain
    # (ties -> smallest id); reported in meta.
    return {g: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for g, c in by_g.items()}


def _select_eval(groups_by_domain: dict[str, list[str]], group_size: dict[str, int],
                 quotas: dict[str, int], rng: np.random.Generator) -> tuple[set[str], dict]:
    chosen: set[str] = set()
    adjustments = {}
    for dom in sorted(groups_by_domain):
        gs = sorted(groups_by_domain[dom])
        perm = [gs[i] for i in rng.permutation(len(gs))]
        need, got = quotas.get(dom, 0), 0
        for g in perm:
            if got >= need:
                break
            if got + group_size[g] <= need:
                chosen.add(g)
                got += group_size[g]
        if got != need:
            adjustments[dom] = {"quota": need, "actual": got}
    return chosen, adjustments


def make_splits(
    rows: list[Prompt],
    *,
    fam_to_group: Optional[dict[str, str]] = None,
    eval_size: int = 200,
    seed: int = 42,
    scheme: str = "main",
    holdout_domains: Optional[list[str]] = None,
    dataset_version: Optional[str] = None,
) -> dict:
    """Build a split manifest (see module docstring). Returns
    {"assignments": {family: {split, group_id, domain_id}}, "meta": {...}}."""
    fam_dom = family_domains(rows)
    fams = sorted(fam_dom)
    fam_to_group = fam_to_group or {f: f for f in fams}
    missing = [f for f in fams if f not in fam_to_group]
    if missing:
        raise ValueError(f"families without a group: {missing[:5]}")
    holdout_domains = sorted(holdout_domains or [])
    if scheme == "main" and holdout_domains:
        raise ValueError("scheme 'main' trains on all domains; use scheme "
                         "'domain_holdout' to withhold domains")
    if scheme == "domain_holdout" and not holdout_domains:
        raise ValueError("scheme 'domain_holdout' requires explicitly named holdout_domains")
    if scheme not in ("main", "domain_holdout"):
        raise ValueError(f"unknown scheme {scheme!r}")
    unknown = sorted(set(holdout_domains) - set(fam_dom.values()))
    if unknown:
        raise ValueError(f"unknown holdout domains {unknown}")

    g_dom = _group_domains(fam_dom, fam_to_group)
    g_size = Counter(fam_to_group[f] for f in fams)
    holdout_groups = {g for g, d in g_dom.items() if d in holdout_domains}
    eligible = {g: d for g, d in g_dom.items() if g not in holdout_groups}
    fam_sizes_by_dom = Counter()
    groups_by_domain: dict[str, list[str]] = defaultdict(list)
    for g, d in eligible.items():
        groups_by_domain[d].append(g)
        fam_sizes_by_dom[d] += g_size[g]
    n_eligible = sum(fam_sizes_by_dom.values())
    if eval_size > n_eligible:
        raise ValueError(f"eval_size {eval_size} > eligible families {n_eligible}")
    quotas = _largest_remainder(eval_size, dict(fam_sizes_by_dom))
    rng = np.random.default_rng(seed)
    eval_groups, adjustments = _select_eval(groups_by_domain, g_size, quotas, rng)

    assignments = {}
    for f in fams:
        g = fam_to_group[f]
        split = DOMAIN_HOLDOUT if g in holdout_groups else (EVAL if g in eval_groups else TRAIN)
        assignments[f] = {"split": split, "group_id": g, "domain_id": fam_dom[f]}
    return _finalize_manifest(assignments, {
        "scheme": scheme,
        "seed": seed,
        "method": SPLIT_METHOD,
        "eval_size_requested": eval_size,
        "holdout_domains": holdout_domains,
        "eval_quota_by_domain": quotas,
        "size_adjustments": adjustments,
        "multi_domain_groups": sorted(
            g for g in set(fam_to_group.values())
            if len({fam_dom[f] for f in fams if fam_to_group[f] == g}) > 1),
        "dataset_version": dataset_version,
    })


def _finalize_manifest(assignments: dict[str, dict], meta: dict) -> dict:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in assignments.values():
        counts[a["split"]][a["domain_id"]] += 1
    meta = dict(meta)
    meta["counts"] = {s: {"total": sum(d.values()), **dict(sorted(d.items()))}
                      for s, d in sorted(counts.items())}
    meta["n_families"] = len(assignments)
    meta["split_id"] = _canonical_hash({"assignments": assignments,
                                        "scheme": meta.get("scheme")})[:16]
    meta.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    return {"assignments": dict(sorted(assignments.items())), "meta": meta}


def extend_split(frozen: dict, rows: list[Prompt], *, fam_to_group: dict[str, str],
                 dataset_version: Optional[str] = None) -> dict:
    """Append-only extension of a frozen manifest. Existing families keep their
    assignment verbatim. New families go to `train_pool`, unless their group
    already contains an eval/holdout family, in which case they are
    quarantined (neither trained nor evaluated) so they cannot leak."""
    old = frozen["assignments"]
    fam_dom = family_domains(rows)
    gone = sorted(set(old) - set(fam_dom))
    if gone:
        raise ValueError(f"frozen families missing from new data: {gone[:5]}")
    group_split = defaultdict(set)
    for f, a in old.items():
        group_split[fam_to_group.get(f, a["group_id"])].add(a["split"])
    assignments = {f: dict(a) for f, a in old.items()}
    added = []
    for f in sorted(set(fam_dom) - set(old)):
        g = fam_to_group[f]
        splits = group_split.get(g, set())
        split = QUARANTINE if splits - {TRAIN} else TRAIN
        assignments[f] = {"split": split, "group_id": g, "domain_id": fam_dom[f],
                          "added_after_freeze": True}
        added.append(f)
    meta = dict(frozen["meta"])
    meta.pop("created_utc", None)
    meta.update({"extended_from_split_id": frozen["meta"]["split_id"],
                 "n_families_added": len(added), "dataset_version": dataset_version})
    return _finalize_manifest(assignments, meta)


def make_domain_ablation(main: dict, domain: str, *, attested_before_outcomes: bool) -> dict:
    """Exp 9 unseen-domain ablation manifest derived from the main manifest.

    `domain`'s training-pool families are withheld from ALL training
    supervision (-> `domain_supplementary`, a separately identified untouched
    set); `eval_main` is unchanged so the domain's frozen held-out subset is
    evaluated. Requires the caller to attest the domain was chosen before
    seeing its outcomes. Needs FRESH adapters trained from the base model:
    an adapter trained under the main manifest has seen this domain."""
    if not attested_before_outcomes:
        raise ValueError("the ablation domain must be chosen before seeing its outcomes; "
                         "pass attested_before_outcomes=True only if that is true")
    if main["meta"].get("scheme") != "main":
        raise ValueError("derive the ablation from the main manifest")
    doms = {a["domain_id"] for a in main["assignments"].values()}
    if domain not in doms:
        raise ValueError(f"unknown domain {domain!r}")
    assignments = {}
    for f, a in main["assignments"].items():
        b = dict(a)
        if a["split"] == TRAIN and a["domain_id"] == domain:
            b["split"] = DOMAIN_SUPPLEMENTARY
        assignments[f] = b
    meta = {k: v for k, v in main["meta"].items() if k not in ("created_utc", "split_id", "counts")}
    meta.update({"scheme": "unseen_domain_ablation", "ablation_domain": domain,
                 "parent_split_id": main["meta"]["split_id"],
                 "attested_domain_chosen_before_outcomes": True,
                 "requires_fresh_adapters": True})
    return _finalize_manifest(assignments, meta)


def verify_split(manifest: dict, rows: list[Prompt]) -> list[str]:
    """Return a list of violations (empty = OK)."""
    errs = []
    a = manifest["assignments"]
    fam_dom = family_domains(rows)
    for f in sorted(set(fam_dom) - set(a)):
        errs.append(f"family {f} not assigned")
    by_g: dict[str, set[str]] = defaultdict(set)
    for f, x in a.items():
        by_g[x["group_id"]].add(x["split"])
        if f in fam_dom and fam_dom[f] != x["domain_id"]:
            errs.append(f"family {f}: domain {fam_dom[f]} != manifest {x['domain_id']}")
    for g, splits in sorted(by_g.items()):
        if TRAIN in splits and splits - {TRAIN, DOMAIN_SUPPLEMENTARY}:
            errs.append(f"group {g} straddles train_pool and {sorted(splits - {TRAIN})}")
    return errs


def trainable_families(manifest: dict) -> set[str]:
    return {f for f, a in manifest["assignments"].items() if a["split"] in TRAINABLE_SPLITS}


def eval_families(manifest: dict, *, domain: Optional[str] = None) -> set[str]:
    return {f for f, a in manifest["assignments"].items()
            if a["split"] == EVAL and (domain is None or a["domain_id"] == domain)}


def assert_trainable(manifest: dict, families: Iterable[str]) -> None:
    """Training-boundary leakage check: every family must be in the training
    pool, and no family may share a group with any non-training family."""
    a = manifest["assignments"]
    fams = set(families)
    unknown = sorted(f for f in fams if f not in a)
    if unknown:
        raise ValueError(f"families not in split manifest: {unknown[:5]}")
    bad = sorted(f for f in fams if a[f]["split"] not in TRAINABLE_SPLITS)
    if bad:
        raise ValueError(f"{len(bad)} families are not in the training pool "
                         f"(e.g. {bad[:5]}) -- eval/holdout leakage")
    nontrain_groups = {x["group_id"] for x in a.values() if x["split"] not in TRAINABLE_SPLITS}
    rel = sorted(f for f in fams if a[f]["group_id"] in nontrain_groups)
    if rel:
        raise ValueError(f"{len(rel)} families share a group with eval/holdout families "
                         f"(e.g. {rel[:5]})")


def load_manifest(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if "assignments" not in m or "split_id" not in m.get("meta", {}):
        raise ValueError(f"{path} is not a split manifest")
    return m


# ------------------------------------------------ stratified index sampler --

def proportional_stratified_indices(keys: list[str], n: int, *, seed: int = 42) -> list[int]:
    """Indices of a proportional-by-key stratified sample of n items,
    deterministic given `seed` and the order of `keys` (callers pass sorted
    items). Used by the Exp 1 gold sampler, Exp 4b and the robustness subset."""
    n_total = len(keys)
    if n_total == 0 or n <= 0:
        return []
    n = min(n, n_total)
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    quotas = _largest_remainder(n, {k: len(v) for k, v in by_key.items()})
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for k in sorted(by_key):
        idx = by_key[k]
        perm = rng.permutation(len(idx))
        picked.extend(idx[i] for i in perm[: quotas[k]])
    return sorted(picked)


# ------------------------------------------------ descriptive agreement ----

def cohens_kappa(rater_a: list, rater_b: list) -> Optional[float]:
    """Unweighted Cohen's kappa. None (NA) when undefined -- e.g. both raters
    used a single identical category, so expected agreement is 1."""
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    if n == 0:
        return None
    cats = sorted(set(rater_a) | set(rater_b), key=str)
    po = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    ca, cb = Counter(rater_a), Counter(rater_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def pairwise_agreement(csv_path: str | Path) -> dict:
    """Descriptive agreement for every `rater1_<field>` / `rater2_<field>`
    column pair in an independent-annotation file (legacy `<field>_score1` /
    `<field>_score2` pairs are also recognised). Rows where either rater is
    blank are excluded and counted. Pre-adjudication labels only; NOT a gate.
    """
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = set(rows[0].keys()) if rows else set()
    pairs = []
    for c in sorted(cols):
        if c.startswith("rater1_") and "rater2_" + c[7:] in cols:
            pairs.append((c[7:], c, "rater2_" + c[7:]))
        elif c.endswith("_score1") and c[:-1] + "2" in cols:
            pairs.append((c[:-7], c, c[:-1] + "2"))
    out = {}
    for name, c1, c2 in pairs:
        both = [(r[c1].strip(), r[c2].strip()) for r in rows
                if (r.get(c1) or "").strip() and (r.get(c2) or "").strip()]
        a, b = [x for x, _ in both], [y for _, y in both]
        k = cohens_kappa(a, b)
        out[name] = {
            "n": len(both), "n_missing": len(rows) - len(both),
            "raw_agreement": (sum(x == y for x, y in both) / len(both)) if both else None,
            "cohens_kappa_unweighted": None if k is None else round(k, 4),
        }
    return {"source": str(csv_path), "n_rows": len(rows), "fields": out,
            "note": "descriptive only; not a dataset validity gate"}


# ------------------------------------------------ descriptive stats --------

def dataset_stats(rows: list[Prompt]) -> dict:
    def _summ(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        a = np.array(xs, dtype=float)
        return {"n": int(a.size), "mean": round(float(a.mean()), 3),
                "median": round(float(np.median(a)), 3),
                "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}

    per_cond = {}
    for c in CONDITIONS:
        rs = [r for r in rows if r.condition == c]
        per_cond[c] = {
            "n": len(rs),
            "word_len": _summ([float(len(r.prompt.split())) for r in rs]),
            "char_len": _summ([float(len(r.prompt)) for r in rs]),
            "cmi_heuristic": _summ([r.cmi_heuristic for r in rs if r.cmi_heuristic is not None]),
            "urdu_word_ratio_heuristic": _summ([r.urdu_word_ratio_heuristic for r in rs
                                                if r.urdu_word_ratio_heuristic is not None]),
        }
    return {
        "per_condition": per_cond,
        "families_by_domain": dict(sorted(Counter(family_domains(rows).values()).items())),
        "heuristic_feature_note": "cmi/urdu_word_ratio here are *_heuristic and unvalidated; "
                                  "do not use for mechanistic inference",
    }


@dataclass
class _Unused:  # keeps dataclass import meaningful for type checkers
    pass
