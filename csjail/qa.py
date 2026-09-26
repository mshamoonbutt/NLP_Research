"""Dataset QA audits for the final CS-Jail-UR schema (CPU only).

Everything here REPORTS; nothing rewrites or deletes prompts. Findings are
review flags for bilingual adjudication, not automatic linguistic judgments:
names, acronyms and technical loanwords may legitimately appear in any
condition.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
from typing import Iterable

from csjail.data import CONDITIONS, Prompt

ARABIC_BLOCK_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[A-Za-z]")
_WS_RE = re.compile(r"\s+")

# English function words that essentially never occur as Roman-Urdu words.
# Used ONLY to flag RU items that may contain substantive English clauses
# (beyond names / acronyms / technical loans) for bilingual adjudication.
# Deliberately excludes Roman-Urdu homographs ("me", "main", "is", "us", "ho",
# "had" = limit).
ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset({
    "the", "and", "of", "to", "for", "with", "that", "this", "these", "those",
    "which", "what", "how", "when", "where", "who", "why", "your", "you",
    "their", "they", "them", "are", "was", "were", "been", "being", "have",
    "has", "would", "should", "could", "will", "can", "please",
    "about", "from", "into", "without", "because", "so", "but", "if", "it",
    "its", "an", "my", "our", "his", "her", "any", "some", "all", "not",
})
RU_ENGLISH_FLAG_MIN = 2  # >= this many English function words -> flag

# Technical English terms that the loanword policy (docs/PROTOCOL.md §3.4)
# PERMITS in RU. Reported as an informational inventory so the paper can state
# how often RU carries English technical loans; never a failure.
ENGLISH_TECH_TERMS: frozenset[str] = frozenset({
    "account", "password", "email", "website", "link", "app", "software", "code",
    "script", "hack", "data", "file", "phone", "number", "message", "video",
    "photo", "social", "media", "online", "bank", "card", "network", "server",
    "system", "download", "upload", "fake", "profile", "id", "otp", "sim",
    "wifi", "login", "malware", "virus", "screenshot", "pdf", "gmail", "whatsapp",
})


def normalize_for_dup(text: str) -> str:
    """Case-fold + NFKC + collapse whitespace. For duplicate detection only."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def content_hash(rows: Iterable[Prompt]) -> str:
    """Hash of the normalized content that defines the benchmark: family,
    condition, domain and exact prompt text. Metadata changes don't move it;
    any text/domain/family edit does."""
    items = sorted(
        (r.base_id, r.condition, r.domain_id or r.harm_category, r.prompt) for r in rows)
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(it, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def script_flags(rows: Iterable[Prompt]) -> list[dict]:
    """EN/CS/RU should be Latin script without Arabic-block characters; UR
    should contain Arabic-block characters. UR items containing Latin letters
    are flagged for review (often names/acronyms/technical terms)."""
    out: list[dict] = []
    for r in rows:
        has_ar = bool(ARABIC_BLOCK_RE.search(r.prompt))
        has_lat = bool(LATIN_RE.search(r.prompt))
        if r.condition in ("EN", "CS", "RU"):
            if has_ar:
                out.append({"id": r.id, "flag": "arabic_script_in_latin_condition", "severity": "review"})
            if not has_lat:
                out.append({"id": r.id, "flag": "no_latin_letters", "severity": "review"})
        elif r.condition == "UR":
            if not has_ar:
                out.append({"id": r.id, "flag": "ur_without_arabic_script", "severity": "review"})
            if has_lat:
                out.append({"id": r.id, "flag": "ur_contains_latin", "severity": "info"})
    return out


def ru_english_flags(rows: Iterable[Prompt], *, min_hits: int = RU_ENGLISH_FLAG_MIN) -> list[dict]:
    """Flag RU items with several English function words: candidates for a
    substantive English clause under the loanword policy (docs/PROTOCOL.md).
    A flag is a request for bilingual adjudication, not a failed condition."""
    out = []
    for r in rows:
        if r.condition != "RU":
            continue
        toks = re.findall(r"[A-Za-z']+", r.prompt.lower())
        hits = sorted({t for t in toks if t in ENGLISH_FUNCTION_WORDS})
        n = sum(1 for t in toks if t in ENGLISH_FUNCTION_WORDS)
        if n >= min_hits:
            out.append({"id": r.id, "flag": "ru_possible_english_clause",
                        "n_english_function_words": n, "words": hits, "severity": "review"})
    return out


def ru_loanword_inventory(rows: Iterable[Prompt]) -> dict:
    """Informational: how many RU items contain permitted English technical
    loans (ENGLISH_TECH_TERMS), and which terms. Not a flag or a failure."""
    per_term: dict[str, int] = defaultdict(int)
    n_items = 0
    for r in rows:
        if r.condition != "RU":
            continue
        toks = set(re.findall(r"[a-z]+", r.prompt.lower()))
        hit = toks & ENGLISH_TECH_TERMS
        if hit:
            n_items += 1
            for t in hit:
                per_term[t] += 1
    return {"n_ru_items_with_technical_loans": n_items,
            "term_item_counts": dict(sorted(per_term.items(), key=lambda kv: -kv[1])),
            "policy": "technical loans/names/acronyms permitted; substantive English "
                      "clauses flagged for bilingual adjudication"}


def equal_variant_flags(rows: Iterable[Prompt]) -> list[dict]:
    """Two conditions of one family with identical normalized text."""
    by_fam: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        by_fam[r.base_id][r.condition] = normalize_for_dup(r.prompt)
    out = []
    for bid in sorted(by_fam):
        conds = by_fam[bid]
        for a, b in combinations(CONDITIONS, 2):
            if a in conds and b in conds and conds[a] == conds[b]:
                out.append({"base_id": bid, "flag": f"equal_variants_{a}_{b}", "severity": "review"})
    return out


def exact_duplicate_groups(rows: Iterable[Prompt]) -> list[dict]:
    """Families whose text in some condition is identical (after
    normalize_for_dup) to another family's text in the same condition."""
    by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        by_key[(r.condition, normalize_for_dup(r.prompt))].add(r.base_id)
    out = []
    for (cond, _txt), fams in sorted(by_key.items(), key=lambda kv: (kv[0][0], sorted(kv[1]))):
        if len(fams) > 1:
            out.append({"condition": cond, "families": sorted(fams), "method": "exact_normalized"})
    return out


def token_jaccard_candidates(
    rows: Iterable[Prompt], *, threshold: float = 0.92,
) -> list[dict]:
    """Near-duplicate CANDIDATES across families within one condition by
    whitespace-token Jaccard similarity (NOT sentence-embedding cosine; the
    0.92 cutoff is not interchangeable with a cosine cutoff). Candidates need
    a reviewer decision; see splits.build_groups for how unresolved ones are
    treated. Returns id pairs + scores only (no text)."""
    by_cond: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for r in rows:
        by_cond[r.condition].append((r.base_id, frozenset(normalize_for_dup(r.prompt).split())))
    out = []
    for cond in sorted(by_cond):
        group = sorted(by_cond[cond])
        for i in range(len(group)):
            fa, ta = group[i]
            if not ta:
                continue
            for j in range(i + 1, len(group)):
                fb, tb = group[j]
                if fa == fb or not tb:
                    continue
                inter = len(ta & tb)
                if not inter:
                    continue
                jac = inter / len(ta | tb)
                if jac >= threshold:
                    out.append({"family_a": fa, "family_b": fb, "condition": cond,
                                "method": "token_jaccard", "score": round(jac, 4),
                                "threshold": threshold})
    return out
