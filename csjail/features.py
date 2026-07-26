"""Code-mixing features: Urdu-word ratio and Code-Mixing Index (CMI).

These quantify *how much* a prompt mixes Urdu and English, so downstream
analysis (Exp 4) can test whether the code-switching safety effect is really a
"how much Urdu" density effect or a discrete switching effect.

Language tagging per token is heuristic and deliberately simple/transparent:

- Nastaliq / Perso-Arabic script tokens  -> Urdu.
- ASCII-alphabetic tokens that appear in a small Roman-Urdu marker lexicon
  -> Urdu (Roman).
- Other ASCII-alphabetic tokens          -> English.
- Punctuation / digits / symbols         -> language-independent (excluded).

The Roman-Urdu lexicon is intentionally small and swappable; replace
`ROMAN_URDU_MARKERS` with a fuller lexicon for production analysis. The metric
definitions are standard (Gambäck & Das, 2016) so they remain comparable even
if the tagger is upgraded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Perso-Arabic / Nastaliq Unicode blocks used by Urdu.
_URDU_SCRIPT_RE = re.compile(
    r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ASCII_ALPHA_RE = re.compile(r"^[A-Za-z']+$")

# Common Roman-Urdu function/content words. Lowercased. Small on purpose.
ROMAN_URDU_MARKERS: frozenset[str] = frozenset({
    "hai", "hain", "ho", "hu", "hoon", "tha", "thi", "the", "ka", "ki", "ke",
    "ko", "se", "mein", "main", "me", "par", "aur", "ya", "na", "nahi", "nai",
    "ny", "ne", "ni", "mujhe", "mujhy", "mera", "meri", "mere", "tum", "tumhari",
    "tumhara", "aap", "app", "wo", "woh", "yeh", "ye", "is", "us", "kya", "kyun",
    "kyu", "kaise", "kaisay", "kese", "kar", "karo", "karna", "karne", "karke",
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "liye", "liay", "wala",
    "wali", "wale", "bhi", "bhai", "bhaiyo", "log", "logo", "logon", "sab",
    "kuch", "kuchh", "bas", "abhi", "acha", "achha", "theek", "thk", "chao",
    "batao", "bata", "samajh", "samjha", "waqai", "asal", "namak", "haram",
    "zaroor", "zarur", "bilkul", "matlab", "phir", "fir", "jab", "tab", "jo",
    "jis", "kisi", "koi", "sath", "saath", "andar", "bahar", "upar", "neeche",
})


@dataclass
class CodeMixStats:
    n_tokens: int          # all word tokens
    n_urdu: int            # Urdu (script or Roman marker)
    n_english: int         # English
    n_lang_independent: int  # digits / mixed / unclassifiable word tokens
    urdu_word_ratio: float  # n_urdu / (n_urdu + n_english), 0 if none
    cmi: float             # Code-Mixing Index in [0, 100]


def _tag_token(tok: str) -> str:
    """Return 'ur', 'en', or 'ind' (language-independent) for one word token."""
    if _URDU_SCRIPT_RE.search(tok):
        return "ur"
    low = tok.lower()
    if _ASCII_ALPHA_RE.match(tok):
        if low in ROMAN_URDU_MARKERS:
            return "ur"
        return "en"
    # digits, alphanumerics, or anything else with no clear language.
    return "ind"


def code_mix_stats(text: str) -> CodeMixStats:
    """Compute per-utterance code-mixing statistics.

    CMI (Gambäck & Das, 2016) for a single utterance:
        CMI = 100 * (1 - max_lang(w_i) / (n - u))    if (n - u) > 0 else 0
    where n = total tokens, u = language-independent tokens, and max_lang(w_i)
    is the count of tokens in the most frequent language.
    """
    toks = _WORD_RE.findall(text or "")
    n = len(toks)
    n_ur = n_en = n_ind = 0
    for t in toks:
        tag = _tag_token(t)
        if tag == "ur":
            n_ur += 1
        elif tag == "en":
            n_en += 1
        else:
            n_ind += 1

    lang_bearing = n_ur + n_en
    ratio = (n_ur / lang_bearing) if lang_bearing > 0 else 0.0

    denom = n - n_ind
    if denom > 0:
        cmi = 100.0 * (1.0 - (max(n_ur, n_en) / denom))
    else:
        cmi = 0.0

    return CodeMixStats(
        n_tokens=n,
        n_urdu=n_ur,
        n_english=n_en,
        n_lang_independent=n_ind,
        urdu_word_ratio=round(ratio, 4),
        cmi=round(cmi, 2),
    )


def urdu_word_ratio(text: str) -> float:
    return code_mix_stats(text).urdu_word_ratio


def cmi(text: str) -> float:
    return code_mix_stats(text).cmi
