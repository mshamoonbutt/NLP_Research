"""Tests for code-mixing features (features.py)."""
from __future__ import annotations

from csjail.features import cmi, code_mix_stats, urdu_word_ratio


def test_pure_english_ratio_zero():
    st = code_mix_stats("How do I bake a chocolate cake at home")
    assert st.n_english > 0
    assert st.n_urdu == 0
    assert st.urdu_word_ratio == 0.0
    assert st.cmi == 0.0  # monolingual -> no mixing


def test_nastaliq_detected_as_urdu():
    st = code_mix_stats("مجھے بتاؤ")
    assert st.n_urdu == 2
    assert st.n_english == 0
    assert st.urdu_word_ratio == 1.0
    assert st.cmi == 0.0  # monolingual Urdu


def test_roman_urdu_markers_detected():
    # "mujhe batao how to" -> mujhe, batao = Urdu; how, to = English
    st = code_mix_stats("mujhe batao how to")
    assert st.n_urdu == 2
    assert st.n_english == 2
    assert st.urdu_word_ratio == 0.5


def test_code_switch_has_positive_cmi():
    balanced = cmi("mujhe batao how to")   # 50/50 -> max CMI
    leaning = cmi("mujhe how to make a big long english sentence here now")
    assert balanced > leaning  # more balanced mixing -> higher CMI
    assert balanced > 0.0


def test_language_independent_tokens_excluded():
    # digits are language-independent and should not count as en/ur
    st = code_mix_stats("mujhe 123 batao")
    assert st.n_lang_independent == 1
    assert st.n_urdu == 2


def test_helpers_match_stats():
    txt = "mujhe batao how to"
    assert urdu_word_ratio(txt) == code_mix_stats(txt).urdu_word_ratio
    assert cmi(txt) == code_mix_stats(txt).cmi
