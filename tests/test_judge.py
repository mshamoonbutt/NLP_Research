"""Parser-only tests for csjail.judge — no network."""
from __future__ import annotations

from csjail.judge import confusion_matrix, parse_judgment


def test_parse_clean_json():
    j = parse_judgment('{"label": 1, "rationale": "produced harmful content"}')
    assert j.parse_ok and j.label == 1 and "harmful" in j.rationale


def test_parse_with_markdown_fence():
    raw = "Sure, here is the judgment:\n```json\n{\"label\": 0, \"rationale\": \"refusal\"}\n```"
    j = parse_judgment(raw)
    assert j.parse_ok and j.label == 0


def test_parse_label_only_regex_fallback():
    raw = 'My answer "label": 2 because partial.'
    j = parse_judgment(raw)
    assert j.parse_ok and j.label == 2


def test_parse_invalid_label_rejected():
    j = parse_judgment('{"label": 5}')
    assert not j.parse_ok and j.label is None


def test_parse_empty():
    j = parse_judgment("")
    assert not j.parse_ok and j.label is None


def test_confusion_matrix_perfect():
    gold = [0, 0, 1, 1, 2]
    pred = [0, 0, 1, 1, 2]
    cm = confusion_matrix(gold, pred)
    # Diagonal only
    assert cm["matrix"][0][0] == 2
    assert cm["matrix"][1][1] == 2
    assert cm["matrix"][2][2] == 1
    # Binary detection: label 1/2 = positive
    bm = cm["binary_harmful_detection"]
    assert bm["tp"] == 3 and bm["fp"] == 0 and bm["fn"] == 0
    assert bm["precision"] == 1.0 and bm["recall"] == 1.0


def test_confusion_matrix_parse_failures_counted():
    cm = confusion_matrix([0, 1], [None, 1])
    assert cm["parse_failures"] == 1
