"""Parser-only tests for csjail.judge — no network."""
from __future__ import annotations

from csjail.judge import confusion_matrix, parse_judgment, precision_by_condition


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


def test_confusion_matrix_zero_positives_is_undefined_not_zero():
    # No gold-harmful examples at all -> precision AND recall are UNDEFINED
    # (None), not misleadingly 0.0. A gate that treats None as a failure
    # would wrongly punish a condition the judge did nothing wrong on.
    gold = [0, 0, 0]
    pred = [0, 0, 0]
    cm = confusion_matrix(gold, pred)
    bm = cm["binary_harmful_detection"]
    assert bm["tp"] == 0 and bm["fp"] == 0 and bm["fn"] == 0
    assert bm["precision"] is None
    assert bm["recall"] is None


def test_confusion_matrix_false_positive_hurts_precision_only():
    gold = [0, 1]
    pred = [1, 1]  # gold=0 predicted harmful -> FP; gold=1 predicted harmful -> TP
    cm = confusion_matrix(gold, pred)["binary_harmful_detection"]
    assert cm["precision"] == 0.5
    assert cm["recall"] == 1.0


def test_confusion_matrix_false_negative_hurts_recall_only():
    gold = [1, 1]
    pred = [1, 0]  # one harmful mislabeled as refusal -> FN
    cm = confusion_matrix(gold, pred)["binary_harmful_detection"]
    assert cm["precision"] == 1.0
    assert cm["recall"] == 0.5


def test_precision_by_condition_excludes_none_from_min():
    # EN has zero positives (undefined); CS has a real precision problem.
    # min_precision must come from CS, not be dragged to 0 by EN's None.
    gold = [0, 0, 0, 1, 1]
    pred = [0, 0, 0, 1, 0]  # EN: 3x true-negative; CS: one FN
    conditions = ["EN", "EN", "EN", "CS", "CS"]
    out = precision_by_condition(gold, pred, conditions)
    assert out["precision_by_condition"]["EN"] is None
    assert out["worst_condition"] == "CS"
    assert out["min_precision"] == 1.0  # CS's one error is a FN, not FP
    assert out["min_recall"] == 0.5
    assert out["worst_recall_condition"] == "CS"
