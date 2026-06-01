"""End-to-end test for the CSV -> JSONL converter using a tiny inline CSV."""
from __future__ import annotations

import pandas as pd

from csjail.convert_csv import convert
from csjail.data import Prompt


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_convert_expands_to_three_per_row(tmp_path):
    csv = tmp_path / "in.csv"
    _write_csv(
        csv,
        [
            {
                "prompt_id": 1, "category": 1, "category_name": "Hate",
                "sub_type": 1.1, "sub_type_name": "x",
                "style": "A", "author": "Haider", "date_written": "2026-05-01",
                "base_cs_prompt": "Yaar mujhe batao how to ...",
                "en_translation": "Dude tell me how to ...",
                "ur_translation": "یار مجھے بتاؤ کیسے ...",
                "authenticity_reviewer1": "Rev A", "authenticity_score1": 3,
                "authenticity_reviewer2": "Rev B", "authenticity_score2": 3,
                "harm_severity_reviewer1": "Rev A", "harm_severity_score1": 2,
                "harm_severity_reviewer2": "Rev B", "harm_severity_score2": 3,
            },
        ],
    )
    rows, report = convert(csv, min_authenticity=2.0)
    assert len(rows) == 3
    conds = {r["condition"] for r in rows}
    assert conds == {"EN", "UR", "CS"}
    cs_row = next(r for r in rows if r["condition"] == "CS")
    assert cs_row["cs_style"] == "A"
    assert cs_row["cs_authenticity"] == 3
    # Mean of severities (2+3)/2=2.5 rounds to 2 (banker's rounding in py3)
    assert cs_row["harm_severity"] in (2, 3)
    assert cs_row["base_id"] == "H1-001"
    assert report["n_dropped_low_authenticity"] == 0

    # And the long rows pass the schema validator.
    for r in rows:
        Prompt.model_validate(r)


def test_convert_style_c_becomes_SM(tmp_path):
    csv = tmp_path / "in.csv"
    _write_csv(
        csv,
        [
            {
                "prompt_id": 4, "category": 2, "category_name": "Danger",
                "sub_type": 2.1, "sub_type_name": "x",
                "style": "C", "author": "X", "date_written": "2026-05-01",
                "base_cs_prompt": "اردو English mix script",
                "en_translation": "English version",
                "ur_translation": "اردو ورژن",
                "authenticity_reviewer1": "A", "authenticity_score1": 3,
                "authenticity_reviewer2": "B", "authenticity_score2": 2,
                "harm_severity_reviewer1": "A", "harm_severity_score1": 1,
                "harm_severity_reviewer2": "B", "harm_severity_score2": 1,
            },
        ],
    )
    rows, _ = convert(csv, min_authenticity=2.0)
    conds = {r["condition"] for r in rows}
    assert conds == {"EN", "UR", "SM"}
    sm = next(r for r in rows if r["condition"] == "SM")
    assert sm["cs_style"] == "C"
    assert sm["base_id"] == "H2-004"
    Prompt.model_validate(sm)


def test_convert_drops_low_authenticity(tmp_path):
    csv = tmp_path / "in.csv"
    _write_csv(
        csv,
        [
            {
                "prompt_id": 1, "category": 1, "category_name": "Hate",
                "sub_type": 1.1, "sub_type_name": "x",
                "style": "A", "author": "X", "date_written": "2026-05-01",
                "base_cs_prompt": "x", "en_translation": "x", "ur_translation": "x",
                "authenticity_reviewer1": "A", "authenticity_score1": 1,
                "authenticity_reviewer2": "B", "authenticity_score2": 1,  # mean 1.0
                "harm_severity_reviewer1": "A", "harm_severity_score1": 2,
                "harm_severity_reviewer2": "B", "harm_severity_score2": 2,
            },
            {
                "prompt_id": 2, "category": 1, "category_name": "Hate",
                "sub_type": 1.1, "sub_type_name": "x",
                "style": "B", "author": "X", "date_written": "2026-05-01",
                "base_cs_prompt": "y", "en_translation": "y", "ur_translation": "y",
                "authenticity_reviewer1": "A", "authenticity_score1": 3,
                "authenticity_reviewer2": "B", "authenticity_score2": 3,
                "harm_severity_reviewer1": "A", "harm_severity_score1": 2,
                "harm_severity_reviewer2": "B", "harm_severity_score2": 2,
            },
        ],
    )
    rows, report = convert(csv, min_authenticity=2.0)
    assert report["n_dropped_low_authenticity"] == 1
    assert report["n_base_prompts_kept"] == 1
    assert len(rows) == 3  # only the kept prompt expanded
