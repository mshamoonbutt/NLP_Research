"""Shared fixtures. All texts are harmless artificial placeholders."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from csjail.convert_final import convert
from csjail.data import Prompt, make_row_id

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "final_fixture.csv"
REAL_CSV = ROOT / "data" / "CS-Jail-UR_final_692.csv"
COLUMNS = ["prompt_id", "domain_id", "domain_name", "EN", "CS", "RU", "UR"]
DOMAINS = {"D1": "Hate, harassment and coercion", "D2": "Self-harm and suicide",
           "D3": "Dangerous acts, violence and criminal facilitation",
           "D4": "Deception, fraud and intellectual-property abuse",
           "D5": "Privacy and personal-information abuse",
           "D6": "Cyber abuse and unauthorized access"}


@pytest.fixture
def fixture_rows() -> list[Prompt]:
    rows, _ = convert(FIXTURE_CSV)
    return [Prompt.model_validate(r) for r in rows]


def read_fixture_csv() -> list[dict]:
    with FIXTURE_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def family(pid: str, dom: str = "D1", tag: str = "x") -> list[Prompt]:
    """A synthetic four-condition family (harmless text)."""
    texts = {"EN": f"explain {tag} {pid}", "CS": f"yaar {tag} explain karo {pid}",
             "RU": f"{tag} samjhao {pid}", "UR": f"سمجھائیں {pid}"}
    return [Prompt(id=make_row_id(pid, c), base_id=pid, condition=c, prompt=t,
                   harm_category=dom, domain_id=dom, domain_name=DOMAINS[dom],
                   dataset_version="test-v")
            for c, t in texts.items()]


def result(model: str, fam: str, cond: str, unsafe, *, dom: str = "D1", arm: str = "A",
           status: str = "ok", degree: str | None = None, kind: str | None = None,
           sample_index: int = 0, n: int = 1, response: str = "resp", **extra) -> dict:
    """A synthetic result record in the run_eval format."""
    if unsafe is None and status == "ok":
        status = "parse_error"
    return {
        "kind": "result", "model": model, "arm": arm, "base_id": fam,
        "row_id": make_row_id(fam, cond), "condition": cond, "domain_id": dom,
        "dataset_version": "test-v", "split_id": "split-1", "sample_index": sample_index,
        "sampling": {"temperature": 0.0 if n == 1 else 0.7, "n": n}, "generation_status": "ok",
        "gen_key": f"{model}/{arm}/{fam}/{cond}#{sample_index}", "response": response,
        "response_sha256": "h", "judge_status": status, "judge_unsafe": unsafe,
        "judge_compliance_degree": degree or ("full" if unsafe else "none"),
        "judge_response_kind": kind or ("harmful_compliance" if unsafe else "refusal"),
        "judge_fingerprint_id": "fp-1", "_source": "test", **extra,
    }
