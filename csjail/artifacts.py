"""Locate and verify frozen pipeline artifacts.

Exp 0 finalizes into `outputs/exp0/<dataset_version>/` and, only on success,
writes `outputs/exp0/LATEST.json`. Every downstream stage resolves its inputs
through `resolve_exp0`, which refuses unfinalized directories and checks that
the dataset file and split manifest are the ones the manifest recorded.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from csjail.data import Prompt, load_dataset, validate_structure
from csjail.splits import load_manifest, verify_split
from csjail.utils.io import sha256_file

ROOT = Path(__file__).resolve().parent.parent
EXP0_ROOT = ROOT / "outputs" / "exp0"
FINALIZED_MARKER = "FINALIZED"


class ArtifactError(Exception):
    pass


@dataclass
class Exp0Artifacts:
    dir: Path
    dataset_path: Path
    split_path: Path
    manifest: dict
    split: dict

    @property
    def dataset_version(self) -> str:
        return self.manifest["dataset_version"]

    @property
    def split_id(self) -> str:
        return self.split["meta"]["split_id"]

    def load_rows(self) -> list[Prompt]:
        rows = load_dataset(self.dataset_path)
        validate_structure(rows)
        errs = verify_split(self.split, rows)
        if errs:
            raise ArtifactError(f"split manifest does not match dataset: {errs[:5]}")
        return rows


def resolve_exp0(exp0_dir: Optional[str | Path] = None, *,
                 split_path: Optional[str | Path] = None) -> Exp0Artifacts:
    """Return verified Exp 0 artifacts. `split_path` may point at an alternative
    manifest (e.g. an Exp 9 ablation) derived from the same dataset."""
    if exp0_dir is None:
        latest = EXP0_ROOT / "LATEST.json"
        if not latest.exists():
            raise ArtifactError("no finalized Exp 0 artifact (outputs/exp0/LATEST.json "
                                "missing) -- run scripts/exp0_finalize_data.py first")
        exp0_dir = ROOT / json.loads(latest.read_text(encoding="utf-8"))["dir"]
    d = Path(exp0_dir)
    if not (d / FINALIZED_MARKER).exists():
        raise ArtifactError(f"{d} is not a finalized Exp 0 directory (no {FINALIZED_MARKER})")
    manifest = json.loads((d / "dataset_manifest.json").read_text(encoding="utf-8"))
    ds = d / manifest["dataset_file"]
    if not ds.exists():
        raise ArtifactError(f"dataset file {ds} missing (it is gitignored: re-run Exp 0 "
                            "locally from the source CSV)")
    got = sha256_file(ds)
    if got != manifest["dataset_file_sha256"]:
        raise ArtifactError(f"{ds} sha256 {got[:12]} != manifest "
                            f"{manifest['dataset_file_sha256'][:12]}: dataset changed after finalization")
    sp = Path(split_path) if split_path else d / "split_manifest.json"
    split = load_manifest(sp)
    if split["meta"].get("dataset_version") != manifest["dataset_version"]:
        raise ArtifactError(f"split {sp} is for dataset {split['meta'].get('dataset_version')}, "
                            f"not {manifest['dataset_version']}")
    if not split_path and split["meta"]["split_id"] != manifest["split_id"]:
        raise ArtifactError("split_manifest.json does not match the finalized split_id")
    return Exp0Artifacts(d, ds, sp, manifest, split)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def file_sha256_or_none(path: Optional[str | Path]) -> Optional[str]:
    return sha256_file(path) if path and Path(path).exists() else None
