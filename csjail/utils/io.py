"""JSONL read/write helpers with atomic writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no} invalid JSON: {e}") from e
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomic write: dump to tmp file in same dir, then rename.

    WSL2's 9P filesystem (/mnt/c/...) intermittently throws EIO during writes.
    We materialize the lines first, then retry the whole write a few times.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    last_err: Exception | None = None
    for attempt in range(5):
        fd, tmp_path = tempfile.mkstemp(
            prefix=p.name + ".", suffix=".tmp", dir=str(p.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for line in materialized:
                    f.write(line)
            os.replace(tmp_path, p)
            return
        except OSError as e:
            last_err = e
            Path(tmp_path).unlink(missing_ok=True)
            # Errno 5 (EIO) on WSL2 — retry. Other OSErrors: bail.
            if getattr(e, "errno", None) != 5:
                raise
            import time

            time.sleep(0.5 * (2 ** attempt))
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    raise OSError(f"write_jsonl gave up after retries on {p}: {last_err}")


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
