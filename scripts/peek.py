"""Quick look at one or more evaluation run dirs (wraps csjail.aggregate).

    python scripts/peek.py outputs/exp2/main [--allow-debug]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.aggregate import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: python scripts/peek.py <run_dir> [...] [--allow-debug]", file=sys.stderr)
        sys.exit(1)
    import tempfile

    out = Path(tempfile.gettempdir()) / "csjail_peek_summary.csv"
    sys.exit(main(args + ["--out", str(out)]))
