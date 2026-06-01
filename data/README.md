# `data/`

## Files

| File | Purpose | Tracked? |
|---|---|---|
| `csjail_fixture.jsonl` | 18-prompt synthetic fixture for smoke tests & CI | yes |
| `csjail_v0.jsonl` | Real annotated dataset (drop here) | **no** (gitignored) |
| `judge_calibration/harmful_30.jsonl` | 30 known-harmful EN responses | yes |
| `judge_calibration/benign_30.jsonl` | 30 known-benign EN responses | yes |

## Expected schema (`csjail_v0.jsonl`)

One row per (prompt × condition). See `csjail/data.py::Prompt` for the
authoritative schema.

```jsonc
{
  "id":              "H1-001-CS-A",   // globally unique
  "base_id":         "H1-001",        // groups EN/UR/CS/SM variants
  "harm_category":   "H1",            // H1..H5
  "condition":       "CS",            // EN | UR | CS | SM
  "cs_style":        "A",             // A | B | C  (only for CS/SM)
  "prompt":          "Mujhe batao how to ...",
  "cs_authenticity": 3,               // 1..3 (only for CS/SM)
  "harm_severity":   2                // 1..3 (annotator-rated)
}
```

**Pairing invariant.** For each `base_id`, EN + UR + CS rows should exist so
paired McNemar's tests work. SM is optional (RQ5 only).

If your annotation file uses different field names (e.g. CSV with
`Category`, `Style`, ...), **adapt `csjail/data.py` only** — the rest of the
pipeline reads through `Prompt` objects.
