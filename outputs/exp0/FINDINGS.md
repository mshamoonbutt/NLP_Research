# Exp0 findings — extended dataset (`annotation_template.xlsx` → `data/csjail_v1.jsonl`)

Run: `convert_v1` + `exp0_finalize_data.py --eval-holdout 300 --held-out-categories 2`.
This covers everything computable on CPU. ASR/judge/DPO numbers require the 4080.

## Dataset is solid
- **4000 long rows** = 1000 base prompts × {CS, EN, RU, UR}. No empty condition texts.
- **Condition parity 1.000** — every base_id has all four conditions (paired stats OK).
- **Balanced**: exactly 100 base prompts per category, 10 categories (C01–C10, 400 rows each).
- **Split**: 700 train / 300 eval; held-out categories **C01, C02** (zero-shot-category eval).
- **Severity**: present on 708/4000 rows (177 base prompts); kept as-is, never imputed.

## Feature summary (per condition)
| cond | n | CMI mean | CMI median | urdu_word_ratio |
|------|----|----------|-----------|-----------------|
| EN   | 1000 | 6.27  | 5.88  | 0.063 |
| CS   | 1000 | 28.78 | 29.41 | 0.288 |
| RU   | 1000 | 36.47 | 36.84 | 0.372 |
| UR   | 1000 | 0.21  | 0.00  | 0.998 |

EN (English) and UR (Nastaliq) land exactly where expected. See caveat 3 for CS/RU.

## Three caveats to weigh (none block the run)
1. **CS == RU on 275/1000 base prompts** (byte-identical). These pairs always agree
   in the CS-vs-RU McNemar (contribute to neither discordant cell) → they *dilute*
   the core code-switching contrast (lower power) but do **not** bias it. The 725
   distinct pairs still give strong paired signal. CS-CMI is ~equal whether CS==RU
   (30.4) or not (28.2), so these aren't degenerate low-mix prompts — they're cases
   where the author's Roman-Urdu rendering equals the code-switched base.
2. **8 base_ids are exact duplicate texts** (4 distinct prompts, all in C06,
   prompt_ids 569–572 repeated at 578–581). Pseudo-replication; trivial (0.8%).
   Drop one of each pair if you want a clean 996.
3. **Roman-Urdu CMI tagger under-counts Urdu content words.** RU's urdu_word_ratio
   is 0.37 (romanized Urdu should be ~0.9), which inflates RU's CMI *above* CS's —
   backwards. Cause: `features.ROMAN_URDU_MARKERS` is a ~110-word lexicon (documented
   as "small on purpose"); most Roman-Urdu content words fall through to "English".
   **Impact:** CMI/urdu_word_ratio are reliable for EN vs UR and as a within-condition
   relative measure, but currently **cannot separate CS from RU**. The primary ASR
   analysis does not use CMI, so this is not a blocker — but Exp4 (code-switching
   *mechanism*, CMI as moderator) needs a stronger Roman-Urdu language-ID first.

## Next (on the 4080)
Exp1 judge gate → Exp2 sweep (3 SLMs × 4 conditions) → Exp3 isolation McNemar
(CS-RU, RU-UR, RU-EN, CS-EN) → Exp6 pairs → Exp7 DPO arms → Exp8 post-eval gate.
See `RUNBOOK.md`.
