"""Exp 6 — `chosen` refusal generation via a model DISTINCT from the judge.

The `chosen` text is the DPO training target: a natural code-switched
refusal that declines *and* briefly redirects. If the same model both writes
`chosen` and judges whether a trained model's outputs look like a refusal,
the pipeline is circular — the trained model is rewarded for producing
judge-shaped refusals, evaluated by that same judge. `assert_distinct_from_judge`
below is the guard against exactly that (see the experiment guide, Exp 6
recommendation #2: "the fix costs nothing").

Few-shot prompted with hand-written exemplars (data/refusal_exemplars.jsonl).
Provider is pluggable; Anthropic is the configured default in
configs/dpo.yaml `prefdata.chosen_generator` specifically because it differs
from the OpenAI judge.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from csjail.utils.io import read_jsonl

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dpo.yaml"

SYSTEM_PROMPT = (
    "You write short, natural refusals in code-switched Urdu-English "
    "(Roman script, casual register) for a safety-training dataset. Given a "
    "harmful request, write a refusal that: (1) clearly declines, with no "
    "hedging or partial compliance, (2) briefly explains why in one clause, "
    "and (3) redirects to a safe alternative in one clause. 2-4 sentences. "
    "Match the register and code-switching style of the examples exactly. "
    "Output ONLY the refusal text, nothing else."
)


@dataclass
class GeneratorConfig:
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 300


def load_generator_config(path: str | Path = CONFIG_PATH) -> GeneratorConfig:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    g = cfg["prefdata"]["chosen_generator"]
    return GeneratorConfig(provider=g["provider"], model=g["model"])


def assert_distinct_from_judge(gen_cfg: GeneratorConfig, judge_cfg) -> None:
    """Raise if the chosen-generator and the judge would be the same model.

    This is not a style preference — a reviewer will spot GPT-4o writing the
    refusals GPT-4o then grades. Fail loud, at config-load time, not after
    burning API budget.
    """
    if gen_cfg.provider == judge_cfg.provider and gen_cfg.model == judge_cfg.model:
        raise ValueError(
            f"chosen-generator ({gen_cfg.provider}/{gen_cfg.model}) must be "
            f"DISTINCT from the judge ({judge_cfg.provider}/{judge_cfg.model}) "
            "-- the judge would be grading its own refusal-writing style. "
            "Set configs/dpo.yaml prefdata.chosen_generator.{provider,model} "
            "to a different model."
        )


def load_exemplars(path: str = "data/refusal_exemplars.jsonl") -> list[dict]:
    return read_jsonl(path)


def _build_user_prompt(prompt: str, exemplars: list[dict]) -> str:
    lines = []
    for ex in exemplars:
        lines.append(f"REQUEST: {ex['prompt']}\nREFUSAL: {ex['chosen']}\n")
    lines.append(f"REQUEST: {prompt}\nREFUSAL:")
    return "\n".join(lines)


class ChosenGenerator:
    """Async few-shot refusal generator. Anthropic is the only provider
    wired up (the point is only that it differs from the OpenAI judge);
    add another provider branch here if you swap the judge."""

    def __init__(self, cfg: Optional[GeneratorConfig] = None):
        self.cfg = cfg or load_generator_config()
        if self.cfg.provider != "anthropic":
            raise NotImplementedError(
                f"provider={self.cfg.provider} not implemented in this slice"
            )
        try:
            from anthropic import AsyncAnthropic  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. pip install anthropic"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY env var is required")
        self._client = AsyncAnthropic(api_key=api_key)

    async def _one(self, prompt: str, exemplars: list[dict]) -> str:
        resp = await self._client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                      "content": _build_user_prompt(prompt, exemplars)}],
        )
        return "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()

    async def generate_many(
        self, prompts: list[str], exemplars: list[dict], *, concurrency: int = 8,
    ) -> list[str]:
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(p: str) -> str:
            async with sem:
                return await self._one(p, exemplars)

        return await asyncio.gather(*(_bounded(p) for p in prompts))

    def generate_sync(self, prompts: list[str], exemplars: list[dict]) -> list[str]:
        return asyncio.run(self.generate_many(prompts, exemplars))


def is_clean_refusal(judge_label: Optional[int]) -> bool:
    """The 'clean refusal' criterion (guide Exp 6 recommendation #8).

    A hedged/partial refusal (judge label 2) is exactly the "boundary
    refusal" pattern that arXiv 2602.11157 found increases jailbreak success
    by up to 16.6pp when distilled into a student model. Reuse the existing
    3-class judge rubric as the clean-refusal check: only label 0 (REFUSAL)
    passes; label 2 (PARTIAL/hedged) or 1 (FULL_COMPLY, i.e. the generator
    failed to refuse) are rejected.
    """
    return judge_label == 0
