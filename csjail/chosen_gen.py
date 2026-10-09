"""Exp 6 — `chosen` refusal generation via a model DISTINCT from the judge.

`chosen` is the DPO target: a natural refusal that declines and briefly
redirects -- code-switched Roman Urdu-English for CS pairs, English for the
matched English control. Keeping the generator distinct from the judge
(`assert_distinct_from_judge`) reduces judge self-preference; it does not
remove all evaluator bias, so human naturalness review and post-training
audits remain required.

Few-shot exemplars: data/refusal_exemplars.jsonl (CS) and
data/refusal_exemplars_en.jsonl (EN). Exemplars carry a domain_id so an
unseen-domain ablation can exclude its held-out domain from ALL safety
supervision, exemplars included. Out-of-taxonomy exemplars are skipped.
Validation of each generated candidate lives in csjail.prefdata.validate_chosen.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from csjail.artifacts import sha256_text
from csjail.utils.io import read_jsonl

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dpo.yaml"
EXEMPLAR_FILES = {"CS": "data/refusal_exemplars.jsonl", "EN": "data/refusal_exemplars_en.jsonl"}

SYSTEM_PROMPTS = {
    "CS": ("You write short, natural refusals in code-switched Urdu-English (Roman "
           "script, casual register) for a safety-training dataset. Given a harmful "
           "request, write a refusal that: (1) clearly declines, with no hedging and "
           "no partial compliance, (2) briefly explains why in one clause, and (3) "
           "redirects to a safe alternative in one clause. 2-4 sentences. Match the "
           "register and code-switching style of the examples. Output ONLY the refusal."),
    "EN": ("You write short, natural refusals in English for a safety-training dataset. "
           "Given a harmful request, write a refusal that: (1) clearly declines, with no "
           "hedging and no partial compliance, (2) briefly explains why in one clause, and "
           "(3) redirects to a safe alternative in one clause. 2-4 sentences. Match the "
           "style of the examples. Output ONLY the refusal."),
}


@dataclass
class GeneratorConfig:
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 300

    def fingerprint(self, language: str, exemplar_ids: list[str]) -> dict:
        return {**asdict(self), "language": language,
                "system_sha256": sha256_text(SYSTEM_PROMPTS[language]),
                "exemplar_ids": sorted(exemplar_ids)}


def load_generator_config(path: str | Path = CONFIG_PATH) -> GeneratorConfig:
    import yaml

    g = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["prefdata"]["chosen_generator"]
    return GeneratorConfig(provider=g["provider"], model=g["model"],
                           temperature=float(g.get("temperature", 0.7)),
                           max_tokens=int(g.get("max_tokens", 300)))


def assert_distinct_from_judge(gen_cfg: GeneratorConfig, judge_cfg) -> None:
    same_provider = gen_cfg.provider == judge_cfg.provider
    judge_model = getattr(judge_cfg, "model_id", judge_cfg.model)
    if same_provider and gen_cfg.model in (judge_cfg.model, judge_model):
        raise ValueError(
            f"chosen-generator ({gen_cfg.provider}/{gen_cfg.model}) must be DISTINCT from the "
            f"judge ({judge_cfg.provider}/{judge_model}); set configs/dpo.yaml "
            "prefdata.chosen_generator to a different model")


def load_exemplars(language: str = "CS", *, exclude_domains: Iterable[str] = (),
                   path: Optional[str] = None) -> list[dict]:
    excl = set(exclude_domains)
    rows = read_jsonl(path or EXEMPLAR_FILES[language])
    return [r for r in rows if r.get("in_taxonomy", True) and r.get("domain_id") not in excl]


def _build_user_prompt(prompt: str, exemplars: list[dict]) -> str:
    lines = [f"REQUEST: {ex['prompt']}\nREFUSAL: {ex['chosen']}\n" for ex in exemplars]
    lines.append(f"REQUEST: {prompt}\nREFUSAL:")
    return "\n".join(lines)


class ChosenGenerator:
    """Async few-shot refusal generator (Anthropic, or an OpenAI-compatible provider from
    csjail.judge.OPENAI_COMPATIBLE). A provider-side safety block returns None (missing),
    never an empty "refusal"."""

    def __init__(self, cfg: Optional[GeneratorConfig] = None, *, language: str = "CS"):
        from csjail.judge import OPENAI_COMPATIBLE

        self.cfg = cfg or load_generator_config()
        self.language = language
        if self.cfg.provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic  # noqa: WPS433
            except ImportError as e:
                raise RuntimeError("anthropic package not installed: pip install -e '.[judge]'") from e
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY env var is required")
            self._client = AsyncAnthropic(api_key=key)
        elif self.cfg.provider in OPENAI_COMPATIBLE:
            from openai import AsyncOpenAI  # noqa: WPS433

            base_url, key_env = OPENAI_COMPATIBLE[self.cfg.provider]
            key = os.environ.get(key_env) if key_env else "ollama"
            if not key:
                raise RuntimeError(f"{key_env} env var is required for the {self.cfg.provider} generator")
            self._client = AsyncOpenAI(api_key=key, base_url=base_url)
        else:
            raise NotImplementedError(f"provider={self.cfg.provider} not implemented")

    async def _call(self, user: str) -> Optional[str]:
        system = SYSTEM_PROMPTS[self.language]
        if self.cfg.provider == "anthropic":
            # anthropic 1.x dropped the `temperature` keyword; the Claude 4.6/4.5 models
            # still accept it, so it goes in the request body (works on 0.x too).
            resp = await self._client.messages.create(
                model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                extra_body={"temperature": self.cfg.temperature}, system=system,
                messages=[{"role": "user", "content": user}])
            if resp.stop_reason == "refusal":   # a safety classifier blocked the call
                return None
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        resp = await self._client.chat.completions.create(
            model=self.cfg.model, max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None) and not (msg.content or "").strip():
            return None                          # the provider declined to write anything
        return (msg.content or "").strip()

    async def _one(self, prompt: str, exemplars: list[dict]) -> Optional[str]:
        from csjail.judge import _retry_wait

        delay = 1.0
        for _attempt in range(10):
            try:
                return await self._call(_build_user_prompt(prompt, exemplars))
            except (TypeError, ValueError):   # a code/SDK mismatch: fail loudly, never as "missing"
                raise
            except Exception as e:  # broad: SDK raises many types; missing stays missing
                emsg = str(e).lower()
                if "rate limit" in emsg or "rate_limit" in emsg:   # wait the provider's hint, no growth
                    await asyncio.sleep(_retry_wait(emsg, 1.0))
                    continue
                await asyncio.sleep(_retry_wait(emsg, delay))
                delay = min(delay * 2, 30.0)
        return None

    def generate_sync(self, prompts: list[str], exemplars: list[dict], *,
                      concurrency: int = 8) -> list[Optional[str]]:
        async def _all():
            sem = asyncio.Semaphore(concurrency)

            async def _b(p):
                async with sem:
                    return await self._one(p, exemplars)
            return await asyncio.gather(*(_b(p) for p in prompts))
        return asyncio.run(_all())
