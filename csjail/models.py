"""vLLM wrapper for the target SLMs (GPU host only).

Provenance recorded for every run: HF id, pinned model/tokenizer revision,
chat-template SHA-256, a rendered template probe (so any default system text
a template inserts is visible), adapter path + weights hash, software
versions. A missing official chat template is an error for paper runs; the
generic fallback exists only behind an explicit debug flag.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"
TEMPLATE_PROBE_TEXT = "<<USER_MESSAGE>>"
_FALLBACK_TEMPLATE = ("{% for m in messages %}{{ m['role'] | upper }}: {{ m['content'] }}\n"
                      "{% endfor %}ASSISTANT:")


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    params_b: Optional[float] = None
    notes: str = ""


def load_model_registry(path: str | Path = CONFIG_PATH) -> dict[str, ModelSpec]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, ModelSpec] = {}
    for section in ("models", "reference_llms"):
        for key, m in (cfg.get(section) or {}).items():
            out[key] = ModelSpec(
                key=key, hf_id=m["hf_id"], revision=m.get("revision"),
                tokenizer_revision=m.get("tokenizer_revision") or m.get("revision"),
                dtype=m.get("dtype", "bfloat16"),
                max_model_len=int(m.get("max_model_len", 4096)),
                gpu_memory_utilization=float(m.get("gpu_memory_utilization", 0.85)),
                params_b=m.get("params_b"), notes=m.get("notes", ""))
    return out


def resolve(key: str) -> ModelSpec:
    reg = load_model_registry()
    if key not in reg:
        raise KeyError(f"Unknown model key '{key}'. Available: {sorted(reg)}")
    return reg[key]


def phase2_models() -> list[str]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return list(cfg.get("phase2_models") or [])


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 0
    n: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenOutput:
    text: str
    sample_index: int
    finish_reason: Optional[str]
    n_prompt_tokens: Optional[int]
    n_completion_tokens: Optional[int]


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def adapter_sha256(adapter_path: Optional[str]) -> Optional[str]:
    if not adapter_path:
        return None
    import hashlib

    h = hashlib.sha256()
    files = sorted(p for p in Path(adapter_path).iterdir()
                   if p.name.startswith("adapter_") and p.is_file())
    if not files:
        raise FileNotFoundError(f"no adapter_* files in {adapter_path}")
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


class SLMRunner:
    """Thin wrapper around vllm.LLM with chat-template application. vLLM is
    imported lazily so CPU tests can import this module."""

    def __init__(self, spec: ModelSpec, *, adapter_path: Optional[str] = None,
                 allow_fallback_template: bool = False, require_pinned: bool = False,
                 tensor_parallel_size: int = 1, download_dir: Optional[str] = None,
                 max_lora_rank: int = 64) -> None:
        if require_pinned and not spec.revision:
            raise RuntimeError(f"model {spec.key} has no pinned `revision` in configs/models.yaml; "
                               "pin it before a paper run (or pass the debug flag)")
        if adapter_path and not Path(adapter_path).exists():
            raise FileNotFoundError(f"adapter not found: {adapter_path}")
        self.spec = spec
        self.adapter_path = adapter_path
        self.adapter_sha256 = adapter_sha256(adapter_path)
        self._lora_request = None
        try:
            from vllm import LLM  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("vllm is not installed (GPU host: pip install -e '.[gpu]')") from e
        from transformers import AutoTokenizer  # noqa: WPS433

        self._tok = AutoTokenizer.from_pretrained(
            spec.hf_id, revision=spec.tokenizer_revision, trust_remote_code=True)
        self.template_is_fallback = self._tok.chat_template is None
        if self.template_is_fallback:
            if not allow_fallback_template:
                raise RuntimeError(f"{spec.hf_id} has no official chat template; refusing to "
                                   "substitute a generic format for a paper run")
            self._tok.chat_template = _FALLBACK_TEMPLATE
        self.chat_template_sha256 = _sha(self._tok.chat_template)
        self.template_probe = self.render(TEMPLATE_PROBE_TEXT)
        self._llm = LLM(
            model=spec.hf_id, revision=spec.revision, tokenizer_revision=spec.tokenizer_revision,
            dtype=spec.dtype, max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size, download_dir=download_dir,
            trust_remote_code=True, seed=0, enable_lora=bool(adapter_path),
            max_lora_rank=max_lora_rank,
        )
        if adapter_path:
            from vllm.lora.request import LoRARequest  # noqa: WPS433

            self._lora_request = LoRARequest("csjail_adapter", 1, adapter_path)

    def render(self, user_prompt: str, system: Optional[str] = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user_prompt}]
        return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: list[str], sampling: SamplingConfig = SamplingConfig(), *,
                 system: Optional[str] = None, show_progress: bool = True) -> list[list[GenOutput]]:
        """One list of `sampling.n` GenOutputs per prompt."""
        from vllm import SamplingParams  # noqa: WPS433

        rendered = [self.render(p, system=system) for p in prompts]
        sp = SamplingParams(temperature=sampling.temperature, top_p=sampling.top_p,
                            max_tokens=sampling.max_tokens, seed=sampling.seed, n=sampling.n)
        outs = self._llm.generate(rendered, sp, use_tqdm=show_progress,
                                  lora_request=self._lora_request)
        result = []
        for o in outs:
            n_prompt = len(o.prompt_token_ids) if o.prompt_token_ids is not None else None
            result.append([GenOutput(c.text, c.index, c.finish_reason, n_prompt, len(c.token_ids))
                           for c in sorted(o.outputs, key=lambda c: c.index)])
        return result

    def generate_text(self, prompts: list[str], sampling: SamplingConfig = SamplingConfig(), *,
                      system: Optional[str] = None) -> list[str]:
        return [g[0].text for g in self.generate(prompts, sampling, system=system,
                                                 show_progress=False)]

    def provenance(self, system: Optional[str] = None) -> dict[str, Any]:
        return {
            "model_key": self.spec.key, "hf_id": self.spec.hf_id,
            "revision": self.spec.revision, "tokenizer_revision": self.spec.tokenizer_revision,
            "dtype": self.spec.dtype, "max_model_len": self.spec.max_model_len,
            "chat_template_sha256": self.chat_template_sha256,
            "template_is_fallback": self.template_is_fallback,
            "template_probe": self.template_probe,
            "system_prompt_sha256": _sha(system) if system else None,
            "adapter_path": self.adapter_path, "adapter_sha256": self.adapter_sha256,
            "runtime": runtime_info(),
        }

    @property
    def model_id(self) -> str:
        return self.spec.hf_id

    def shutdown(self) -> None:
        # vLLM 0.6.x keeps the weights reachable through its parallel state and
        # executor; without tearing those down, the next model loaded in the
        # same process (3-model sweeps on one 15 GB T4) runs out of memory.
        import contextlib
        import gc

        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import (  # noqa: WPS433
                destroy_distributed_environment, destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        with contextlib.suppress(Exception):
            del self._llm.llm_engine.model_executor
        with contextlib.suppress(Exception):
            del self._llm
        gc.collect()
        with contextlib.suppress(Exception):
            import torch  # noqa: WPS433

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def runtime_info() -> dict[str, Any]:
    import importlib

    info: dict[str, Any] = {}
    for mod_name in ("vllm", "transformers", "torch", "trl", "peft"):
        try:
            info[mod_name] = getattr(importlib.import_module(mod_name), "__version__", "?")
        except ImportError:
            info[mod_name] = None
    return info


@dataclass
class ModelIdentity:
    """Everything that defines a generation 'system' for cache keys."""
    model_key: str
    hf_id: str
    revision: Optional[str]
    chat_template_sha256: Optional[str]
    adapter_sha256: Optional[str]
    system_prompt_sha256: Optional[str]
    arm: str = "A"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_runner(cls, runner: SLMRunner, *, arm: str, system: Optional[str]) -> "ModelIdentity":
        p = runner.provenance(system)
        return cls(p["model_key"], p["hf_id"], p["revision"], p["chat_template_sha256"],
                   p["adapter_sha256"], p["system_prompt_sha256"], arm)

    def as_dict(self) -> dict:
        return asdict(self)
