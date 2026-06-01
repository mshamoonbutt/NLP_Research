"""vLLM wrapper for the three SLMs. Linux/A100 only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    notes: str = ""


def load_model_registry(path: str | Path = CONFIG_PATH) -> dict[str, ModelSpec]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out: dict[str, ModelSpec] = {}
    for section in ("models", "reference_llms"):
        for key, m in (cfg.get(section) or {}).items():
            out[key] = ModelSpec(
                key=key,
                hf_id=m["hf_id"],
                dtype=m.get("dtype", "bfloat16"),
                max_model_len=int(m.get("max_model_len", 4096)),
                gpu_memory_utilization=float(m.get("gpu_memory_utilization", 0.85)),
                notes=m.get("notes", ""),
            )
    return out


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 0


class SLMRunner:
    """Thin wrapper around vllm.LLM with chat-template application.

    vLLM is imported lazily so unit tests can import this module on Windows
    without CUDA. Instantiating SLMRunner DOES require a working GPU + vllm.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        tensor_parallel_size: int = 1,
        download_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.spec = spec

        try:
            from vllm import LLM  # noqa: WPS433 (lazy import)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "vllm is not installed. Run scripts/setup_a100.sh on the A100 host."
            ) from e

        self._llm = LLM(
            model=spec.hf_id,
            dtype=spec.dtype,
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=download_dir,
            trust_remote_code=trust_remote_code,
            seed=0,
        )

        # Tokenizer reused for chat-template rendering.
        from transformers import AutoTokenizer  # noqa: WPS433

        self._tok = AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=trust_remote_code
        )
        # Some SLM tokenizers lack a chat template; fall back to a minimal one.
        if self._tok.chat_template is None:
            self._tok.chat_template = (
                "{% for m in messages %}"
                "{{ m['role'] | upper }}: {{ m['content'] }}\n"
                "{% endfor %}ASSISTANT:"
            )

    def render(self, user_prompt: str, system: Optional[str] = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_prompt})
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: list[str],
        sampling: SamplingConfig = SamplingConfig(),
        *,
        system: Optional[str] = None,
        show_progress: bool = True,
    ) -> list[str]:
        from vllm import SamplingParams  # noqa: WPS433

        rendered = [self.render(p, system=system) for p in prompts]
        sp = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )
        outputs = self._llm.generate(rendered, sp, use_tqdm=show_progress)
        # vLLM returns RequestOutput; we want the first sampled text per prompt.
        return [o.outputs[0].text for o in outputs]

    @property
    def model_id(self) -> str:
        return self.spec.hf_id

    def shutdown(self) -> None:
        """Release GPU memory between models."""
        try:
            import gc

            import torch  # noqa: WPS433

            del self._llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def resolve(key: str) -> ModelSpec:
    reg = load_model_registry()
    if key not in reg:
        raise KeyError(f"Unknown model key '{key}'. Available: {sorted(reg)}")
    return reg[key]


def runtime_info() -> dict[str, Any]:
    """Provenance: capture installed versions for the results header."""
    info: dict[str, Any] = {}
    for mod_name in ("vllm", "transformers", "torch"):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            info[mod_name] = getattr(mod, "__version__", "?")
        except ImportError:
            info[mod_name] = None
    return info
