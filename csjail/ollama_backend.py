"""CPU fallback generation through a local Ollama server (no CUDA needed).

Same interface as csjail.models.SLMRunner (generate / generate_text /
provenance / shutdown), so the Exp 1 sampler can use either backend.

Scope: DEVELOPMENT / DEBUG ONLY. Ollama serves quantized GGUF weights (the
configured tags are 8-bit q8_0), not the pinned bf16 Hugging Face revisions,
and uses its own Modelfile chat template. Outputs are therefore close to, but
not the same as, the production vLLM outputs. Every record carries the
backend, Ollama model digest, quantization level and template hash, and the
Exp 1 sampler refuses to build a VALIDATION sample with this backend unless
explicitly overridden.

Greedy decoding = temperature 0 with a fixed seed. n > 1 draws are issued as
separate requests with seeds seed, seed+1, ... (Ollama has no `n`).
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import yaml

from csjail.models import CONFIG_PATH, GenOutput, ModelSpec, SamplingConfig

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
BACKEND = "ollama"


def ollama_tag(model_key: str, path: str | Path = CONFIG_PATH) -> str:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tags = cfg.get("ollama_fallback") or {}
    if model_key not in tags:
        raise KeyError(f"no ollama_fallback tag for {model_key!r} in configs/models.yaml")
    return tags[model_key]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class OllamaRunner:
    def __init__(self, spec: ModelSpec, *, tag: Optional[str] = None, host: str = DEFAULT_HOST,
                 workers: int = 2, num_ctx: int = 4096, timeout_s: float = 900.0,
                 adapter_path: Optional[str] = None, **_ignored: Any) -> None:
        if adapter_path:
            raise NotImplementedError("LoRA adapters are not supported by the Ollama fallback")
        self.spec = spec
        self.tag = tag or ollama_tag(spec.key)
        self.host = host.rstrip("/")
        self.workers, self.num_ctx, self.timeout_s = workers, num_ctx, timeout_s
        self.adapter_path, self.adapter_sha256 = None, None
        show = self._post("/api/show", {"model": self.tag})
        self._template = show.get("template") or ""
        self._details = show.get("details") or {}
        self._params = show.get("parameters") or ""
        # A Modelfile SYSTEM line is a default system prompt applied when none
        # is sent (Qwen's has one) -- recorded and folded into the template hash.
        self.default_system = show.get("system")
        if self.default_system is None:
            for line in (show.get("modelfile") or "").splitlines():
                if line.startswith("SYSTEM "):
                    self.default_system = line[len("SYSTEM "):].strip().strip('"')
        self.chat_template_sha256 = _sha(self._template + "\nDEFAULT_SYSTEM:" +
                                         (self.default_system or ""))
        self.template_is_fallback = False
        self.template_probe = self._template        # Ollama Go template text
        self.digest = next((m.get("digest") for m in self._get("/api/tags").get("models", [])
                            if m.get("name") == self.tag or m.get("model") == self.tag), None)
        if not self.digest:
            raise RuntimeError(f"ollama model {self.tag!r} is not pulled")

    # -- http ---------------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.host + path, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    # -- generation ---------------------------------------------------------
    def _one(self, prompt: str, sampling: SamplingConfig, seed: int, system: Optional[str],
             index: int) -> GenOutput:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        out = self._post("/api/chat", {
            "model": self.tag, "messages": msgs, "stream": False,
            "options": {"temperature": sampling.temperature, "top_p": sampling.top_p,
                        "seed": seed, "num_predict": sampling.max_tokens, "num_ctx": self.num_ctx},
        })
        return GenOutput(text=(out.get("message") or {}).get("content", ""), sample_index=index,
                         finish_reason=out.get("done_reason"),
                         n_prompt_tokens=out.get("prompt_eval_count"),
                         n_completion_tokens=out.get("eval_count"))

    def generate(self, prompts: list[str], sampling: SamplingConfig = SamplingConfig(), *,
                 system: Optional[str] = None, show_progress: bool = True) -> list[list[GenOutput]]:
        jobs = [(p, i) for p in prompts for i in range(sampling.n)]

        def run(job):
            p, i = job
            return self._one(p, sampling, sampling.seed + i, system, i)

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(run, jobs))
        it = iter(results)
        return [[next(it) for _ in range(sampling.n)] for _ in prompts]

    def generate_text(self, prompts: list[str], sampling: SamplingConfig = SamplingConfig(), *,
                      system: Optional[str] = None) -> list[str]:
        return [g[0].text for g in self.generate(prompts, sampling, system=system,
                                                 show_progress=False)]

    def provenance(self, system: Optional[str] = None) -> dict[str, Any]:
        return {
            "backend": BACKEND, "model_key": self.spec.key, "hf_id": self.spec.hf_id,
            "revision": None, "reference_hf_revision": self.spec.revision,
            "ollama_tag": self.tag, "ollama_digest": self.digest,
            "quantization": self._details.get("quantization_level"),
            "parameter_size": self._details.get("parameter_size"),
            "ollama_parameters": self._params,
            "chat_template_sha256": self.chat_template_sha256,
            "template_is_fallback": False, "template_probe": self.template_probe,
            "default_system": self.default_system,
            "system_prompt_sha256": _sha(system) if system else None,
            "adapter_path": None, "adapter_sha256": None, "num_ctx": self.num_ctx,
            "runtime": {"ollama": self._get("/api/version").get("version")},
            "scope_note": "development/debug only: quantized GGUF via Ollama, not the "
                          "pinned bf16 HF weights used by the production vLLM backend",
        }

    def shutdown(self) -> None:
        pass
