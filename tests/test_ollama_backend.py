"""Ollama CPU fallback: request mapping, provenance, and scope restrictions."""
from __future__ import annotations

import io
import json

import pytest

import csjail.ollama_backend as ob
from csjail.models import SamplingConfig, resolve


class FakeHTTP:
    def __init__(self):
        self.chat_bodies = []

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/show"):
            body = {"template": "{{ .System }}<|user|>{{ .Prompt }}",
                    "details": {"quantization_level": "Q8_0", "parameter_size": "3.8B"},
                    "parameters": "stop <|end|>"}
        elif url.endswith("/api/tags"):
            body = {"models": [{"name": "phi3:3.8b-mini-4k-instruct-q8_0", "digest": "sha-abc"}]}
        elif url.endswith("/api/version"):
            body = {"version": "0.34.3"}
        else:
            sent = json.loads(req.data.decode("utf-8"))
            self.chat_bodies.append(sent)
            body = {"message": {"content": f"reply seed={sent['options']['seed']}"},
                    "done_reason": "stop", "prompt_eval_count": 7, "eval_count": 3}
        return io.BytesIO(json.dumps(body).encode("utf-8"))


@pytest.fixture
def http(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(ob.urllib.request, "urlopen", fake)
    return fake


def test_tag_lookup_from_config():
    assert ob.ollama_tag("phi3") == "phi3:3.8b-mini-4k-instruct-q8_0"
    with pytest.raises(KeyError):
        ob.ollama_tag("nope")


def test_generate_maps_greedy_and_draws(http):
    r = ob.OllamaRunner(resolve("phi3"), workers=1)
    outs = r.generate(["a", "b"], SamplingConfig(temperature=0.0, max_tokens=64, seed=5, n=1))
    assert [o[0].text for o in outs] == ["reply seed=5", "reply seed=5"]
    assert outs[0][0].finish_reason == "stop" and outs[0][0].n_completion_tokens == 3
    opts = http.chat_bodies[0]["options"]
    assert opts["temperature"] == 0.0 and opts["num_predict"] == 64 and "system" not in \
        json.dumps(http.chat_bodies[0]["messages"])
    draws = r.generate(["c"], SamplingConfig(temperature=0.7, n=3, seed=10))
    assert [d.text for d in draws[0]] == ["reply seed=10", "reply seed=11", "reply seed=12"]
    assert [d.sample_index for d in draws[0]] == [0, 1, 2]


def test_provenance_marks_nonproduction(http):
    p = ob.OllamaRunner(resolve("phi3")).provenance()
    assert p["backend"] == "ollama" and p["revision"] is None
    assert p["quantization"] == "Q8_0" and p["ollama_digest"] == "sha-abc"
    assert "development/debug only" in p["scope_note"]


def test_missing_model_and_adapters_rejected(http, monkeypatch):
    with pytest.raises(NotImplementedError):
        ob.OllamaRunner(resolve("phi3"), adapter_path="x")
    with pytest.raises(RuntimeError, match="not pulled"):
        ob.OllamaRunner(resolve("qwen25"))


def test_show_404_is_reported_as_not_pulled(monkeypatch):
    def raise_404(req, timeout=None):
        raise ob.urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(ob.urllib.request, "urlopen", raise_404)
    with pytest.raises(RuntimeError, match="ollama pull"):
        ob.OllamaRunner(resolve("phi3"))


def test_validation_sample_refuses_ollama():
    import importlib.util
    import sys
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "scripts" / "exp1_sample_for_annotation.py"
    spec = importlib.util.spec_from_file_location("exp1_s", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp1_s"] = mod
    spec.loader.exec_module(mod)
    assert mod.main(["--role", "validation", "--backend", "ollama"]) == 1
