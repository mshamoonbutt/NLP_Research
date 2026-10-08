"""Exp 7 — QLoRA-DPO trainer (GPU only; heavy imports are lazy).

Formatting: pairs are passed to TRL in CONVERSATIONAL form
  prompt   = [{"role": "user", "content": ...}]
  chosen   = [{"role": "assistant", "content": ...}]
  rejected = [{"role": "assistant", "content": ...}]
so TRL applies the model's own chat template -- the same template vLLM uses
at evaluation (no system prompt), with the template's own end-of-turn/EOS.
A rendered example is stored in training_manifest.json for inspection.

LoRA targets are LOGICAL projections (q,k,v,o,gate,up,down) resolved against
the loaded architecture: fused modules (qkv_proj, gate_up_proj on Phi-3)
are used when present. A projection that cannot be resolved is an error,
never a silent partial adaptation. Resolved modules and the trainable
parameter count are recorded.

Pinned stack (pyproject [train]) is a candidate set, not GPU-verified; the
recorded versions are compared against it and mismatches are logged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

PINNED = {"trl": "0.12.2", "transformers": "4.46.3", "peft": "0.13.2", "bitsandbytes": "0.44.1"}

PROJECTION_ALIASES: dict[str, list[list[str]]] = {
    # each entry: ordered candidate groups; first group fully present wins
    "q": [["q_proj"], ["qkv_proj"], ["query_key_value"], ["c_attn"]],
    "k": [["k_proj"], ["qkv_proj"], ["query_key_value"], ["c_attn"]],
    "v": [["v_proj"], ["qkv_proj"], ["query_key_value"], ["c_attn"]],
    "o": [["o_proj"], ["out_proj"], ["dense"]],
    "gate": [["gate_proj"], ["gate_up_proj"], ["w1"]],
    "up": [["up_proj"], ["gate_up_proj"], ["w3"]],
    "down": [["down_proj"], ["w2"]],
}


class LoraTargetError(Exception):
    pass


def resolve_lora_targets(linear_leaf_names: Iterable[str], projections: list[str]) -> dict:
    """Map logical projections to the model's actual Linear leaf names."""
    present = set(linear_leaf_names)
    resolved: dict[str, list[str]] = {}
    missing = []
    for p in projections:
        if p not in PROJECTION_ALIASES:
            raise LoraTargetError(f"unknown projection {p!r}")
        hit = next((g for g in PROJECTION_ALIASES[p] if all(x in present for x in g)), None)
        if hit is None:
            missing.append(p)
        else:
            resolved[p] = hit
    if missing:
        raise LoraTargetError(f"projections {missing} not found among Linear modules "
                              f"{sorted(present)[:20]}")
    modules = sorted({m for g in resolved.values() for m in g})
    return {"by_projection": resolved, "target_modules": modules,
            "fused": sorted(p for p, g in resolved.items() if g[0] in
                            ("qkv_proj", "gate_up_proj", "query_key_value", "c_attn"))}


def to_conversational(pair: dict) -> dict:
    return {"prompt": [{"role": "user", "content": pair["prompt"]}],
            "chosen": [{"role": "assistant", "content": pair["chosen"]}],
            "rejected": [{"role": "assistant", "content": pair["rejected"]}]}


def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(f"'{pkg}' is required for DPO training: pip install -e '.[train]' "
                           "on the GPU host") from e


def _versions() -> dict:
    import importlib

    out = {}
    for m in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except ImportError:
            out[m] = None
    return out


def train_dpo(base_hf_id: str, pairs: list[dict], dpo_config: dict[str, Any], output_dir: str, *,
              revision: Optional[str] = None, manifest_extra: Optional[dict] = None) -> str:
    """Fine-tune on `pairs` (already leakage-checked by the caller); save the
    LoRA adapter + training_manifest.json to output_dir; return the path."""
    torch = _require("torch")
    for p in ("transformers", "peft", "trl", "datasets"):
        _require(p)
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    versions = _versions()
    mismatched = {k: (versions.get(k), v) for k, v in PINNED.items() if versions.get(k) != v}
    if mismatched:
        print(f"[train] WARNING: training stack differs from pins: {mismatched}")

    dtype_name = str(dpo_config.get("compute_dtype", "float16"))
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("compute_dtype bfloat16 on a GPU without bf16 (e.g. T4); use float16")
    bnb = None
    if str(dpo_config.get("quantization", "")).startswith("4bit"):
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dtype,
                                 bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base_hf_id, revision=revision, trust_remote_code=True)
    if tok.chat_template is None:
        raise RuntimeError(f"{base_hf_id} has no chat template; refusing to train on raw strings")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_hf_id, revision=revision, quantization_config=bnb, torch_dtype=dtype,
        trust_remote_code=True, device_map="auto")
    if bnb is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    leaf_linear = {name.rsplit(".", 1)[-1] for name, mod in model.named_modules()
                   if mod.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt")}
    lc = dpo_config.get("lora", {})
    targets = resolve_lora_targets(leaf_linear, list(lc.get("target_projections",
                                                            ["q", "k", "v", "o", "gate", "up", "down"])))
    peft_config = LoraConfig(r=int(lc.get("r", 16)), lora_alpha=int(lc.get("alpha", 32)),
                             lora_dropout=float(lc.get("dropout", 0.05)),
                             target_modules=targets["target_modules"], bias="none",
                             task_type="CAUSAL_LM")

    conv = [to_conversational(p) for p in pairs]
    rendered_example = tok.apply_chat_template(conv[0]["prompt"] + conv[0]["chosen"],
                                               tokenize=False) if conv else None
    train_ds = Dataset.from_list(conv)
    args = DPOConfig(
        output_dir=str(out), num_train_epochs=float(dpo_config.get("epochs", 2)),
        per_device_train_batch_size=int(dpo_config.get("per_device_batch", 2)),
        gradient_accumulation_steps=int(dpo_config.get("grad_accum", 8)),
        learning_rate=float(dpo_config.get("learning_rate", 5e-5)),
        lr_scheduler_type=dpo_config.get("lr_scheduler", "cosine"),
        warmup_ratio=float(dpo_config.get("warmup_ratio", 0.1)),
        optim=dpo_config.get("optimizer", "paged_adamw_8bit"),
        beta=float(dpo_config.get("beta", 0.1)), loss_type=dpo_config.get("loss_type", "sigmoid"),
        max_length=int(dpo_config.get("max_length", 1024)),
        max_prompt_length=int(dpo_config.get("max_prompt_length", 512)),
        bf16=dtype is torch.bfloat16, fp16=dtype is torch.float16,
        gradient_checkpointing=True, logging_steps=10, save_strategy="no",
        report_to=[], seed=int(dpo_config.get("seed", 42)))
    trainer = DPOTrainer(model=model, args=args, train_dataset=train_ds, processing_class=tok,
                         peft_config=peft_config)
    if dtype is torch.float16:
        # fp16 AMP cannot unscale fp16 gradients: the trainable LoRA weights stay fp32.
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    adapted = sorted({n.rsplit(".lora_A", 1)[0].rsplit(".", 1)[-1]
                      for n, _ in trainer.model.named_parameters() if ".lora_A" in n})
    if set(adapted) != set(targets["target_modules"]):
        raise LoraTargetError(f"adapted modules {adapted} != resolved {targets['target_modules']}")
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    manifest = {
        "kind": "training_manifest", "base_hf_id": base_hf_id, "revision": revision,
        "n_pairs": len(pairs), "dpo_config": dpo_config, "lora_targets": targets,
        "adapted_modules": adapted, "trainable_params": int(trainable), "versions": versions,
        "pins_mismatched": mismatched, "chat_format": "conversational (TRL applies chat template)",
        "rendered_example": rendered_example, "log_history": trainer.state.log_history,
        **(manifest_extra or {}),
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                           default=str), encoding="utf-8")
    return str(out)
