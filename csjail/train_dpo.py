"""Exp 7 — QLoRA-DPO trainer (GPU only).

Fine-tunes a 4-bit NF4 base with a LoRA adapter using trl's DPOTrainer. Runs on
the RTX 4080 (16 GB) comfortably for sub-4B models. All heavy deps
(torch/transformers/trl/peft/bitsandbytes/datasets) are imported lazily inside
`train_dpo`, so importing this module on a CPU-only box (for tests) does not
require them.

The DPO loss is left pluggable via `dpo_config["loss_type"]` (trl supports
"sigmoid" [standard DPO], "ipo", "kto_pair", ...), leaving room for the optional
MPO / CS-weighted arms (F/G) without rewriting the harness.

Install on the GPU host:  pip install -e ".[train]"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"'{pkg}' is required for DPO training. Install training deps on the "
            f"GPU host: pip install -e \".[train]\"  (needs a CUDA GPU)."
        ) from e


def _load_pairs_as_dataset(pairs_path: str):
    """Load {prompt,chosen,rejected} JSONL into a HF Dataset."""
    _require("datasets")
    from datasets import Dataset

    import json
    rows = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if all(k in r and r[k] for k in ("prompt", "chosen", "rejected")):
                rows.append({
                    "prompt": r["prompt"],
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                })
    if not rows:
        raise ValueError(f"no valid preference pairs in {pairs_path}")
    return Dataset.from_list(rows)


def train_dpo(
    base_hf_id: str,
    pairs_path: str,
    dpo_config: dict[str, Any],
    output_dir: str,
    val_pairs_path: Optional[str] = None,
) -> str:
    """Fine-tune `base_hf_id` on preference pairs; save LoRA adapter to
    `output_dir`; return the adapter path.

    `dpo_config` keys (see configs/dpo.yaml): lora{r,alpha,dropout,target_modules},
    beta, learning_rate, epochs, per_device_batch, grad_accum, max_length,
    max_prompt_length, optimizer, lr_scheduler, warmup_ratio, loss_type,
    quantization ("4bit_nf4"|None), optional_sft_warmup.
    """
    torch = _require("torch")
    _require("transformers")
    _require("peft")
    _require("trl")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 4-bit NF4 quantization for the frozen base (QLoRA).
    quant = dpo_config.get("quantization", "4bit_nf4")
    bnb_config = None
    if quant and quant.startswith("4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tok = AutoTokenizer.from_pretrained(base_hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_hf_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lc = dpo_config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lc.get("r", 16)),
        lora_alpha=int(lc.get("alpha", 32)),
        lora_dropout=float(lc.get("dropout", 0.05)),
        target_modules=lc.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds = _load_pairs_as_dataset(pairs_path)
    eval_ds = _load_pairs_as_dataset(val_pairs_path) if val_pairs_path else None

    # Optional light SFT warmup on the `chosen` refusals before DPO — helps when
    # plain DPO underfits (config flag). Kept minimal and off by default.
    if dpo_config.get("optional_sft_warmup"):
        _sft_warmup(model, tok, train_ds, dpo_config, str(out / "_sft_tmp"))

    dpo_args = DPOConfig(
        output_dir=str(out),
        num_train_epochs=float(dpo_config.get("epochs", 2)),
        per_device_train_batch_size=int(dpo_config.get("per_device_batch", 2)),
        gradient_accumulation_steps=int(dpo_config.get("grad_accum", 8)),
        learning_rate=float(dpo_config.get("learning_rate", 5e-5)),
        lr_scheduler_type=dpo_config.get("lr_scheduler", "cosine"),
        warmup_ratio=float(dpo_config.get("warmup_ratio", 0.1)),
        optim=dpo_config.get("optimizer", "paged_adamw_8bit"),
        beta=float(dpo_config.get("beta", 0.1)),
        loss_type=dpo_config.get("loss_type", "sigmoid"),
        max_length=int(dpo_config.get("max_length", 1024)),
        max_prompt_length=int(dpo_config.get("max_prompt_length", 512)),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=int(dpo_config.get("seed", 42)),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    return str(out)


def _sft_warmup(model, tok, train_ds, dpo_config, output_dir: str) -> None:
    """Minimal SFT pass on the `chosen` responses (prompt -> chosen)."""
    from trl import SFTConfig, SFTTrainer

    def _fmt(ex):
        return {"text": ex["prompt"] + "\n" + ex["chosen"]}

    sft_ds = train_ds.map(_fmt)
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=int(dpo_config.get("per_device_batch", 2)),
        gradient_accumulation_steps=int(dpo_config.get("grad_accum", 8)),
        learning_rate=float(dpo_config.get("learning_rate", 5e-5)),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        report_to=[],
        max_seq_length=int(dpo_config.get("max_length", 1024)),
    )
    SFTTrainer(model=model, args=sft_args, train_dataset=sft_ds).train()
