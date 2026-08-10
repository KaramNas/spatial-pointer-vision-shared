#!/usr/bin/env python
"""
QLoRA fine-tuning of Qwen2-VL for the "what am I pointing at?" task.

Loads the base model in 4-bit (bitsandbytes NF4), attaches a LoRA adapter
over the language-model attention/MLP projections, and trains on a JSONL
dataset of (cropped image, object label) pairs (see dataset/README.md).

Quick smoke test (a few seconds on an RTX 4070, uses the dummy dataset):

    python training/finetune.py \
        --dataset_path dataset/processed/dummy_train.jsonl \
        --output_dir training/checkpoints/dev-run \
        --epochs 1 --batch_size 1 --gradient_accumulation_steps 1 --max_steps 5

Run with --help for the full flag list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_utils import DEFAULT_PROMPT, PointingDataset, build_collate_fn  # noqa: E402

# Language-model attention + MLP projections. Left out of the default target
# list are the vision tower's projections ("attn.qkv", "attn.proj", "mlp.fc1",
# "mlp.fc2") -- pass --lora_target_modules explicitly to also adapt those;
# skipping them keeps a first run smaller/faster on a 12GB card.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model / data
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model. Fall back to Qwen/Qwen2-VL-7B-Instruct if 2B's "
        "accuracy isn't sufficient (still fits in 4-bit on 12GB with "
        "batch_size 1 + gradient checkpointing).",
    )
    parser.add_argument("--dataset_path", default="dataset/processed/dummy_train.jsonl")
    parser.add_argument("--output_dir", default="training/checkpoints/run")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)

    # Training loop
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If set (>0), overrides --epochs with a fixed optimizer-step count. "
        "Useful for smoke tests on the dummy dataset.",
    )
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA / QLoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", nargs="+", default=DEFAULT_TARGET_MODULES
    )

    # Logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Log metrics to Weights & Biases. Requires `wandb login` (or "
        "WANDB_API_KEY) to already be configured. Off by default so the "
        "script runs with zero external accounts.",
    )
    parser.add_argument("--wandb_project", default="spatial-pointer-vision")

    return parser.parse_args()


def load_model(args: argparse.Namespace):
    print(f"[finetune] Loading base model in 4-bit: {args.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    report_to = "none"
    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))
        report_to = "wandb"

    print(f"[finetune] Loading dataset from {args.dataset_path}")
    dataset = PointingDataset(args.dataset_path)
    print(f"[finetune] {len(dataset)} training examples")

    print(f"[finetune] Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    # Right-padding is required so the prompt-length label-masking trick in
    # dataset_utils.build_collate_fn lines up correctly (Qwen2's default is
    # left-padding, which is meant for generation, not training).
    processor.tokenizer.padding_side = "right"

    model = load_model(args)
    collate_fn = build_collate_fn(processor, prompt=args.prompt)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        remove_unused_columns=False,  # our collate_fn needs the raw dict fields
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print("[finetune] Starting training...")
    trainer.train()

    print(f"[finetune] Saving LoRA adapter + processor to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("[finetune] Done.")


if __name__ == "__main__":
    main()
