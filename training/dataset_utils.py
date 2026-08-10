"""
Dataset utilities for fine-tuning Qwen2-VL on the "what am I pointing at?" task.

Expected JSONL schema (one JSON object per line) -- see dataset/README.md for
how real training data gets produced:

    {
        "image_path": "dataset/processed/images/000123.jpg",
        "object_label": "coffee mug",
        "bounding_box": [x1, y1, x2, y2]
    }

`bounding_box` gives the pixel coordinates in the ORIGINAL (uncropped) frame
that `image_path` was cropped from. It is kept for provenance/debugging and
is NOT used to build the training prompt.

Design note: by the time an image reaches this dataset, it has already been
cropped tightly around the pointed-at object -- either by ObjectCropper.cs on
the headset in production, or by dataset/prepare_dataset.py when converting
offline annotations. The model therefore always sees the same kind of input
at train time and at serving time (server/main.py): a single-object crop, no
box coordinates in the prompt. That keeps the fine-tuning task simple
("name the object in this image") and matches the WebSocket contract the
Quest app actually uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

DEFAULT_PROMPT = "What object is this? Answer with just the object's common name."

REQUIRED_FIELDS = ("image_path", "object_label", "bounding_box")


class PointingDataset(Dataset):
    """Loads (cropped image, object label) pairs from a JSONL file.

    Images are opened lazily in __getitem__ rather than preloaded into memory,
    since a real dataset could contain thousands of frames.
    """

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Dataset JSONL not found: {self.jsonl_path}. "
                "See dataset/README.md to generate one, or point --dataset_path "
                "at dataset/processed/dummy_train.jsonl for a smoke test."
            )

        self.examples: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                missing = [field for field in REQUIRED_FIELDS if field not in record]
                if missing:
                    raise ValueError(
                        f"{self.jsonl_path}:{line_num} is missing required "
                        f"field(s) {missing}. Expected schema: {REQUIRED_FIELDS}"
                    )
                self.examples.append(record)

        if not self.examples:
            raise ValueError(f"No examples found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.examples[idx]
        image_path = Path(record["image_path"])
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return {
            "image": image,
            "object_label": record["object_label"],
            "bounding_box": record["bounding_box"],
        }


def build_collate_fn(processor, prompt: str = DEFAULT_PROMPT):
    """Builds a batch collate function that turns raw examples into the
    input_ids / attention_mask / pixel_values / labels tensors Trainer needs.

    The prompt portion of each sequence is masked out of the loss (labels set
    to -100) so the model is only trained to predict the object name, not to
    reproduce the fixed instruction text.

    Caveat: prompt-token-count is computed by tokenizing the prompt-only
    chat template separately and counting its non-pad tokens, then assuming
    that prefix lines up with the same tokens at the start of the full
    (prompt + answer) sequence. This holds in practice for Qwen2's BPE
    tokenizer given a fixed prompt template, but isn't a formal guarantee for
    arbitrary tokenizers -- worth revisiting if you swap the base model.
    """

    def collate_fn(batch: list[dict[str, Any]]):
        images = [example["image"] for example in batch]

        full_texts = [
            processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": example["object_label"]}],
                    },
                ],
                tokenize=False,
            )
            for example in batch
        ]

        prompt_only_texts = [
            processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _ in batch
        ]

        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_inputs = processor(
            text=prompt_only_texts, images=images, padding=True, return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100  # padding

        for i in range(len(batch)):
            prompt_len = int(prompt_inputs["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn
