#!/usr/bin/env python
"""
Generates a tiny synthetic dataset so training/finetune.py and
dataset/prepare_dataset.py are testable before any real Quest footage
exists. Produces a handful of solid-color/shape images with obvious labels
and the matching dataset/processed/dummy_train.jsonl.

This is a one-off dev utility, not part of the real data pipeline -- see
dataset/README.md for how actual training data is produced.

Usage:
    python dataset/make_dummy_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "dataset" / "processed" / "dummy_images"
JSONL_PATH = REPO_ROOT / "dataset" / "processed" / "dummy_train.jsonl"

# (filename, object_label, fill_color, shape)
EXAMPLES = [
    ("dummy_000_red_mug.jpg", "red mug", (200, 40, 40), "ellipse"),
    ("dummy_001_blue_bottle.jpg", "blue bottle", (40, 70, 200), "rectangle"),
    ("dummy_002_green_plant.jpg", "green plant", (50, 160, 60), "ellipse"),
    ("dummy_003_yellow_notebook.jpg", "yellow notebook", (220, 200, 40), "rectangle"),
]

IMAGE_SIZE = (224, 224)


def make_image(fill_color: tuple[int, int, int], shape: str) -> Image.Image:
    img = Image.new("RGB", IMAGE_SIZE, color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    box = (40, 40, IMAGE_SIZE[0] - 40, IMAGE_SIZE[1] - 40)
    if shape == "ellipse":
        draw.ellipse(box, fill=fill_color, outline=(0, 0, 0))
    else:
        draw.rectangle(box, fill=fill_color, outline=(0, 0, 0))
    return img


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for filename, label, color, shape in EXAMPLES:
        img = make_image(color, shape)
        out_path = IMAGES_DIR / filename
        img.save(out_path, quality=90)

        records.append(
            {
                "image_path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "object_label": label,
                # Full-frame box: these dummy images are already "pre-cropped"
                # synthetic examples, so the box just covers the whole image.
                "bounding_box": [0, 0, IMAGE_SIZE[0], IMAGE_SIZE[1]],
            }
        )

    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} images to {IMAGES_DIR}")
    print(f"Wrote {JSONL_PATH}")


if __name__ == "__main__":
    main()
