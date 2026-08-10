#!/usr/bin/env python
"""
Converts annotated frames (COCO-format export from Label Studio/CVAT, or
Label Studio's native JSON export) into the JSONL schema
training/finetune.py expects: one line per object, each with an
already-cropped image_path, object_label, and the original-frame
bounding_box. See dataset/README.md for the full annotation workflow this
script sits in the middle of.

Two input formats are supported via --format:

  coco (default)
      Standard COCO detection JSON: {"images": [...], "annotations": [...],
      "categories": [...]}. This is what CVAT exports, and what Label Studio
      exports if you choose "COCO" on the export screen.

  label_studio
      Label Studio's native "JSON" export (a list of tasks, each with
      data.image and annotations[0].result[].value as PERCENTAGE
      coordinates). Label Studio renames uploaded files with a hash prefix
      (e.g. "3f2a1c-frame_000123.jpg"), so this script strips that prefix
      and matches the remainder against --images_dir by filename suffix.
      If your Label Studio project uploaded files another way, you may need
      to adjust `_resolve_label_studio_filename`.

Usage:
    python dataset/prepare_dataset.py \
        --format coco \
        --annotations dataset/annotations/coco_export.json \
        --images_dir dataset/raw/session_01 \
        --output_dir dataset/processed \
        --output_jsonl dataset/processed/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RawAnnotation:
    """One (image, label, bbox) triple before cropping, bbox in xyxy pixels."""

    file_name: str
    label: str
    bbox_xyxy: tuple[float, float, float, float]


def load_coco(annotations_path: Path) -> list[RawAnnotation]:
    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    categories_by_id = {cat["id"]: cat["name"] for cat in coco["categories"]}

    results = []
    for ann in coco["annotations"]:
        image = images_by_id.get(ann["image_id"])
        if image is None:
            continue
        label = categories_by_id.get(ann["category_id"])
        if label is None:
            continue
        x, y, w, h = ann["bbox"]  # COCO bbox is [x, y, width, height]
        results.append(
            RawAnnotation(
                file_name=image["file_name"],
                label=label,
                bbox_xyxy=(x, y, x + w, y + h),
            )
        )
    return results


_LABEL_STUDIO_HASH_PREFIX = re.compile(r"^[0-9a-fA-F-]{6,}-")


def _resolve_label_studio_filename(data_image_field: str) -> str:
    """Strips Label Studio's local-upload path and hash prefix, e.g.
    '/data/upload/3/8f1c2a-frame_000123.jpg' -> 'frame_000123.jpg'.
    """
    basename = Path(data_image_field).name
    return _LABEL_STUDIO_HASH_PREFIX.sub("", basename)


def load_label_studio(annotations_path: Path) -> list[RawAnnotation]:
    with annotations_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    results = []
    for task in tasks:
        image_field = task.get("data", {}).get("image")
        if not image_field:
            continue
        file_name = _resolve_label_studio_filename(image_field)

        for annotation in task.get("annotations", []):
            for item in annotation.get("result", []):
                value = item.get("value", {})
                labels = value.get("rectanglelabels") or value.get("labels")
                if not labels:
                    continue
                label = labels[0]

                orig_w = item.get("original_width")
                orig_h = item.get("original_height")
                if orig_w is None or orig_h is None:
                    continue

                # Label Studio stores rectangle coords as PERCENTAGES of the
                # original image dimensions.
                x1 = value["x"] / 100.0 * orig_w
                y1 = value["y"] / 100.0 * orig_h
                w = value["width"] / 100.0 * orig_w
                h = value["height"] / 100.0 * orig_h

                results.append(
                    RawAnnotation(
                        file_name=file_name,
                        label=label,
                        bbox_xyxy=(x1, y1, x1 + w, y1 + h),
                    )
                )
    return results


def _find_source_image(images_dir: Path, file_name: str) -> Path | None:
    direct = images_dir / file_name
    if direct.exists():
        return direct
    # Fall back to a suffix match in case of nested export paths.
    matches = list(images_dir.rglob(file_name))
    return matches[0] if matches else None


def crop_and_save(
    source_image_path: Path,
    bbox_xyxy: tuple[float, float, float, float],
    output_path: Path,
    padding_frac: float,
) -> tuple[int, int, int, int]:
    """Crops `source_image_path` around `bbox_xyxy` (with `padding_frac`
    extra context on each side, clamped to image bounds) and saves it to
    `output_path`. Returns the clamped integer bbox actually used.
    """
    with Image.open(source_image_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size

        x1, y1, x2, y2 = bbox_xyxy
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = box_w * padding_frac
        pad_y = box_h * padding_frac

        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(img_w, int(x2 + pad_x))
        crop_y2 = min(img_h, int(y2 + pad_y))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError(f"Degenerate bounding box for {source_image_path}: {bbox_xyxy}")

        cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, quality=92)

        return crop_x1, crop_y1, crop_x2, crop_y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--format", choices=["coco", "label_studio"], default="coco")
    parser.add_argument("--annotations", required=True, type=Path, help="Path to the exported annotation JSON")
    parser.add_argument("--images_dir", required=True, type=Path, help="Directory containing the source (uncropped) frames")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "processed" / "images",
        help="Directory to write cropped training images into",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=REPO_ROOT / "dataset" / "processed" / "train.jsonl",
    )
    parser.add_argument(
        "--padding_frac",
        type=float,
        default=0.15,
        help="Extra context to keep around each bounding box when cropping, "
        "as a fraction of the box's width/height",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    loader = load_coco if args.format == "coco" else load_label_studio
    raw_annotations = loader(args.annotations)
    print(f"[prepare_dataset] Loaded {len(raw_annotations)} annotations from {args.annotations}")

    records = []
    skipped = 0
    counts_per_source: dict[str, int] = {}

    for raw in raw_annotations:
        source_path = _find_source_image(args.images_dir, raw.file_name)
        if source_path is None:
            print(f"[prepare_dataset] WARNING: source image not found for '{raw.file_name}', skipping")
            skipped += 1
            continue

        obj_idx = counts_per_source.get(raw.file_name, 0)
        counts_per_source[raw.file_name] = obj_idx + 1

        crop_name = f"{Path(raw.file_name).stem}_obj{obj_idx}.jpg"
        crop_path = args.output_dir / crop_name

        try:
            clamped_bbox = crop_and_save(source_path, raw.bbox_xyxy, crop_path, args.padding_frac)
        except ValueError as e:
            print(f"[prepare_dataset] WARNING: {e}, skipping")
            skipped += 1
            continue

        records.append(
            {
                "image_path": str(crop_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
                "object_label": raw.label,
                "bounding_box": list(clamped_bbox),
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"[prepare_dataset] Wrote {len(records)} records to {args.output_jsonl} ({skipped} skipped)")


if __name__ == "__main__":
    main()
