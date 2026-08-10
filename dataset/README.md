# Dataset pipeline

End-to-end path from a Quest 3 recording session to a JSONL file
`training/finetune.py` can train on.

```
1. Record on Quest 3        2. Extract frames        3. Annotate
   (passthrough capture         extract_frames.py        Label Studio / CVAT
   or screen recording)     --------------------->   --------------------->
                             dataset/raw/<session>/    dataset/annotations/*.json

4. Convert to training JSONL
   prepare_dataset.py
   ------------------------------------------------->  dataset/processed/train.jsonl
                                                         dataset/processed/images/*.jpg
```

## 1. Record a session on Quest 3

Capture footage of yourself pointing at various physical objects. Two ways
to do this without any custom recording code:

- **Quest's built-in screen recording** (Settings -> System -> Screen
  Capture, or the Sharing button on the wrist menu) while running any app
  with passthrough enabled (even the OS home environment). Simplest option,
  works today with zero code.
- **A dedicated in-app recorder** in `quest-app/` that saves raw passthrough
  camera frames + hand-tracking hit points directly (higher quality, no
  video re-compression artifacts) -- not implemented yet; would live
  alongside `PointingRayController.cs`. Out of scope until the Unity project
  exists (see [quest-app/README.md](../quest-app/README.md)).

Save the resulting video file(s) somewhere accessible from this machine,
e.g. `dataset/raw/session_01/session_01.mp4` (files under `dataset/raw/`
are git-ignored -- they're your private source footage, not committed).

## 2. Extract frames

```powershell
python dataset/extract_frames.py `
  --video dataset/raw/session_01/session_01.mp4 `
  --output_dir dataset/raw/session_01/frames `
  --fps 2
```

`--fps 2` samples 2 frames/sec regardless of the source video's frame rate;
use `--every_n_frames N` instead if you'd rather sample by raw frame count.
Aim for enough frames that each pointing gesture is represented by 3-5
frames from slightly different angles, but not so many near-duplicate
frames that annotation becomes tedious -- 2 fps is a reasonable start.

## 3. Annotate in Label Studio (or CVAT)

Goal: draw one bounding box per frame around the object being pointed at,
labeled with that object's name.

**Label Studio** (recommended for a single annotator):

```bash
pip install label-studio
label-studio start
```

1. Create a project, choose the "Object Detection with Bounding Boxes"
   template.
2. Import the frames from `dataset/raw/session_01/frames/`.
3. Define your label set up front (e.g. "mug", "bottle", "laptop", ...) --
   consistent label strings matter, since `object_label` becomes the
   model's training target verbatim.
4. For each frame, draw a tight box around the pointed-at object only.
5. Export: **Export > COCO** (recommended, works directly with
   `prepare_dataset.py --format coco`) or **Export > JSON** (Label Studio's
   native format, works with `--format label_studio`).
6. Save the export into `dataset/annotations/`.

**CVAT** is a reasonable alternative if you're annotating with a team or
want video-native interpolation between keyframes; export as COCO 1.0 and
use `--format coco`.

## 4. Convert to training JSONL

```powershell
python dataset/prepare_dataset.py `
  --format coco `
  --annotations dataset/annotations/coco_export.json `
  --images_dir dataset/raw/session_01/frames `
  --output_dir dataset/processed/images `
  --output_jsonl dataset/processed/train.jsonl
```

This crops each annotated bounding box out of its source frame (with a
little extra padding for context), saves the crop under
`dataset/processed/images/`, and writes one JSONL line per object:

```json
{"image_path": "dataset/processed/images/frame_000123_obj0.jpg", "object_label": "coffee mug", "bounding_box": [412, 180, 560, 340]}
```

`image_path` points at the cropped image (what the model actually trains
on); `bounding_box` records where that crop came from in the original frame
(kept for provenance/debugging, not used in the training prompt -- see the
docstring in `training/dataset_utils.py`).

Point `training/finetune.py --dataset_path` at the resulting file, or
concatenate multiple sessions' JSONL files together with a plain
`Get-Content session1.jsonl, session2.jsonl | Set-Content combined.jsonl`.

## Dummy dataset for smoke-testing

`dataset/processed/dummy_train.jsonl` + `dataset/processed/dummy_images/`
are a handful of synthetic (non-photographic) images checked into the repo
so `training/finetune.py` is runnable before any real footage exists.
Regenerate them with:

```powershell
python dataset/make_dummy_dataset.py
```

Don't train a real checkpoint on this data -- it exists purely to exercise
the training loop's plumbing (dataset loading, collation, a few optimizer
steps, checkpoint saving).
