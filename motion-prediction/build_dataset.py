#!/usr/bin/env python
"""
Turns extracted pose-keypoint clips (.npy files from extract_poses.py or
make_dummy_pose_data.py) into (past, future) training windows for the
trajectory predictor.

This step is self-supervised: it needs NO manual labeling. Any window of
consecutive frames can be split into "past" and "future" halves and used
as a training example, since we're just teaching the model "given recent
motion, what comes next" -- not classifying what type of motion it is.
(Punch-type classification, if you want it later, is a separate
supervised task that *would* need manual labels -- out of scope here.)

Usage:
    python motion-prediction/build_dataset.py \
        --poses_dir motion-prediction/data/poses \
        --output motion-prediction/data/dataset/train.npz \
        --past_frames 15 --future_frames 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pose_schema import NUM_COORDS, NUM_LANDMARKS


def load_clips(poses_dir: Path) -> list[np.ndarray]:
    clip_paths = sorted(poses_dir.glob("*.npy"))
    if not clip_paths:
        raise FileNotFoundError(
            f"No .npy pose clips found in {poses_dir}. Run extract_poses.py "
            "on real footage, or make_dummy_pose_data.py for a smoke test."
        )

    clips = []
    for path in clip_paths:
        clip = np.load(path)
        if clip.ndim != 3 or clip.shape[1:] != (NUM_LANDMARKS, NUM_COORDS):
            raise ValueError(
                f"{path} has shape {clip.shape}, expected (num_frames, "
                f"{NUM_LANDMARKS}, {NUM_COORDS}) -- see pose_schema.py"
            )
        clips.append(clip)
    return clips


def windows_from_clip(clip: np.ndarray, past_frames: int, future_frames: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Slides a (past_frames + future_frames)-length window across `clip`,
    returning stacked (X_past, Y_future) arrays.
    """
    window_len = past_frames + future_frames
    num_frames = clip.shape[0]

    x_list, y_list = [], []
    for start in range(0, num_frames - window_len + 1, stride):
        window = clip[start : start + window_len]
        x_list.append(window[:past_frames])
        y_list.append(window[past_frames:])

    if not x_list:
        return (
            np.empty((0, past_frames, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32),
            np.empty((0, future_frames, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32),
        )
    return np.stack(x_list), np.stack(y_list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--poses_dir", type=Path, default=Path(__file__).resolve().parent / "data" / "poses")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "dataset" / "train.npz")
    parser.add_argument("--past_frames", type=int, default=15, help="~0.5s of history at 30fps")
    parser.add_argument("--future_frames", type=int, default=6, help="~0.2s of future to predict at 30fps")
    parser.add_argument("--stride", type=int, default=2, help="Frames to skip between consecutive windows (smaller = more overlap = more examples)")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of CLIPS (not windows) held out for validation, to avoid leakage between train/val from the same clip")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = load_clips(args.poses_dir)
    print(f"[build_dataset] Loaded {len(clips)} clips from {args.poses_dir}")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(clips))
    num_val_clips = max(1, int(len(clips) * args.val_fraction)) if len(clips) > 1 else 0
    val_clip_indices = set(indices[:num_val_clips].tolist())

    train_x, train_y, val_x, val_y = [], [], [], []
    for clip_idx, clip in enumerate(clips):
        x, y = windows_from_clip(clip, args.past_frames, args.future_frames, args.stride)
        if clip_idx in val_clip_indices:
            val_x.append(x)
            val_y.append(y)
        else:
            train_x.append(x)
            train_y.append(y)

    def concat_or_empty(arrays: list[np.ndarray], shape_tail: tuple[int, ...]) -> np.ndarray:
        non_empty = [a for a in arrays if len(a) > 0]
        if not non_empty:
            return np.empty((0,) + shape_tail, dtype=np.float32)
        return np.concatenate(non_empty)

    train_x = concat_or_empty(train_x, (args.past_frames, NUM_LANDMARKS, NUM_COORDS))
    train_y = concat_or_empty(train_y, (args.future_frames, NUM_LANDMARKS, NUM_COORDS))
    val_x = concat_or_empty(val_x, (args.past_frames, NUM_LANDMARKS, NUM_COORDS))
    val_y = concat_or_empty(val_y, (args.future_frames, NUM_LANDMARKS, NUM_COORDS))

    print(f"[build_dataset] {len(train_x)} training windows, {len(val_x)} validation windows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
    )
    print(f"[build_dataset] Wrote dataset to {args.output}")


if __name__ == "__main__":
    main()
