#!/usr/bin/env python
"""
Generates synthetic pose-keypoint sequences so the rest of the pipeline
(build_dataset.py, models.py, train.py, evaluate.py) is testable before any
real boxing footage exists -- mirrors dataset/make_dummy_dataset.py's role
in the object-identification project.

Simulates a crude "jab" motion: the right wrist extends forward and
retracts in a roughly sinusoidal arc, with the shoulder/hip drifting
slightly to fake torso rotation. This is NOT a realistic punch model --
it exists purely to give the trajectory-prediction code real numbers to
run against, so bugs in the pipeline show up now instead of the first
time you feed it real data.

Usage:
    python motion-prediction/make_dummy_pose_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose_schema import LANDMARK_NAMES, NUM_COORDS, NUM_LANDMARKS

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "motion-prediction" / "data" / "poses"

FPS = 30
NUM_CLIPS = 8
CLIP_SECONDS = 3
RNG = np.random.default_rng(42)

# Rough "rest" position for each landmark, in MediaPipe's normalized
# [0, 1] image coordinates (x, y) plus a small z. Arbitrary but plausible
# for a person standing facing the camera, arms in a boxing guard.
REST_POSE = {
    "left_shoulder": (0.40, 0.35, 0.0),
    "right_shoulder": (0.60, 0.35, 0.0),
    "left_elbow": (0.35, 0.50, 0.0),
    "right_elbow": (0.65, 0.50, 0.0),
    "left_wrist": (0.38, 0.40, 0.0),
    "right_wrist": (0.62, 0.40, 0.0),
    "left_hip": (0.42, 0.65, 0.0),
    "right_hip": (0.58, 0.65, 0.0),
}


def make_jab_clip(num_frames: int, punch_start_frame: int, punch_duration_frames: int) -> np.ndarray:
    """Returns an array of shape (num_frames, NUM_LANDMARKS, NUM_COORDS)."""
    clip = np.zeros((num_frames, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

    for frame_idx in range(num_frames):
        # 0 -> 1 -> 0 progress through the punch (extend then retract),
        # 0 outside the punch window entirely.
        t = frame_idx - punch_start_frame
        if 0 <= t < punch_duration_frames:
            phase = t / punch_duration_frames
            extension = np.sin(phase * np.pi)  # smooth out-and-back
        else:
            extension = 0.0

        noise = RNG.normal(scale=0.003, size=(NUM_LANDMARKS, NUM_COORDS))

        for landmark_idx, name in enumerate(LANDMARK_NAMES):
            x, y, z = REST_POSE[name]
            if name == "right_wrist":
                x += extension * 0.30  # extends forward/across
                y -= extension * 0.05
                z -= extension * 0.20
            elif name == "right_elbow":
                x += extension * 0.15
                z -= extension * 0.10
            elif name == "right_shoulder":
                x += extension * 0.04  # slight torso rotation
            elif name == "right_hip":
                x += extension * 0.02

            clip[frame_idx, landmark_idx] = (x, y, z)

        clip[frame_idx] += noise

    return clip


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_frames_per_clip = FPS * CLIP_SECONDS

    for clip_idx in range(NUM_CLIPS):
        punch_start = RNG.integers(low=20, high=num_frames_per_clip - 30)
        punch_duration = RNG.integers(low=8, high=15)  # ~0.25-0.5s at 30fps

        clip = make_jab_clip(num_frames_per_clip, int(punch_start), int(punch_duration))

        clip_name = f"dummy_jab_{clip_idx:03d}"
        np.save(OUTPUT_DIR / f"{clip_name}.npy", clip)

        metadata = {
            "clip_name": clip_name,
            "fps": FPS,
            "num_frames": num_frames_per_clip,
            "landmark_names": LANDMARK_NAMES,
            "source": "synthetic (make_dummy_pose_data.py)",
        }
        with (OUTPUT_DIR / f"{clip_name}.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"Wrote {NUM_CLIPS} synthetic pose clips to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
