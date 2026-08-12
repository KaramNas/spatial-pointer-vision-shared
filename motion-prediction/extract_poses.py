#!/usr/bin/env python
"""
Runs MediaPipe Pose over recorded video (single file or a whole directory)
and saves the tracked landmarks (see pose_schema.py) per clip as a .npy
array, in the same format make_dummy_pose_data.py produces -- so
build_dataset.py doesn't care whether a clip is real or synthetic.

Usage:
    python motion-prediction/extract_poses.py --video path/to/clip.mp4
    python motion-prediction/extract_poses.py --video_dir path/to/clips/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from pose_schema import LANDMARK_INDICES, LANDMARK_NAMES, NUM_COORDS, NUM_LANDMARKS

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def extract_pose_sequence(video_path: Path, min_detection_confidence: float) -> tuple[np.ndarray, float, int]:
    """Returns (poses, fps, num_frames_with_no_detection).

    poses has shape (num_frames, NUM_LANDMARKS, NUM_COORDS). Frames where
    MediaPipe fails to detect a person (motion blur on a fast punch, person
    briefly out of frame, etc.) hold the last successfully detected pose
    rather than a hard failure -- flagged via the returned miss count so
    you can decide whether a clip has too many gaps to trust.
    """
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames: list[np.ndarray] = []
    missing_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    ) as pose:
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks is not None:
                    frame_landmarks = np.array(
                        [
                            (
                                results.pose_landmarks.landmark[idx].x,
                                results.pose_landmarks.landmark[idx].y,
                                results.pose_landmarks.landmark[idx].z,
                            )
                            for idx in LANDMARK_INDICES
                        ],
                        dtype=np.float32,
                    )
                else:
                    missing_count += 1
                    frame_landmarks = frames[-1].copy() if frames else np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

                frames.append(frame_landmarks)
        finally:
            cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    return np.stack(frames), fps, missing_count


def process_one(video_path: Path, output_dir: Path, min_detection_confidence: float) -> None:
    poses, fps, missing_count = extract_pose_sequence(video_path, min_detection_confidence)

    if missing_count > 0:
        print(
            f"[extract_poses] WARNING: {video_path.name}: no person detected in "
            f"{missing_count}/{len(poses)} frames (held last known pose for those)"
        )

    clip_name = video_path.stem
    np.save(output_dir / f"{clip_name}.npy", poses)

    metadata = {
        "clip_name": clip_name,
        "fps": fps,
        "num_frames": len(poses),
        "landmark_names": LANDMARK_NAMES,
        "source_video": str(video_path),
        "frames_with_no_detection": missing_count,
    }
    with (output_dir / f"{clip_name}.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[extract_poses] {video_path.name}: {len(poses)} frames -> {output_dir / (clip_name + '.npy')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", type=Path, help="Single video file to process")
    source_group.add_argument("--video_dir", type=Path, help="Directory of video files to process (batch mode)")
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).resolve().parent / "data" / "poses"
    )
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        if not args.video.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
        process_one(args.video, args.output_dir, args.min_detection_confidence)
    else:
        if not args.video_dir.exists():
            raise FileNotFoundError(f"Directory not found: {args.video_dir}")
        video_paths = sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not video_paths:
            raise FileNotFoundError(f"No video files ({sorted(VIDEO_EXTENSIONS)}) found in {args.video_dir}")
        for video_path in video_paths:
            process_one(video_path, args.output_dir, args.min_detection_confidence)


if __name__ == "__main__":
    main()
