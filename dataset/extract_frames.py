#!/usr/bin/env python
"""
Extract still frames from a recorded video (e.g. a Quest 3 passthrough
capture, or a screen recording of a session) at a fixed sampling rate.

This is step 2 of the dataset workflow described in dataset/README.md:
    record on Quest -> extract_frames.py -> annotate in Label Studio ->
    prepare_dataset.py -> dataset/processed/*.jsonl

Usage:
    python dataset/extract_frames.py --video path/to/session.mp4 \
        --output_dir dataset/raw/session_01 --fps 2

    python dataset/extract_frames.py --video path/to/session.mp4 \
        --output_dir dataset/raw/session_01 --every_n_frames 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int,
    max_frames: int | None = None,
    image_format: str = "jpg",
    prefix: str = "frame",
) -> int:
    """Samples every `every_n_frames`-th frame from `video_path` into
    `output_dir`. Returns the number of frames written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    source_frame_idx = 0
    saved_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video

            if source_frame_idx % every_n_frames == 0:
                out_path = output_dir / f"{prefix}_{saved_count:06d}.{image_format}"
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"Failed to write frame to {out_path}")
                saved_count += 1
                if max_frames is not None and saved_count >= max_frames:
                    break

            source_frame_idx += 1
    finally:
        cap.release()

    return saved_count


def resolve_every_n_frames(cap_fps: float, args: argparse.Namespace) -> int:
    if args.every_n_frames is not None:
        return args.every_n_frames
    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError("--fps must be > 0")
        if cap_fps <= 0:
            raise ValueError(
                "Source video reported an invalid FPS; use --every_n_frames instead of --fps"
            )
        return max(1, round(cap_fps / args.fps))
    # Default: keep every frame.
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", required=True, type=Path, help="Path to the source video file")
    parser.add_argument(
        "--output_dir", required=True, type=Path, help="Directory to write extracted frames into"
    )
    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument(
        "--fps", type=float, default=None, help="Target sampling rate in frames/sec"
    )
    rate_group.add_argument(
        "--every_n_frames", type=int, default=None, help="Keep 1 out of every N source frames"
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after saving this many frames")
    parser.add_argument("--image_format", default="jpg", choices=["jpg", "png"])
    parser.add_argument("--prefix", default="frame", help="Filename prefix for saved frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {args.video}")
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    every_n_frames = resolve_every_n_frames(cap_fps, args)
    print(
        f"[extract_frames] Source FPS: {cap_fps:.2f}, sampling every "
        f"{every_n_frames} frame(s) -> {args.output_dir}"
    )

    saved_count = extract_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        every_n_frames=every_n_frames,
        max_frames=args.max_frames,
        image_format=args.image_format,
        prefix=args.prefix,
    )
    print(f"[extract_frames] Wrote {saved_count} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
