"""
Shared pose keypoint schema for the motion-prediction pipeline.

Tracks a fixed subset of MediaPipe Pose's 33-point body landmark topology --
just the ones that matter for upper-body strike motion (shoulders, elbows,
wrists, plus hips for torso-rotation context, since punch power comes from
hip/shoulder rotation, not just the arm).

Every pose sequence in this pipeline -- whether extracted from real video
via extract_poses.py or synthesized by make_dummy_pose_data.py for
smoke-testing -- uses this exact same landmark ordering and array shape, so
downstream code (build_dataset.py, models.py) never needs to know whether
the data is real or synthetic.

Array shape convention used throughout this pipeline:
    (num_frames, NUM_LANDMARKS, NUM_COORDS)
"""

from __future__ import annotations

# (name, MediaPipe Pose landmark index) -- indices match MediaPipe's
# standard 33-point BlazePose topology.
TRACKED_LANDMARKS = [
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_hip", 23),
    ("right_hip", 24),
]

LANDMARK_NAMES = [name for name, _ in TRACKED_LANDMARKS]
LANDMARK_INDICES = [idx for _, idx in TRACKED_LANDMARKS]
NUM_LANDMARKS = len(TRACKED_LANDMARKS)
NUM_COORDS = 3  # x, y, z -- MediaPipe's normalized, image-relative coordinates
