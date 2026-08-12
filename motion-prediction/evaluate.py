#!/usr/bin/env python
"""
Compares the trained LSTM trajectory predictor against the no-training
constant-velocity baseline on the held-out validation set, reporting mean
per-keypoint position error at each future timestep. This comparison is
the actual result: if the LSTM doesn't clearly beat the baseline, it isn't
earning its complexity yet (needs more/better data, tuning, etc).

Usage:
    python motion-prediction/evaluate.py \
        --dataset motion-prediction/data/dataset/train.npz \
        --model motion-prediction/checkpoints/lstm_v1.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import PosePredictorLSTM, constant_velocity_baseline


def mean_position_error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Mean Euclidean distance between predicted and actual landmark
    positions, per future timestep, averaged over landmarks and examples.

    predicted/actual shape: (num_examples, future_frames, num_landmarks, num_coords)
    Returns: (future_frames,) array of mean error per timestep.
    """
    per_point_error = np.linalg.norm(predicted - actual, axis=-1)  # (examples, future_frames, landmarks)
    return per_point_error.mean(axis=(0, 2))  # (future_frames,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "data" / "dataset" / "train.npz")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "checkpoints" / "lstm_v1.pt")
    parser.add_argument("--plot_output", type=Path, default=Path(__file__).resolve().parent / "checkpoints" / "error_vs_horizon.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset)
    val_x, val_y = data["val_x"], data["val_y"]
    if len(val_x) == 0:
        raise ValueError(f"No validation windows in {args.dataset} -- nothing to evaluate against.")

    with args.model.with_suffix(".json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = PosePredictorLSTM(
        future_frames=metadata["future_frames"],
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
    ).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=True))
    model.eval()

    with torch.no_grad():
        lstm_predictions = model(torch.from_numpy(val_x).float().to(args.device)).cpu().numpy()

    baseline_predictions = np.stack(
        [constant_velocity_baseline(past, metadata["future_frames"]) for past in val_x]
    )

    lstm_error = mean_position_error(lstm_predictions, val_y)
    baseline_error = mean_position_error(baseline_predictions, val_y)

    print(f"[evaluate] {len(val_x)} validation examples, {metadata['future_frames']} future frames")
    print(f"{'frame':>6} {'baseline_err':>14} {'lstm_err':>14}")
    for t in range(metadata["future_frames"]):
        print(f"{t + 1:>6} {baseline_error[t]:>14.5f} {lstm_error[t]:>14.5f}")

    overall_baseline = baseline_error.mean()
    overall_lstm = lstm_error.mean()
    print(f"\n[evaluate] Mean error across all horizons: baseline={overall_baseline:.5f}  lstm={overall_lstm:.5f}")
    if overall_lstm < overall_baseline:
        improvement = (1 - overall_lstm / overall_baseline) * 100
        print(f"[evaluate] LSTM beats the baseline by {improvement:.1f}%")
    else:
        print("[evaluate] LSTM does NOT beat the constant-velocity baseline -- needs more data/tuning.")

    frames = np.arange(1, metadata["future_frames"] + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(frames, baseline_error, marker="o", label="Constant-velocity baseline")
    plt.plot(frames, lstm_error, marker="o", label="LSTM predictor")
    plt.xlabel("Frames into the future")
    plt.ylabel("Mean position error (normalized coords)")
    plt.title("Prediction error vs. how far ahead we're predicting")
    plt.legend()
    plt.tight_layout()
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_output, dpi=150)
    print(f"[evaluate] Saved plot to {args.plot_output}")


if __name__ == "__main__":
    main()
