#!/usr/bin/env python
"""
Trains the LSTM trajectory predictor on a dataset built by build_dataset.py.

Usage:
    python motion-prediction/train.py \
        --dataset motion-prediction/data/dataset/train.npz \
        --output motion-prediction/checkpoints/lstm_v1.pt \
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import PosePredictorLSTM


def load_dataset(path: Path):
    data = np.load(path)
    return (
        torch.from_numpy(data["train_x"]).float(),
        torch.from_numpy(data["train_y"]).float(),
        torch.from_numpy(data["val_x"]).float(),
        torch.from_numpy(data["val_y"]).float(),
        int(data["past_frames"]),
        int(data["future_frames"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "data" / "dataset" / "train.npz")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "checkpoints" / "lstm_v1.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_x, train_y, val_x, val_y, past_frames, future_frames = load_dataset(args.dataset)
    print(f"[train] {len(train_x)} training windows, {len(val_x)} validation windows "
          f"(past={past_frames} frames, future={future_frames} frames)")

    if len(train_x) == 0:
        raise ValueError(
            f"No training windows in {args.dataset}. Record more/longer clips, "
            "or reduce --past_frames/--future_frames in build_dataset.py."
        )

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)

    model = PosePredictorLSTM(
        future_frames=future_frames, hidden_size=args.hidden_size, num_layers=args.num_layers
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)

            optimizer.zero_grad()
            predicted = model(batch_x)
            loss = loss_fn(predicted, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_x)
        epoch_loss /= len(train_x)

        val_loss = None
        if len(val_x) > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x.to(args.device))
                val_loss = loss_fn(val_pred, val_y.to(args.device)).item()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            msg = f"[train] epoch {epoch}/{args.epochs}  train_mse={epoch_loss:.6f}"
            if val_loss is not None:
                msg += f"  val_mse={val_loss:.6f}"
            print(msg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    metadata = {
        "future_frames": future_frames,
        "past_frames": past_frames,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train] Saved model to {args.output}")


if __name__ == "__main__":
    main()
