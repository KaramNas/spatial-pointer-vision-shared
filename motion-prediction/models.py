"""
Trajectory prediction models: a no-training constant-velocity baseline, and
a small learned LSTM model. The whole point of the baseline is to have
something to prove the learned model actually earns its complexity against
-- see evaluate.py.

Input/output convention:
    past:   (batch, past_frames, NUM_LANDMARKS, NUM_COORDS)
    future: (batch, future_frames, NUM_LANDMARKS, NUM_COORDS)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pose_schema import NUM_COORDS, NUM_LANDMARKS


def constant_velocity_baseline(past: np.ndarray, future_frames: int) -> np.ndarray:
    """Extrapolates in a straight line from the last observed velocity.

    "Velocity" here is estimated from the last two frames of `past` --
    simple, has no free parameters to tune, and is a strong baseline for
    short horizons where motion is close to linear (a punch's very start,
    a ball in flight, etc). Any learned model needs to beat this to be
    worth the added complexity.
    """
    last_frame = past[-1]
    prev_frame = past[-2]
    velocity = last_frame - prev_frame  # per-frame displacement

    steps = np.arange(1, future_frames + 1).reshape(-1, 1, 1)
    return last_frame[None, :, :] + steps * velocity[None, :, :]


class PosePredictorLSTM(nn.Module):
    """Encodes the past-frame sequence with an LSTM, then decodes the whole
    future block in one shot (not autoregressively -- simpler to train and
    evaluate, a standard "direct multi-step" forecasting setup).

    Predicts a DELTA from the last observed frame rather than absolute
    coordinates, then adds it back (`last_frame + delta`). This gives the
    model the same "stay near where you last were" starting point the
    constant-velocity baseline gets for free, so it only has to learn the
    part that's actually hard -- how the motion deviates from that.
    """

    def __init__(
        self,
        future_frames: int,
        num_landmarks: int = NUM_LANDMARKS,
        num_coords: int = NUM_COORDS,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_coords = num_coords
        self.future_frames = future_frames
        self.feature_dim = num_landmarks * num_coords

        self.encoder = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_fc = nn.Linear(hidden_size, future_frames * self.feature_dim)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        batch_size, past_frames = past.shape[0], past.shape[1]
        last_frame = past[:, -1]  # (batch, num_landmarks, num_coords)

        flattened = past.reshape(batch_size, past_frames, self.feature_dim)
        _, (h_n, _) = self.encoder(flattened)
        final_hidden = h_n[-1]  # top layer's final hidden state: (batch, hidden_size)

        delta = self.decoder_fc(final_hidden)
        delta = delta.reshape(batch_size, self.future_frames, self.num_landmarks, self.num_coords)

        return last_frame.unsqueeze(1) + delta
