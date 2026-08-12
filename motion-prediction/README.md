# Motion prediction (sports strike anticipation)

A separate sub-project from the rest of this repo: predicting a sparring
partner's near-future body pose from their recent motion, aimed at boxing
strike anticipation. Different ML problem than the object-identification
pipeline (time-series forecasting on pose keypoints, not vision-language
image classification), so it doesn't touch `training/`, `server/`,
`dataset/`, or `scripts/` -- it's fully self-contained here.

## Status

Working, real, end-to-end -- verified on this machine with synthetic pose
data (see "Smoke test" below). **Not yet trained on any real footage.**
Everything past step 1 below is untested against an actual person because
no real boxing video has been recorded yet.

## Pipeline

```
1. Record video of your sparring partner throwing strikes
   (Quest passthrough, screen-recorded -- or any camera, really)
                    |
                    v
2. extract_poses.py -- MediaPipe Pose extracts 8 upper-body landmarks
   (shoulders, elbows, wrists, hips) per frame
                    |
                    v
3. build_dataset.py -- slides a window across each clip, splitting it into
   (past N frames) -> (future M frames) training pairs. Self-supervised:
   no manual labeling needed for this part.
                    |
                    v
4. train.py -- trains an LSTM to predict future pose from past pose
                    |
                    v
5. evaluate.py -- compares the trained model against a constant-velocity
   baseline (simple straight-line extrapolation), per how-far-ahead
   you're predicting. This comparison is the actual result.
```

The live "ghost skeleton" AR overlay (Quest sees your opponent in real
time, projects a translucent predicted-future skeleton onto them) is a
later phase, not built yet -- see "What's not built yet" below.

## Usage

```powershell
# 1. Install this sub-project's one extra dependency (mediapipe);
#    torch/numpy/opencv are already in the repo's shared .venv.
pip install -r motion-prediction/requirements.txt

# 2. Extract poses from recorded footage
python motion-prediction/extract_poses.py --video_dir path/to/your/clips/

# 3. Build the training dataset (self-supervised windowing)
python motion-prediction/build_dataset.py

# 4. Train
python motion-prediction/train.py --epochs 50

# 5. Evaluate against the baseline
python motion-prediction/evaluate.py
```

Default window sizes: 15 past frames (~0.5s at 30fps) predicting 6 future
frames (~0.2s ahead). Tune via `--past_frames`/`--future_frames` on
`build_dataset.py` -- shorter horizons are easier and more linear (closer
to what the baseline already does well), longer horizons are where a
learned model should pull further ahead of it, if the motion data
supports it.

## Smoke test (synthetic data, no real footage needed)

```powershell
python motion-prediction/make_dummy_pose_data.py   # fake jab motion, not real data
python motion-prediction/build_dataset.py
python motion-prediction/train.py --epochs 30
python motion-prediction/evaluate.py
```

This exercises every stage of the pipeline for real -- training loss
actually drops, the trained model gets compared against the baseline with
real numbers -- but the underlying motion is a crude sinusoidal
approximation of a jab, not real human movement. It exists purely to catch
pipeline bugs before you've recorded real footage, the same role
`dataset/make_dummy_dataset.py` plays for the object-identification
project. Do not read anything into the actual accuracy numbers it
produces.

## Recording real data

- Mount/prop the Quest stationary, camera facing your sparring partner
  (not worn -- this is an external view, not egocentric hand tracking)
- Record via Quest's built-in screen recording, transfer the video file
  to this PC
- Get consent from whoever is being recorded, same as any training
  footage of a person
- `extract_poses.py` accepts a whole directory of clips at once
  (`--video_dir`), so recording many short clips (one strike or a few
  strikes per clip) is easier to manage than one long session

## What's not built yet

- **Punch-type classification** (jab vs. hook vs. uppercut). Not needed
  for trajectory prediction itself (see `build_dataset.py`'s docstring --
  that part is self-supervised), but would need manual labeling if you
  want it later.
- **Live capture + real-time inference server.** Everything above runs
  offline against recorded video. Streaming live from the Quest to a PC
  and running the model in a tight loop is a separate, harder engineering
  task -- get the offline model accurate first.
- **The AR "ghost skeleton" overlay in Unity.** Needs the live pipeline
  above to exist first, plus a new Unity script to render a posed,
  translucent skeleton rig at the predicted position. Not started.
