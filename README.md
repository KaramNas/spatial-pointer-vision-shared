# Spatial Pointer Vision

Point at a physical object with your hand while wearing a Meta Quest 3, and a
fine-tuned vision-language model tells you what it is.

The Quest 3 uses hand tracking + the environment depth mesh to figure out
exactly which object a user's pointing gesture intersects, crops that region
out of the passthrough camera feed, and sends it to a vision-language model
(Qwen2-VL, QLoRA fine-tuned on a small custom "pointed-at object" dataset) for
identification. Inference can run tethered (Quest -> WebSocket -> local PC
running a FastAPI server with a GPU) or, eventually, fully on-device via an
ONNX Runtime Mobile export.

This is a portfolio project. It is built incrementally and documents which
parts are fully working today vs. scaffolded for future work -- see
[Project status](#project-status) at the bottom.

## Architecture

```
Quest 3 (Unity)                     Local PC (tethered mode)
------------------                  --------------------------------
Hand tracking (Meta XR SDK)
        |
        v
PointingRayController  -- ray -->   DepthRaycaster (hits Scene/Depth mesh)
        |                                    |
        v                                    v
                                     ObjectCropper (passthrough camera
                                     frame -> cropped region around hit point)
                                                |
                                                v
                                     InferenceClient --ws--> FastAPI /ws/identify
                                                                    |
                                                                    v
                                                         Qwen2-VL + LoRA adapter
                                                         (server/main.py, GPU)
                                                                    |
                                     ResultOverlay <--ws-- label + confidence
```

## Repository layout

```
spatial-pointer-vision-shared/
├── training/               Python: QLoRA fine-tuning of Qwen2-VL
│   ├── finetune.py
│   ├── dataset_utils.py
│   ├── requirements.txt
│   └── checkpoints/        (git-ignored; LoRA adapter checkpoints land here)
├── dataset/                Python: recording -> frames -> annotation -> JSONL
│   ├── extract_frames.py
│   ├── prepare_dataset.py
│   ├── raw/                (git-ignored; source video/images)
│   ├── annotations/        (git-ignored; Label Studio/CVAT export)
│   ├── processed/          JSONL training data (dummy example is tracked)
│   └── README.md           full annotation workflow
├── server/                 Python: FastAPI + WebSocket inference server
│   ├── main.py
│   ├── model_service.py
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/                Python: export pipeline
│   └── export_onnx.py      merge LoRA -> ONNX -> int8 quantize
├── quest-app/               C#: Unity 2022 LTS Quest 3 client (scaffold)
│   ├── Assets/Scripts/
│   │   ├── PointingRayController.cs
│   │   ├── DepthRaycaster.cs
│   │   ├── ObjectCropper.cs
│   │   ├── InferenceClient.cs
│   │   └── ResultOverlay.cs
│   └── README.md
├── docker-compose.yml       spins up the server container locally
└── .github/workflows/lint.yml
```

## Machine / environment this was built against

- Windows 11, Intel i9-14900KF, 64GB RAM, NVIDIA RTX 4070 (12GB VRAM)
- CUDA 12.1-compatible driver (>= 530.x)
- Python 3.10 or 3.11

A 12GB card is enough for QLoRA (4-bit) fine-tuning of Qwen2-VL-2B-Instruct
comfortably, and Qwen2-VL-7B-Instruct in 4-bit with batch size 1 + gradient
checkpointing if 2B's accuracy isn't sufficient.

## Setup

### 1. Python environment (training + server)

```powershell
# from the repo root
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r training/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r server/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible to PyTorch:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `2.4.1+cu121 True NVIDIA GeForce RTX 4070`.

### 2. Dataset pipeline

See [dataset/README.md](dataset/README.md) for the full recording ->
annotation -> JSONL workflow. A tiny synthetic example lives at
`dataset/processed/dummy_train.jsonl` so the training script is runnable
end-to-end before you have real Quest footage.

### 3. Fine-tune

```powershell
python training/finetune.py `
  --dataset_path dataset/processed/dummy_train.jsonl `
  --output_dir training/checkpoints/dev-run `
  --epochs 1 --batch_size 1 --gradient_accumulation_steps 1 --max_steps 5
```

Run `python training/finetune.py --help` for the full list of CLI flags
(base model, LoRA rank/alpha, learning rate, W&B toggle, etc).

### 4. Run the inference server

```powershell
$env:MODEL_CHECKPOINT_PATH = "training/checkpoints/dev-run"
python server/main.py
```

Or via Docker:

```powershell
docker compose up --build
```

The server exposes:
- `GET /health` -- liveness + whether a model is currently loaded
- `WS /ws/identify` -- send raw image bytes, receive `{"label": ..., "confidence": ...}`

### 5. Export for on-device inference (scaffolded, not yet functional end-to-end)

```powershell
python scripts/export_onnx.py --checkpoint training/checkpoints/dev-run --output scripts/exported/model_int8.onnx
```

See the TODOs inside `scripts/export_onnx.py` for what's still needed to get
an exported model actually running on Quest 3's Snapdragon XR2 Gen 2 chip.

### 6. Quest app

Requires Unity 2022 LTS + the Meta XR SDK, neither of which is installed in
this environment. See [quest-app/README.md](quest-app/README.md).

## Project status

**Working end-to-end today (on this machine, without any hardware):**
- Python env installs cleanly and `torch.cuda.is_available()` is `True`
- `training/finetune.py` runs a real QLoRA training loop against the dummy
  JSONL dataset and writes a LoRA checkpoint to `training/checkpoints/`
- `dataset/extract_frames.py` and `dataset/prepare_dataset.py` run against
  local files
- `server/main.py` loads a real (or dummy-trained) checkpoint and serves
  real Qwen2-VL inference over `/ws/identify`
- `docker-compose.yml` builds and runs the server container
- CI lint workflow runs on every push/PR

**Scaffolded, needs manual setup before it can be tested:**
- `scripts/export_onnx.py` -- ONNX export + int8 quantization logic is real,
  but running it on-device requires the Unity/Android/XR2 toolchain (see
  TODOs in the file)
- `quest-app/` -- all five C# scripts have real method signatures, XML doc
  comments, and a working `InferenceClient.cs` WebSocket client, but
  everything touching the Meta XR SDK (hand tracking, depth mesh, passthrough
  camera) is marked `// TODO: verify against current Meta XR SDK docs` since
  it can't be compiled or tested without Unity + the SDK installed
- Real dataset: you need actual Quest 3 recordings run through Label Studio
  before `prepare_dataset.py` has real input to convert

See each subfolder's README/docstrings for details.
