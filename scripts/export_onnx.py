#!/usr/bin/env python
"""
Export pipeline: merges a training/finetune.py LoRA checkpoint into the base
Qwen2-VL model, exports a single forward pass to ONNX, and applies dynamic
int8 quantization aimed at the on-device (Snapdragon XR2 Gen 2) target.

What this script actually does (real, testable on this machine):
    1. Load the base model + LoRA adapter, merge them into one set of weights
       (peft's merge_and_unload) so no PEFT runtime is needed downstream.
    2. Wrap the model's forward() (a single teacher-forced pass over
       input_ids/pixel_values -> logits) in an nn.Module with a fixed,
       ONNX-friendly positional-argument signature, and export it with
       torch.onnx.export.
    3. Sanity-check the exported graph with onnx.checker and (optionally) an
       onnxruntime CPU inference session.
    4. Apply dynamic int8 weight quantization via
       onnxruntime.quantization.quantize_dynamic (no calibration dataset
       required -- activations are quantized dynamically at runtime, weights
       are quantized ahead of time).

What this does NOT do yet, and why (see TODOs inline below):
    - Export the full autoregressive generate() loop with KV-cache. ONNX
      export works on a fixed computation graph; Qwen2-VL's incremental
      decoding with a growing KV cache needs either a loop unrolled to a
      fixed max length, or a decoder graph with explicit past_key_values
      inputs/outputs wired up by hand (see optimum's encoder-decoder ONNX
      export for the general pattern). This script exports one forward pass
      only, which is enough to validate the graph topology/quantization
      pipeline but is NOT yet a drop-in replacement for `model.generate()`.
    - Split the vision tower and language model into two separate ONNX
      graphs. On-device, you very likely want to run the (expensive, but
      input-shape-dependent) vision encoder once per crop, then run the
      (cheap, incremental) language model decoder in a loop -- as one fused
      graph the whole vision tower re-runs on every decode step, which is
      wasteful. Splitting them is the natural next step once the single-pass
      export above is validated.
    - Convert the quantized .onnx to the .ort format ONNX Runtime Mobile
      actually loads, or wire it into an Android/Unity build. See the
      end-of-file TODO block.

Usage:
    python scripts/export_onnx.py \
        --checkpoint training/checkpoints/dev-run \
        --output scripts/exported/model_int8.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# A tiny solid-color placeholder image used purely to trace the export graph
# with *some* valid pixel_values/image_grid_thw shape. Any real image works
# equally well here -- shapes are what matter for tracing, not content.
_TRACE_IMAGE_SIZE = (224, 224)


class Qwen2VLForwardOnly(nn.Module):
    """Thin wrapper exposing a fixed, ONNX-friendly positional signature over
    Qwen2VLForConditionalGeneration.forward(). torch.onnx.export needs a
    plain tuple of tensor args in a stable order -- the HF model's forward()
    takes everything as keyword args with many optional/None defaults, which
    the tracer handles poorly.
    """

    def __init__(self, model: Qwen2VLForConditionalGeneration):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        return outputs.logits


def load_merged_model(checkpoint: Path, base_model_name: str):
    print(f"[export_onnx] Loading base model: {base_model_name}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=torch.float32
    )

    print(f"[export_onnx] Loading + merging LoRA adapter from {checkpoint}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    model = model.merge_and_unload()
    model.eval()

    processor_source = str(checkpoint) if (checkpoint / "preprocessor_config.json").exists() else base_model_name
    processor = AutoProcessor.from_pretrained(processor_source)

    return model, processor


def build_trace_inputs(processor, prompt: str):
    placeholder_image = Image.new("RGB", _TRACE_IMAGE_SIZE, color=(128, 128, 128))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[placeholder_image], return_tensors="pt")
    return inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"]


def export_to_onnx(model, trace_inputs, output_path: Path, opset: int) -> None:
    input_ids, attention_mask, pixel_values, image_grid_thw = trace_inputs
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = Qwen2VLForwardOnly(model)

    print(f"[export_onnx] Exporting to {output_path} (opset {opset})")
    # NOTE: Qwen2-VL's vision rotary-embedding / M-RoPE logic is intricate
    # enough that the legacy TorchScript-based exporter (used here) is not
    # guaranteed to trace every op cleanly on every transformers version. If
    # this raises an unsupported-op error, the next thing to try is PyTorch's
    # newer dynamo-based exporter: torch.onnx.export(..., dynamo=True), which
    # has broader coverage for dynamic control flow but was still maturing as
    # of PyTorch 2.4. This is flagged rather than silently worked around
    # because getting it wrong would silently produce an incorrect graph.
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, pixel_values, image_grid_thw),
        str(output_path),
        input_names=["input_ids", "attention_mask", "pixel_values", "image_grid_thw"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "pixel_values": {0: "num_patches"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    print("[export_onnx] Validating exported graph with onnx.checker...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("[export_onnx] Graph is structurally valid.")


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    print(f"[export_onnx] Applying dynamic int8 quantization -> {int8_path}")
    # Dynamic quantization: weights are quantized ahead of time, activations
    # are quantized on the fly at inference time. Chosen over static
    # quantization because it needs no calibration dataset -- a reasonable
    # default for a first export. Static (calibrated) quantization would
    # likely recover more accuracy; revisit once you have real eval data
    # (see dataset/README.md) to calibrate against.
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print("[export_onnx] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="training/finetune.py checkpoint directory")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output", required=True, type=Path, help="Path for the final int8 .onnx file")
    parser.add_argument(
        "--prompt",
        default="What object is this? Answer with just the object's common name.",
        help="Must match the prompt used in training/finetune.py for a faithful export",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--skip_quantization",
        action="store_true",
        help="Export the fp32 ONNX graph only, skip int8 quantization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, processor = load_merged_model(args.checkpoint, args.base_model_name)
    trace_inputs = build_trace_inputs(processor, args.prompt)

    fp32_path = args.output.with_name(args.output.stem + "_fp32.onnx")
    export_to_onnx(model, trace_inputs, fp32_path, args.opset)

    if args.skip_quantization:
        print(f"[export_onnx] --skip_quantization set; fp32 graph is the final output: {fp32_path}")
        return

    quantize_int8(fp32_path, args.output)
    print(f"[export_onnx] Final int8 model: {args.output}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# TODO: what's still needed to run this on-device on the Quest 3 (Snapdragon
# XR2 Gen 2), roughly in the order you'd tackle them:
#
# 1. Split vision tower vs. language model decoder into two ONNX graphs (see
#    module docstring above) instead of one fused forward pass, and give the
#    decoder graph explicit past_key_values inputs/outputs so a runtime loop
#    can reuse KV cache across decode steps instead of recomputing attention
#    over the whole prefix every token.
#
# 2. Convert the final .onnx to ONNX Runtime Mobile's optimized .ort format:
#        python -m onnxruntime.tools.convert_onnx_models_to_ort scripts/exported/model_int8.onnx
#    This also lets you build a minimal ORT Mobile binary containing only the
#    ops your graph actually uses, which matters a lot for APK size.
#
# 3. Decide an execution provider for the XR2 Gen 2's Hexagon DSP/NPU (e.g.
#    the QNN execution provider) vs. falling back to the CPU/XNNPACK EP --
#    this needs benchmarking on real hardware, not something to guess at from
#    a desktop dev machine.
#
# 4. Package onnxruntime-mobile's Android AAR into the Unity project and
#    write a native Android plugin (Java/Kotlin, or C++ via JNI) that Unity's
#    C# can call into -- Unity has no first-party ONNX Runtime Mobile
#    bindings, so quest-app/Assets/Scripts/InferenceClient.cs's on-device
#    code path (currently just the WebSocket client for tethered mode) would
#    need a second implementation behind the same interface.
#
# 5. Validate accuracy end-to-end against the same eval set the fp32 model
#    was checked against (int8 dynamic quantization is usually gentle for
#    transformer weights, but "usually" isn't "always" -- actually measure
#    it before trusting on-device predictions).
# ---------------------------------------------------------------------------
