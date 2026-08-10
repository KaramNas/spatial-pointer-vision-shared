"""
Wraps loading a fine-tuned Qwen2-VL checkpoint (base model + optional LoRA
adapter produced by training/finetune.py) and running single-image
"what object is this?" inference for the WebSocket server in main.py.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

logger = logging.getLogger("model_service")

DEFAULT_PROMPT = "What object is this? Answer with just the object's common name."

# Fallback vision-token bounds used only when no fine-tuned checkpoint is
# loaded (a checkpoint saved by training/finetune.py carries its own
# min_pixels/max_pixels in preprocessor_config.json, so train/serve stay in
# sync automatically -- these are just sane, VRAM-conscious defaults for
# "base model, no fine-tune" testing).
FALLBACK_MIN_PIXELS = 256 * 28 * 28
FALLBACK_MAX_PIXELS = 768 * 28 * 28


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference request arrives before/while the model is loading."""


class InvalidImageError(ValueError):
    """Raised when the received bytes can't be decoded as an image."""


class ModelService:
    """Loads a Qwen2-VL model (+ optional LoRA adapter) once and serves
    inference requests against it.

    Not safe for concurrent .identify() calls from multiple asyncio tasks at
    once -- the FastAPI server in main.py serializes access with a lock.
    """

    def __init__(
        self,
        checkpoint_path: str | None,
        base_model_name: str,
        device: str = "cuda",
        prompt: str = DEFAULT_PROMPT,
    ):
        self.checkpoint_path = checkpoint_path
        self.base_model_name = base_model_name
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.prompt = prompt
        self.model = None
        self.processor = None

        if device == "cuda" and self.device == "cpu":
            logger.warning("CUDA requested but not available; falling back to CPU (inference will be slow)")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Loads the base model (+ LoRA adapter if configured). Raises on
        failure -- callers decide whether that's fatal (see main.py's
        lifespan handler, which logs and keeps the server up so /health
        still reports model_loaded=False instead of crashing the process).
        """
        # Serve in the same 4-bit (QLoRA) precision training used, rather
        # than merging the adapter into a full bf16 model: a merged 7B model
        # in bf16 is ~14GB, which doesn't fit a 12GB card at all. Keeping the
        # base model 4-bit-quantized and running the LoRA adapter unmerged
        # (PEFT applies it as a forward-time addition) keeps both the 2B
        # default and the 7B fallback comfortably within a 12GB budget, at
        # the cost of a small amount of extra latency per request versus a
        # merged model -- an easy trade for a locally-tethered server.
        logger.info("Loading base model %s on %s", self.base_model_name, self.device)
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
        else:
            # bitsandbytes 4-bit requires CUDA; CPU fallback loads full fp32.
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_name, torch_dtype=torch.float32
            )

        processor_source = self.base_model_name
        processor_kwargs = {}

        if self.checkpoint_path:
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"MODEL_CHECKPOINT_PATH does not exist: {checkpoint}"
                )
            logger.info("Loading LoRA adapter from %s", checkpoint)
            model = PeftModel.from_pretrained(model, str(checkpoint))

            if (checkpoint / "preprocessor_config.json").exists():
                processor_source = str(checkpoint)
            else:
                logger.warning(
                    "Checkpoint has no saved processor config; falling back to "
                    "the base model's default image processing settings"
                )
        else:
            logger.warning(
                "No MODEL_CHECKPOINT_PATH set -- serving the un-fine-tuned base "
                "model. Useful for testing the server/wire protocol, but "
                "identification quality will be poor until you point this at "
                "a training/finetune.py checkpoint."
            )
            processor_kwargs = {"min_pixels": FALLBACK_MIN_PIXELS, "max_pixels": FALLBACK_MAX_PIXELS}

        model.eval()

        self.model = model
        self.processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
        logger.info("Model ready (checkpoint=%s)", self.checkpoint_path or "<none, base model only>")

    @torch.inference_mode()
    def identify(self, image_bytes: bytes, max_new_tokens: int = 16) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded yet")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # force decode now so corrupt data raises here, not later
            image = image.convert("RGB")
        except Exception as e:
            raise InvalidImageError(f"Could not decode image bytes: {e}") from e

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0, prompt_len:]
        label = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        confidence = self._mean_token_confidence(output.scores, generated_ids)

        return {"label": label, "confidence": confidence}

    @staticmethod
    def _mean_token_confidence(scores, generated_ids: torch.Tensor) -> float:
        """Approximates a confidence score as the mean softmax probability
        the model assigned to each token it actually greedily generated.
        This reflects how confident the model was in its own top choice at
        each decoding step -- it is NOT a calibrated probability that the
        label is correct, just a cheap, always-available proxy for it.
        """
        if len(scores) == 0:
            return 0.0
        token_probs = []
        for step_logits, token_id in zip(scores, generated_ids):
            step_probs = torch.softmax(step_logits[0], dim=-1)
            token_probs.append(step_probs[token_id].item())
        return float(sum(token_probs) / len(token_probs))
