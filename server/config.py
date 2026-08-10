"""Server configuration, read from environment variables (or a .env file in
the working directory, if present) via pydantic-settings.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Base model to load. Must match what training/finetune.py was run with
    # if MODEL_CHECKPOINT_PATH is set (a LoRA adapter only applies cleanly to
    # the base model it was trained against).
    base_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Path to a checkpoint directory produced by training/finetune.py
    # (trainer.save_model() + processor.save_pretrained() output). If unset,
    # the server serves the raw base model -- see model_service.py.
    model_checkpoint_path: str | None = None

    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000
    max_new_tokens: int = 16


settings = Settings()
