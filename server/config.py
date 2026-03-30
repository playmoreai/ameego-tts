from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


MODEL_ID_MAP = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

SUPPORTED_LANGUAGES = frozenset({
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
})


class Settings(BaseSettings):
    model_sizes: str = "0.6B,1.7B"
    default_model_size: str = "0.6B"
    model_id_0_6b: str = MODEL_ID_MAP["0.6B"]
    model_id_1_7b: str = MODEL_ID_MAP["1.7B"]
    model_device: str = "cuda"
    model_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "sdpa"
    cuda_graph_max_seq_len: int = 2048
    chunk_size: int = 2
    max_connections: int = 4
    max_text_length: int = 5000
    clone_prompt_cache_size: int = 32

    @model_validator(mode="after")
    def validate_model_selection(self) -> "Settings":
        sizes = self.loaded_model_sizes
        if self.default_model_size not in sizes:
            raise ValueError(
                f"DEFAULT_MODEL_SIZE={self.default_model_size!r} must be one of MODEL_SIZES={sizes}"
            )
        return self

    @property
    def loaded_model_sizes(self) -> list[str]:
        sizes = [s.strip() for s in self.model_sizes.split(",") if s.strip()]
        for s in sizes:
            if s not in MODEL_ID_MAP:
                raise ValueError(f"Unknown model size: {s}. Available: {list(MODEL_ID_MAP)}")
        return sizes

    def model_id_for_size(self, size: str) -> str:
        if size == "0.6B":
            return self.model_id_0_6b
        if size == "1.7B":
            return self.model_id_1_7b
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_ID_MAP)}")

    model_config = {"env_prefix": ""}


settings = Settings()
