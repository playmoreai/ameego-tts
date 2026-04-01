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
    app_profile: Literal["test", "api"] = "test"
    allowed_origins: str = ""
    model_sizes: str = "1.7B"
    default_model_size: str = "1.7B"
    initial_mode: Literal["voice_clone", "voice_design"] = "voice_clone"
    initial_clone_model_size: str = "1.7B"
    voice_design_enabled: bool = False
    voice_design_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    voice_storage_dir: str = ""
    model_id_0_6b: str = MODEL_ID_MAP["0.6B"]
    model_id_1_7b: str = MODEL_ID_MAP["1.7B"]
    model_device: str = "cuda"
    model_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "sdpa"
    cuda_graph_max_seq_len: int = 2048
    clone_0_6b_replicas: int = 2
    clone_1_7b_replicas: int = 1
    voice_design_replicas: int = 1
    chunk_size: int = 2
    max_connections: int = 8
    max_waiting_synth_requests: int = 1
    max_text_length: int = 5000
    clone_prompt_cache_size: int = 32

    @model_validator(mode="after")
    def validate_model_selection(self) -> "Settings":
        sizes = self.loaded_model_sizes
        if self.default_model_size not in sizes:
            raise ValueError(
                f"DEFAULT_MODEL_SIZE={self.default_model_size!r} must be one of MODEL_SIZES={sizes}"
            )
        if self.initial_clone_model_size not in sizes:
            raise ValueError(
                "INITIAL_CLONE_MODEL_SIZE="
                f"{self.initial_clone_model_size!r} must be one of MODEL_SIZES={sizes}"
            )
        if self.initial_mode == "voice_design" and not self.voice_design_enabled:
            raise ValueError("INITIAL_MODE='voice_design' requires VOICE_DESIGN_ENABLED=true")
        if self.clone_0_6b_replicas < 1 or self.clone_1_7b_replicas < 1 or self.voice_design_replicas < 1:
            raise ValueError("Replica counts must be >= 1")
        if self.max_connections < 1:
            raise ValueError("MAX_CONNECTIONS must be >= 1")
        if self.max_waiting_synth_requests < 0:
            raise ValueError("MAX_WAITING_SYNTH_REQUESTS must be >= 0")
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

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    model_config = {"env_prefix": ""}


settings = Settings()
