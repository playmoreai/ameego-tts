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
    host: str = "0.0.0.0"
    port: int = 8080
    chunk_size: int = 2
    max_connections: int = 4
    max_text_length: int = 5000
    clone_prompt_cache_size: int = 32

    @property
    def loaded_model_sizes(self) -> list[str]:
        sizes = [s.strip() for s in self.model_sizes.split(",") if s.strip()]
        for s in sizes:
            if s not in MODEL_ID_MAP:
                raise ValueError(f"Unknown model size: {s}. Available: {list(MODEL_ID_MAP)}")
        return sizes

    model_config = {"env_prefix": ""}


settings = Settings()
