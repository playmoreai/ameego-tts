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
    model_size: str = "0.6B"
    host: str = "0.0.0.0"
    port: int = 8080
    chunk_size: int = 2
    max_connections: int = 4
    max_text_length: int = 5000
    clone_prompt_cache_size: int = 32

    @property
    def model_id(self) -> str:
        return MODEL_ID_MAP[self.model_size]

    model_config = {"env_prefix": ""}


settings = Settings()
