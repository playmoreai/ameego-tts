from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, TypeVar

import numpy as np
import soundfile as sf

from server.config import MODEL_ID_MAP, Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Temp directory for reference audio files (deterministic names for cache reuse)
_REF_AUDIO_DIR = Path(tempfile.gettempdir()) / "ameego_tts_ref"
_REF_AUDIO_DIR.mkdir(exist_ok=True)

# Default reference audio path (generated at startup for warmup & fallback)
_DEFAULT_REF_PATH = _REF_AUDIO_DIR / "_default_ref.wav"


def _create_default_ref_audio(sample_rate: int = 24000) -> Path:
    """Generate a short speech-like reference WAV for warmup & default voice.

    Only writes the file once; subsequent calls are no-ops.
    """
    if _DEFAULT_REF_PATH.exists():
        return _DEFAULT_REF_PATH
    duration = 1.5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * 200 * t + 4 * np.sin(2 * np.pi * 3 * t))
    signal += 0.2 * np.sin(2 * np.pi * 500 * t + 3 * np.sin(2 * np.pi * 5 * t))
    fade = int(0.05 * sample_rate)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    sf.write(str(_DEFAULT_REF_PATH), signal, sample_rate)
    return _DEFAULT_REF_PATH


class TTSEngine:
    def __init__(
        self,
        model,
        sample_rate: int,
        config: Settings,
        model_size: str = "0.6B",
        gpu_lock: asyncio.Lock | None = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.config = config
        self.model_size = model_size
        self._lock = gpu_lock or asyncio.Lock()
        self._ref_audio_cache: OrderedDict[str, Path] = OrderedDict()
        self._voice_prompt_cache: OrderedDict[str, dict] = OrderedDict()
        self._default_ref_path = _create_default_ref_audio(sample_rate)

    @classmethod
    def from_config(
        cls,
        config: Settings,
        model_size: str,
        gpu_lock: asyncio.Lock | None = None,
    ) -> TTSEngine:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        model_id = MODEL_ID_MAP[model_size]
        logger.info("Loading model %s (%s)...", model_id, model_size)
        model = FasterQwen3TTS.from_pretrained(
            model_id,
            device="cuda",
            dtype=torch.bfloat16,
        )
        sample_rate = model.sample_rate
        logger.info("Model %s loaded. Sample rate: %d", model_size, sample_rate)
        return cls(model, sample_rate, config, model_size=model_size, gpu_lock=gpu_lock)

    def warm_up(self) -> None:
        """Warm up model to trigger CUDA graph capture."""
        logger.info("Warming up model %s (CUDA graph capture)...", self.model_size)
        t0 = time.perf_counter()
        try:
            for _ in self.model.generate_voice_clone_streaming(
                text="Hello, this is a warm up.",
                language="English",
                ref_audio=str(self._default_ref_path),
                ref_text="",
                chunk_size=self.config.chunk_size,
                xvec_only=True,
            ):
                pass
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Warm-up %s complete in %.0fms", self.model_size, elapsed)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning(
                "Warm-up %s failed (%.0fms): %s. CUDA graphs will capture on first request.",
                self.model_size, elapsed, e,
            )

    # --- GPU lock management ---

    async def run_on_gpu(self, fn: Callable[..., T], *args: Any) -> T:
        """Run a blocking function on GPU with lock serialization."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(None, fn, *args)

    # --- Reference audio management ---

    def save_ref_audio(self, audio_bytes: bytes, audio_format: str) -> tuple[str, Path]:
        """Save reference audio as WAV. Returns (hash_key, path).

        Raises ValueError if conversion fails.
        """
        hash_key = hashlib.sha256(audio_bytes).hexdigest()[:16]
        wav_path = _REF_AUDIO_DIR / f"{hash_key}.wav"

        if not wav_path.exists():
            if audio_format.lower() == "wav":
                wav_path.write_bytes(audio_bytes)
            else:
                tmp_path = _REF_AUDIO_DIR / f"{hash_key}_tmp.{audio_format}"
                try:
                    tmp_path.write_bytes(audio_bytes)
                    data, sr = sf.read(str(tmp_path))
                    sf.write(str(wav_path), data, sr)
                    logger.info(
                        "Converted ref audio %s->WAV: sr=%d, duration=%.1fs",
                        audio_format, sr, len(data) / sr,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert {audio_format} to WAV: {e}"
                    ) from e
                finally:
                    tmp_path.unlink(missing_ok=True)

        # LRU cache (don't delete files on eviction — shared across engines)
        self._ref_audio_cache[hash_key] = wav_path
        self._ref_audio_cache.move_to_end(hash_key)
        while len(self._ref_audio_cache) > self.config.clone_prompt_cache_size:
            self._ref_audio_cache.popitem(last=False)
        return hash_key, wav_path

    def precompute_voice_prompt(self, prompt_id: str) -> float:
        """Pre-compute speaker embedding (xvec_only) from cached ref audio.
        Returns processing time in ms. Must be called from a thread (blocking).
        """
        wav_path = self._ref_audio_cache.get(prompt_id)
        if wav_path is None:
            raise ValueError(f"Reference audio not found: {prompt_id}")

        t0 = time.perf_counter()

        prompt_items = self.model.model.create_voice_clone_prompt(
            ref_audio=str(wav_path),
            ref_text="",
            x_vector_only_mode=True,
        )
        spk_emb = prompt_items[0].ref_spk_embedding
        voice_clone_prompt = dict(
            ref_code=[None],
            ref_spk_embedding=[spk_emb],
            x_vector_only_mode=[True],
            icl_mode=[False],
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._voice_prompt_cache[prompt_id] = voice_clone_prompt
        self._voice_prompt_cache.move_to_end(prompt_id)
        while len(self._voice_prompt_cache) > self.config.clone_prompt_cache_size:
            self._voice_prompt_cache.popitem(last=False)

        logger.info(
            "Pre-computed speaker embedding %s in %.0fms (model=%s, ref=%s)",
            prompt_id, elapsed_ms, self.model_size, wav_path.name,
        )
        return elapsed_ms

    def get_voice_prompt(self, prompt_id: str) -> dict | None:
        """Get pre-computed voice clone prompt by ID."""
        result = self._voice_prompt_cache.get(prompt_id)
        if result is not None:
            self._voice_prompt_cache.move_to_end(prompt_id)
        return result

    def get_ref_audio_path(self, prompt_id: str) -> Path | None:
        """Get cached reference audio path by prompt ID."""
        path = self._ref_audio_cache.get(prompt_id)
        if path is not None:
            self._ref_audio_cache.move_to_end(prompt_id)
        return path

    # --- Streaming synthesis ---

    def _stream_synthesis(
        self,
        text: str,
        language: str,
        chunk_size: int,
        ref_audio_path: str | None = None,
        voice_clone_prompt: Any | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """Synchronous generator that yields PCM16 bytes."""
        t0 = time.perf_counter()

        kwargs = dict(
            text=text,
            language=language,
            ref_text="",
            chunk_size=chunk_size,
            xvec_only=True,
        )
        if voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = voice_clone_prompt
            logger.info(
                "[%s] Starting synthesis (pre-computed prompt): text=%r, lang=%s, chunk_size=%d",
                self.model_size, text[:80], language, chunk_size,
            )
        else:
            kwargs["ref_audio"] = ref_audio_path or str(self._default_ref_path)
            logger.info(
                "[%s] Starting synthesis (ref audio): text=%r, lang=%s, chunk_size=%d, ref=%s",
                self.model_size, text[:80], language, chunk_size, kwargs["ref_audio"],
            )

        for i, (audio_chunk, sr, _) in enumerate(
            self.model.generate_voice_clone_streaming(**kwargs)
        ):
            if cancel_event and cancel_event.is_set():
                logger.info("[%s] Synthesis cancelled at chunk %d", self.model_size, i)
                break

            if i == 0:
                ttfa = (time.perf_counter() - t0) * 1000
                logger.info(
                    "[%s] First chunk: shape=%s, range=[%.4f, %.4f], sr=%d, ttfa=%.0fms",
                    self.model_size, audio_chunk.shape,
                    float(audio_chunk.min()), float(audio_chunk.max()), sr, ttfa,
                )

            clipped = np.clip(audio_chunk, -1.0, 1.0)
            pcm_int16 = (clipped * 32767).astype(np.int16)
            yield pcm_int16.tobytes()

    async def stream_tts(
        self,
        text: str,
        language: str,
        chunk_size: int | None = None,
        ref_audio_path: str | None = None,
        voice_clone_prompt: Any | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Async generator yielding PCM16 bytes chunks. Serializes GPU access."""
        cs = chunk_size or self.config.chunk_size
        _cancel = cancel_event or threading.Event()

        async with self._lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

            def _run_sync():
                error = None
                try:
                    for pcm_bytes in self._stream_synthesis(
                        text, language, cs, ref_audio_path,
                        voice_clone_prompt=voice_clone_prompt,
                        cancel_event=_cancel,
                    ):
                        if _cancel.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, pcm_bytes)
                except Exception as e:
                    logger.error("Synthesis error: %s", e, exc_info=True)
                    error = e
                finally:
                    if error is not None:
                        loop.call_soon_threadsafe(queue.put_nowait, error)
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            task = loop.run_in_executor(None, _run_sync)

            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                _cancel.set()
                await task


class TTSEngineRegistry:
    """Manages multiple TTSEngine instances sharing a single GPU lock."""

    def __init__(self, engines: dict[str, TTSEngine], default_model: str):
        self._engines = engines
        self.default_model = default_model

    def get(self, model_size: str | None = None) -> TTSEngine:
        """Get engine by model size. None returns default."""
        key = model_size or self.default_model
        engine = self._engines.get(key)
        if engine is None:
            raise ValueError(
                f"Model not loaded: {key}. Available: {self.available_models}"
            )
        return engine

    @property
    def available_models(self) -> list[str]:
        return list(self._engines.keys())

    def items(self):
        return self._engines.items()

    @classmethod
    def from_config(cls, config: Settings) -> TTSEngineRegistry:
        gpu_lock = asyncio.Lock()
        engines: dict[str, TTSEngine] = {}
        for size in config.loaded_model_sizes:
            engines[size] = TTSEngine.from_config(config, size, gpu_lock=gpu_lock)
        return cls(engines, config.default_model_size)

    def warm_up_all(self) -> None:
        for engine in self._engines.values():
            engine.warm_up()
