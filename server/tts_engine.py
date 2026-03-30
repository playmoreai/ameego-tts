from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal, TypeVar

import numpy as np
import soundfile as sf

from server.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

RuntimeMode = Literal["voice_clone", "voice_design"]

_VENDORED_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "faster-qwen3-tts"
if _VENDORED_PACKAGE_ROOT.exists():
    vendored_path = str(_VENDORED_PACKAGE_ROOT)
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)

_REF_AUDIO_DIR = Path(tempfile.gettempdir()) / "ameego_tts_ref"
_REF_AUDIO_DIR.mkdir(exist_ok=True)
_DEFAULT_REF_PATH = _REF_AUDIO_DIR / "_default_ref.wav"


def _create_default_ref_audio(sample_rate: int = 24000) -> Path:
    """Generate a short speech-like reference WAV for warmup & fallback."""
    if _DEFAULT_REF_PATH.exists():
        return _DEFAULT_REF_PATH
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * 200 * t + 4 * np.sin(2 * np.pi * 3 * t))
    signal += 0.2 * np.sin(2 * np.pi * 500 * t + 3 * np.sin(2 * np.pi * 5 * t))
    fade = int(0.05 * sample_rate)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    sf.write(str(_DEFAULT_REF_PATH), signal, sample_rate)
    return _DEFAULT_REF_PATH


class RuntimeStateError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class TTSEngine:
    def __init__(
        self,
        model,
        sample_rate: int,
        config: Settings,
        engine_mode: RuntimeMode,
        model_key: str,
        model_id: str,
        gpu_lock: asyncio.Lock | None = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.config = config
        self.engine_mode = engine_mode
        self.model_key = model_key
        self.model_id = model_id
        self._lock = gpu_lock or asyncio.Lock()
        self._ref_audio_cache: OrderedDict[str, Path] = OrderedDict()
        self._voice_prompt_cache: OrderedDict[str, dict] = OrderedDict()
        self._default_ref_path = _create_default_ref_audio(sample_rate)

    @property
    def display_name(self) -> str:
        if self.engine_mode == "voice_design":
            return "Voice Design"
        return self.model_key

    @classmethod
    def from_model_id(
        cls,
        config: Settings,
        *,
        engine_mode: RuntimeMode,
        model_key: str,
        model_id: str,
        gpu_lock: asyncio.Lock | None = None,
    ) -> TTSEngine:
        from faster_qwen3_tts import FasterQwen3TTS

        logger.info(
            "Loading model %s (mode=%s, key=%s, device=%s, dtype=%s)...",
            model_id,
            engine_mode,
            model_key,
            config.model_device,
            config.model_dtype,
        )
        model = FasterQwen3TTS.from_pretrained(
            model_id,
            device=config.model_device,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
            max_seq_len=config.cuda_graph_max_seq_len,
        )
        sample_rate = model.sample_rate
        logger.info("Model %s loaded. Sample rate: %d", model_id, sample_rate)
        return cls(
            model,
            sample_rate,
            config,
            engine_mode=engine_mode,
            model_key=model_key,
            model_id=model_id,
            gpu_lock=gpu_lock,
        )

    def warm_up(self) -> None:
        logger.info("Warming up %s (%s)...", self.engine_mode, self.model_key)
        t0 = time.perf_counter()
        try:
            if self.engine_mode == "voice_design":
                for _ in self.model.generate_voice_design_streaming(
                    text="Hello, this is a warm up.",
                    instruct="Warm, clear narrator voice.",
                    language="English",
                    chunk_size=self.config.chunk_size,
                ):
                    pass
            else:
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
            logger.info("Warm-up %s complete in %.0fms", self.model_key, elapsed)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning(
                "Warm-up %s failed (%.0fms): %s. First request will capture lazily.",
                self.model_key,
                elapsed,
                e,
            )

    async def run_on_gpu(self, fn: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(None, fn, *args)

    def clear_voice_clone_cache(self) -> None:
        self._ref_audio_cache.clear()
        self._voice_prompt_cache.clear()

    def save_ref_audio(self, audio_bytes: bytes, audio_format: str) -> tuple[str, Path]:
        if self.engine_mode != "voice_clone":
            raise ValueError("Reference audio upload is only supported in voice clone mode.")

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
                        audio_format,
                        sr,
                        len(data) / sr,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to convert {audio_format} to WAV: {e}") from e
                finally:
                    tmp_path.unlink(missing_ok=True)

        self._ref_audio_cache[hash_key] = wav_path
        self._ref_audio_cache.move_to_end(hash_key)
        while len(self._ref_audio_cache) > self.config.clone_prompt_cache_size:
            self._ref_audio_cache.popitem(last=False)
        return hash_key, wav_path

    def precompute_voice_prompt(self, prompt_id: str) -> float:
        if self.engine_mode != "voice_clone":
            raise ValueError("Voice prompts are only supported in voice clone mode.")

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
        voice_clone_prompt = {
            "ref_code": [None],
            "ref_spk_embedding": [spk_emb],
            "x_vector_only_mode": [True],
            "icl_mode": [False],
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._voice_prompt_cache[prompt_id] = voice_clone_prompt
        self._voice_prompt_cache.move_to_end(prompt_id)
        while len(self._voice_prompt_cache) > self.config.clone_prompt_cache_size:
            self._voice_prompt_cache.popitem(last=False)

        logger.info(
            "Pre-computed speaker embedding %s in %.0fms (model=%s, ref=%s)",
            prompt_id,
            elapsed_ms,
            self.model_key,
            wav_path.name,
        )
        return elapsed_ms

    def get_voice_prompt(self, prompt_id: str) -> dict | None:
        result = self._voice_prompt_cache.get(prompt_id)
        if result is not None:
            self._voice_prompt_cache.move_to_end(prompt_id)
        return result

    def get_ref_audio_path(self, prompt_id: str) -> Path | None:
        path = self._ref_audio_cache.get(prompt_id)
        if path is not None:
            self._ref_audio_cache.move_to_end(prompt_id)
        return path

    def _stream_voice_clone(
        self,
        *,
        text: str,
        language: str,
        chunk_size: int,
        ref_audio_path: str | None = None,
        voice_clone_prompt: Any | None = None,
        instruct: str | None = None,
        cancel_event: threading.Event | None = None,
    ):
        t0 = time.perf_counter()
        kwargs = {
            "text": text,
            "language": language,
            "ref_text": "",
            "chunk_size": chunk_size,
            "xvec_only": True,
        }
        if instruct:
            kwargs["instruct"] = instruct

        if voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = voice_clone_prompt
            logger.info(
                "[%s] Starting clone synthesis (pre-computed prompt): text=%r, lang=%s, chunk=%d",
                self.model_key,
                text[:80],
                language,
                chunk_size,
            )
        else:
            kwargs["ref_audio"] = ref_audio_path or str(self._default_ref_path)
            logger.info(
                "[%s] Starting clone synthesis (ref=%s): text=%r, lang=%s, chunk=%d",
                self.model_key,
                kwargs["ref_audio"],
                text[:80],
                language,
                chunk_size,
            )

        for i, (audio_chunk, sr, _) in enumerate(
            self.model.generate_voice_clone_streaming(**kwargs)
        ):
            if cancel_event and cancel_event.is_set():
                logger.info("[%s] Clone synthesis cancelled at chunk %d", self.model_key, i)
                break

            if i == 0:
                ttfa = (time.perf_counter() - t0) * 1000
                logger.info(
                    "[%s] First clone chunk: shape=%s, range=[%.4f, %.4f], sr=%d, ttfa=%.0fms",
                    self.model_key,
                    audio_chunk.shape,
                    float(audio_chunk.min()),
                    float(audio_chunk.max()),
                    sr,
                    ttfa,
                )

            clipped = np.clip(audio_chunk, -1.0, 1.0)
            yield (clipped * 32767).astype(np.int16).tobytes()

    def _stream_voice_design(
        self,
        *,
        text: str,
        instruct: str,
        language: str,
        chunk_size: int,
        cancel_event: threading.Event | None = None,
    ):
        t0 = time.perf_counter()
        logger.info(
            "[voice_design] Starting voice design synthesis: text=%r, instruct=%r, lang=%s, chunk=%d",
            text[:80],
            instruct[:80],
            language,
            chunk_size,
        )
        for i, (audio_chunk, sr, _) in enumerate(
            self.model.generate_voice_design_streaming(
                text=text,
                instruct=instruct,
                language=language,
                chunk_size=chunk_size,
            )
        ):
            if cancel_event and cancel_event.is_set():
                logger.info("[voice_design] Synthesis cancelled at chunk %d", i)
                break

            if i == 0:
                ttfa = (time.perf_counter() - t0) * 1000
                logger.info(
                    "[voice_design] First chunk: shape=%s, range=[%.4f, %.4f], sr=%d, ttfa=%.0fms",
                    audio_chunk.shape,
                    float(audio_chunk.min()),
                    float(audio_chunk.max()),
                    sr,
                    ttfa,
                )

            clipped = np.clip(audio_chunk, -1.0, 1.0)
            yield (clipped * 32767).astype(np.int16).tobytes()

    async def stream_tts(
        self,
        *,
        mode: RuntimeMode,
        text: str,
        language: str,
        chunk_size: int | None = None,
        ref_audio_path: str | None = None,
        voice_clone_prompt: Any | None = None,
        instruct: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AsyncGenerator[bytes, None]:
        cs = chunk_size or self.config.chunk_size
        _cancel = cancel_event or threading.Event()

        async with self._lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

            def _run_sync():
                error = None
                try:
                    if mode == "voice_design":
                        if not instruct:
                            raise ValueError("Voice design synthesis requires an instruction.")
                        iterator = self._stream_voice_design(
                            text=text,
                            instruct=instruct,
                            language=language,
                            chunk_size=cs,
                            cancel_event=_cancel,
                        )
                    else:
                        iterator = self._stream_voice_clone(
                            text=text,
                            language=language,
                            chunk_size=cs,
                            ref_audio_path=ref_audio_path,
                            voice_clone_prompt=voice_clone_prompt,
                            instruct=instruct,
                            cancel_event=_cancel,
                        )

                    for pcm_bytes in iterator:
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


class TTSRuntime:
    def __init__(self, config: Settings):
        self.config = config
        self._gpu_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._engine: TTSEngine | None = None
        self._active_jobs = 0
        self.runtime_generation = 0
        self.runtime_status: Literal["loading", "ready", "switching", "error"] = "loading"
        self.active_mode: RuntimeMode = config.initial_mode
        self.active_clone_model_size: str = config.initial_clone_model_size
        self.active_model_id: str | None = None
        self.switch_target_mode: RuntimeMode | None = None
        self.switch_target_clone_model_size: str | None = None
        self.message: str | None = None

    @property
    def available_clone_models(self) -> list[str]:
        return self.config.loaded_model_sizes

    @property
    def default_clone_model_size(self) -> str:
        return self.config.default_model_size

    def normalize_clone_model_size(self, clone_model_size: str | None) -> str:
        resolved = clone_model_size or self.active_clone_model_size or self.config.default_model_size
        if resolved not in self.available_clone_models:
            raise RuntimeStateError(
                "INVALID_MODEL",
                f"Unknown clone model: {resolved}. Available: {self.available_clone_models}",
            )
        return resolved

    async def initialize(self) -> None:
        await self._load_into_runtime(self.active_mode, self.active_clone_model_size)

    async def _build_engine(self, target_mode: RuntimeMode, clone_model_size: str) -> TTSEngine:
        loop = asyncio.get_running_loop()
        model_key, model_id = self._resolve_target(target_mode, clone_model_size)

        def _build() -> TTSEngine:
            engine = TTSEngine.from_model_id(
                self.config,
                engine_mode=target_mode,
                model_key=model_key,
                model_id=model_id,
                gpu_lock=self._gpu_lock,
            )
            engine.warm_up()
            return engine

        return await loop.run_in_executor(None, _build)

    async def _release_engine(self, engine: TTSEngine | None) -> None:
        if engine is None:
            return
        loop = asyncio.get_running_loop()

        def _cleanup() -> None:
            engine.clear_voice_clone_cache()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug("Skipping torch.cuda.empty_cache()", exc_info=True)

        await loop.run_in_executor(None, _cleanup)

    def _resolve_target(self, target_mode: RuntimeMode, clone_model_size: str) -> tuple[str, str]:
        if target_mode == "voice_design":
            if not self.config.voice_design_enabled:
                raise RuntimeStateError(
                    "VOICE_DESIGN_DISABLED",
                    "Voice Design is not enabled on this deployment.",
                )
            return ("VoiceDesign", self.config.voice_design_model_id)
        return (clone_model_size, self.config.model_id_for_size(clone_model_size))

    async def _load_into_runtime(self, target_mode: RuntimeMode, clone_model_size: str) -> None:
        async with self._state_lock:
            self.runtime_status = "loading"
            self.message = None
            self.switch_target_mode = None
            self.switch_target_clone_model_size = None

        engine = await self._build_engine(target_mode, clone_model_size)

        async with self._state_lock:
            self._engine = engine
            self.active_mode = target_mode
            self.active_model_id = engine.model_id
            self.runtime_generation += 1
            if target_mode == "voice_clone":
                self.active_clone_model_size = clone_model_size
            self.runtime_status = "ready"
            self.message = None

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": self.runtime_status,
            "active_mode": self.active_mode,
            "active_model_id": self.active_model_id,
            "voice_design_enabled": self.config.voice_design_enabled,
            "active_clone_model_size": self.active_clone_model_size,
            "switch_target_mode": self.switch_target_mode,
            "switch_target_clone_model_size": self.switch_target_clone_model_size,
            "message": self.message,
            "available_models": self.available_clone_models,
            "available_clone_models": self.available_clone_models,
            "default_model": self.default_clone_model_size,
            "default_clone_model_size": self.default_clone_model_size,
            "runtime_generation": self.runtime_generation,
        }

    async def start_mode_switch(
        self,
        *,
        target_mode: RuntimeMode,
        clone_model_size: str | None = None,
    ) -> dict[str, Any]:
        target_clone_model_size = (
            self.normalize_clone_model_size(clone_model_size)
            if target_mode == "voice_clone"
            else self.active_clone_model_size
        )

        async with self._state_lock:
            if self.runtime_status == "switching":
                return {
                    "status": "busy",
                    "message": "Model switch already in progress.",
                }
            if self._active_jobs > 0:
                return {
                    "status": "busy",
                    "message": "The server is busy. Try again after the current task finishes.",
                }
            if target_mode == "voice_design" and not self.config.voice_design_enabled:
                return {
                    "status": "error",
                    "message": "Voice Design is not enabled on this deployment.",
                }
            if (
                self.runtime_status == "ready"
                and target_mode == self.active_mode
                and (
                    target_mode == "voice_design"
                    or target_clone_model_size == self.active_clone_model_size
                )
            ):
                return {
                    "status": "unchanged",
                    "mode": self.active_mode,
                    "model": self.active_clone_model_size,
                }

            self.runtime_status = "switching"
            self.switch_target_mode = target_mode
            self.switch_target_clone_model_size = (
                target_clone_model_size if target_mode == "voice_clone" else None
            )
            self.message = None
            asyncio.create_task(
                self._perform_switch(target_mode, target_clone_model_size)
            )

        return {
            "status": "switching",
            "from_mode": self.active_mode,
            "to_mode": target_mode,
            "model": target_clone_model_size if target_mode == "voice_clone" else None,
        }

    async def _perform_switch(self, target_mode: RuntimeMode, clone_model_size: str) -> None:
        old_engine: TTSEngine | None = None
        async with self._state_lock:
            old_engine = self._engine
            self._engine = None

        try:
            await self._release_engine(old_engine)
            new_engine = await self._build_engine(target_mode, clone_model_size)
        except Exception as e:
            logger.error("Failed to switch runtime to %s/%s: %s", target_mode, clone_model_size, e)
            async with self._state_lock:
                self.runtime_status = "error"
                self.message = str(e)
                self.switch_target_mode = None
                self.switch_target_clone_model_size = None
                self.active_model_id = None
            return

        async with self._state_lock:
            self._engine = new_engine
            self.active_mode = target_mode
            self.active_model_id = new_engine.model_id
            self.runtime_generation += 1
            if target_mode == "voice_clone":
                self.active_clone_model_size = clone_model_size
            self.runtime_status = "ready"
            self.message = None
            self.switch_target_mode = None
            self.switch_target_clone_model_size = None

    @asynccontextmanager
    async def acquire_request_engine(
        self,
        *,
        mode: RuntimeMode,
        clone_model_size: str | None = None,
    ):
        normalized_clone_model_size = (
            self.normalize_clone_model_size(clone_model_size)
            if mode == "voice_clone"
            else self.active_clone_model_size
        )
        async with self._state_lock:
            if mode == "voice_design" and not self.config.voice_design_enabled:
                raise RuntimeStateError(
                    "VOICE_DESIGN_DISABLED",
                    "Voice Design is not enabled on this deployment.",
                )
            if self.runtime_status == "switching":
                raise RuntimeStateError(
                    "MODE_SWITCH_IN_PROGRESS",
                    "The server is switching models. Try again when it is ready.",
                )
            if self.runtime_status != "ready" or self._engine is None:
                raise RuntimeStateError(
                    "MODE_NOT_READY",
                    "The requested mode is not ready yet.",
                )
            if mode != self.active_mode:
                raise RuntimeStateError(
                    "MODE_NOT_READY",
                    f"The server is ready for {self.active_mode}, not {mode}.",
                )
            if (
                mode == "voice_clone"
                and normalized_clone_model_size != self.active_clone_model_size
            ):
                raise RuntimeStateError(
                    "MODE_NOT_READY",
                    f"The clone model {normalized_clone_model_size} is not active yet.",
                )
            self._active_jobs += 1
            engine = self._engine

        try:
            yield engine
        finally:
            async with self._state_lock:
                self._active_jobs = max(0, self._active_jobs - 1)
