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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal, TypeVar

import numpy as np
import soundfile as sf

from server.audio_utils import normalize_audio_to_wav
from server.config import Settings
from server.voice_store import VoiceStore

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
_PROMPT_ID_SEPARATOR = ":"


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
            try:
                duration_ms = normalize_audio_to_wav(
                    audio_bytes=audio_bytes,
                    audio_format=audio_format,
                    output_path=wav_path,
                )
                logger.info(
                    "Normalized ref audio %s->WAV: duration=%.1fms",
                    audio_format,
                    duration_ms,
                )
            except Exception as e:
                raise ValueError(f"Failed to normalize {audio_format} reference audio: {e}") from e

        self._ref_audio_cache[hash_key] = wav_path
        self._ref_audio_cache.move_to_end(hash_key)
        while len(self._ref_audio_cache) > self.config.clone_prompt_cache_size:
            self._ref_audio_cache.popitem(last=False)
        return hash_key, wav_path

    def cache_ref_audio_path(self, prompt_id: str, wav_path: Path) -> None:
        self._ref_audio_cache[prompt_id] = wav_path
        self._ref_audio_cache.move_to_end(prompt_id)
        while len(self._ref_audio_cache) > self.config.clone_prompt_cache_size:
            self._ref_audio_cache.popitem(last=False)

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

        for i, (audio_chunk, sr, _) in enumerate(self.model.generate_voice_clone_streaming(**kwargs)):
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


@dataclass
class EngineReplica:
    replica_id: str
    engine: TTSEngine
    active_jobs: int = 0

    @property
    def is_idle(self) -> bool:
        return self.active_jobs == 0


@dataclass
class EngineLease:
    replica: EngineReplica
    local_prompt_id: str | None = None

    @property
    def engine(self) -> TTSEngine:
        return self.replica.engine


@dataclass
class RuntimePool:
    mode: RuntimeMode
    clone_model_size: str | None
    model_key: str
    model_id: str
    replicas: list[EngineReplica]

    @property
    def total_replicas(self) -> int:
        return len(self.replicas)

    @property
    def busy_replicas(self) -> int:
        return sum(1 for replica in self.replicas if replica.active_jobs > 0)

    @property
    def available_capacity(self) -> int:
        return self.total_replicas - self.busy_replicas

    def get_replica(self, replica_id: str) -> EngineReplica | None:
        for replica in self.replicas:
            if replica.replica_id == replica_id:
                return replica
        return None

    def first_idle_replica(self) -> EngineReplica | None:
        for replica in self.replicas:
            if replica.is_idle:
                return replica
        return None

    def least_busy_replica(self) -> EngineReplica | None:
        if not self.replicas:
            return None
        return min(self.replicas, key=lambda replica: (replica.active_jobs, replica.replica_id))

    def descriptor(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "clone_model_size": self.clone_model_size,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "replica_count": self.total_replicas,
        }


class TTSRuntime:
    def __init__(self, config: Settings):
        self.config = config
        self.voice_store = VoiceStore(config.voice_storage_dir or None)
        self._state_lock = asyncio.Lock()
        self._state_changed = asyncio.Condition(self._state_lock)
        self._pool: RuntimePool | None = None
        self._waiting_synth_requests = 0
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

    def replica_count_for(self, target_mode: RuntimeMode, clone_model_size: str) -> int:
        if target_mode == "voice_design":
            return self.config.voice_design_replicas
        if clone_model_size == "0.6B":
            return self.config.clone_0_6b_replicas
        if clone_model_size == "1.7B":
            return self.config.clone_1_7b_replicas
        raise RuntimeStateError("INVALID_MODEL", f"Unknown clone model: {clone_model_size}")

    def encode_prompt_id(
        self,
        replica_id: str,
        local_prompt_id: str,
        runtime_generation: int | None = None,
    ) -> str:
        generation = self.runtime_generation if runtime_generation is None else runtime_generation
        return f"g{generation}{_PROMPT_ID_SEPARATOR}{replica_id}{_PROMPT_ID_SEPARATOR}{local_prompt_id}"

    def decode_prompt_id(self, prompt_id: str) -> tuple[int, str, str]:
        generation_token, separator, remainder = prompt_id.partition(_PROMPT_ID_SEPARATOR)
        replica_id, separator2, local_prompt_id = remainder.partition(_PROMPT_ID_SEPARATOR)
        if (
            not separator
            or not separator2
            or not generation_token.startswith("g")
            or not replica_id
            or not local_prompt_id
        ):
            raise RuntimeStateError("PROMPT_NOT_FOUND", f"Voice clone prompt not found: {prompt_id}")
        try:
            generation = int(generation_token[1:])
        except ValueError as exc:
            raise RuntimeStateError("PROMPT_NOT_FOUND", f"Voice clone prompt not found: {prompt_id}") from exc
        return generation, replica_id, local_prompt_id

    async def ensure_voice_prompt_for_replica(
        self,
        *,
        replica: EngineReplica,
        voice_id: str,
    ) -> tuple[str, dict]:
        stored_voice = self.voice_store.get_voice(voice_id)
        if stored_voice is None:
            raise RuntimeStateError("VOICE_NOT_FOUND", f"Voice not found: {voice_id}")

        local_prompt_id = f"voice:{voice_id}"
        engine = replica.engine
        voice_prompt = engine.get_voice_prompt(local_prompt_id)
        if voice_prompt is not None:
            return local_prompt_id, voice_prompt

        engine.cache_ref_audio_path(local_prompt_id, stored_voice.audio_path)
        await engine.run_on_gpu(engine.precompute_voice_prompt, local_prompt_id)
        voice_prompt = engine.get_voice_prompt(local_prompt_id)
        if voice_prompt is None:
            raise RuntimeStateError("VOICE_ASSET_ERROR", f"Failed to prepare voice: {voice_id}")
        return local_prompt_id, voice_prompt

    async def initialize(self) -> None:
        await self._load_into_runtime(self.active_mode, self.active_clone_model_size)

    async def shutdown(self) -> None:
        async with self._state_lock:
            pool = self._pool
            self._pool = None
            self._waiting_synth_requests = 0
            self.runtime_status = "loading"
            self.message = None
            self.switch_target_mode = None
            self.switch_target_clone_model_size = None
            self.active_model_id = None
            self._state_changed.notify_all()
        await self._release_pool(pool)

    async def _build_engine(
        self,
        target_mode: RuntimeMode,
        model_key: str,
        model_id: str,
    ) -> TTSEngine:
        loop = asyncio.get_running_loop()

        def _build() -> TTSEngine:
            engine = TTSEngine.from_model_id(
                self.config,
                engine_mode=target_mode,
                model_key=model_key,
                model_id=model_id,
            )
            engine.warm_up()
            return engine

        return await loop.run_in_executor(None, _build)

    async def _build_pool(self, target_mode: RuntimeMode, clone_model_size: str) -> RuntimePool:
        model_key, model_id = self._resolve_target(target_mode, clone_model_size)
        replica_count = self.replica_count_for(target_mode, clone_model_size)
        replicas: list[EngineReplica] = []
        try:
            for index in range(replica_count):
                engine = await self._build_engine(target_mode, model_key, model_id)
                replica_label = "vd" if target_mode == "voice_design" else clone_model_size.lower().replace(".", "")
                replica_id = f"{replica_label}-r{index + 1}"
                replicas.append(EngineReplica(replica_id=replica_id, engine=engine))
        except Exception:
            if replicas:
                await self._release_pool(
                    RuntimePool(
                        mode=target_mode,
                        clone_model_size=clone_model_size if target_mode == "voice_clone" else None,
                        model_key=model_key,
                        model_id=model_id,
                        replicas=replicas,
                    )
                )
            raise

        return RuntimePool(
            mode=target_mode,
            clone_model_size=clone_model_size if target_mode == "voice_clone" else None,
            model_key=model_key,
            model_id=model_id,
            replicas=replicas,
        )

    async def _release_pool(self, pool: RuntimePool | None) -> None:
        if pool is None:
            return
        loop = asyncio.get_running_loop()

        def _cleanup() -> None:
            for replica in pool.replicas:
                engine = replica.engine
                engine.clear_voice_clone_cache()
                engine.model = None
                replica.active_jobs = 0
            pool.replicas.clear()
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
            self._waiting_synth_requests = 0
            self.message = None
            self.switch_target_mode = None
            self.switch_target_clone_model_size = None
            self._state_changed.notify_all()

        pool = await self._build_pool(target_mode, clone_model_size)

        await self._activate_pool(pool, target_mode=target_mode, clone_model_size=clone_model_size)

    async def _activate_pool(
        self,
        pool: RuntimePool,
        *,
        target_mode: RuntimeMode,
        clone_model_size: str,
    ) -> None:
        async with self._state_lock:
            self._pool = pool
            self.active_mode = target_mode
            self.active_model_id = pool.model_id
            self.runtime_generation += 1
            self._waiting_synth_requests = 0
            if target_mode == "voice_clone":
                self.active_clone_model_size = clone_model_size
            self.runtime_status = "ready"
            self.message = None
            self.switch_target_mode = None
            self.switch_target_clone_model_size = None
            self._state_changed.notify_all()

    def health_payload(self) -> dict[str, Any]:
        pool = self._pool
        busy_replicas = pool.busy_replicas if pool else 0
        active_replicas = pool.total_replicas if pool else 0
        available_capacity = pool.available_capacity if pool else 0
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
            "active_profile": self.config.app_profile,
            "mode_switch_enabled": self.config.app_profile == "test",
            "active_replica_count": active_replicas,
            "busy_replica_count": busy_replicas,
            "available_synth_capacity": available_capacity,
            "waiting_synth_requests": self._waiting_synth_requests,
            "max_waiting_synth_requests": self.config.max_waiting_synth_requests,
            "current_pool": pool.descriptor() if pool else None,
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
            pool = self._pool
            if self.runtime_status == "switching":
                return {
                    "status": "busy",
                    "message": "Model switch already in progress.",
                }
            if pool and (pool.busy_replicas > 0 or self._waiting_synth_requests > 0):
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
            self._state_changed.notify_all()
            asyncio.create_task(self._perform_switch(target_mode, target_clone_model_size))

        return {
            "status": "switching",
            "from_mode": self.active_mode,
            "to_mode": target_mode,
            "model": target_clone_model_size if target_mode == "voice_clone" else None,
        }

    async def _perform_switch(self, target_mode: RuntimeMode, clone_model_size: str) -> None:
        old_pool: RuntimePool | None = None
        async with self._state_lock:
            old_pool = self._pool
            self._pool = None

        try:
            await self._release_pool(old_pool)
            new_pool = await self._build_pool(target_mode, clone_model_size)
        except Exception as e:
            logger.error("Failed to switch runtime to %s/%s: %s", target_mode, clone_model_size, e)
            async with self._state_lock:
                self.runtime_status = "error"
                self._waiting_synth_requests = 0
                self.message = str(e)
                self.switch_target_mode = None
                self.switch_target_clone_model_size = None
                self.active_model_id = None
                self._state_changed.notify_all()
            return

        await self._activate_pool(
            new_pool,
            target_mode=target_mode,
            clone_model_size=clone_model_size,
        )

    @asynccontextmanager
    async def acquire_request_engine(
        self,
        *,
        mode: RuntimeMode,
        clone_model_size: str | None = None,
        prompt_id: str | None = None,
        purpose: Literal["synthesize", "precompute"] = "synthesize",
    ):
        normalized_clone_model_size = (
            self.normalize_clone_model_size(clone_model_size)
            if mode == "voice_clone"
            else self.active_clone_model_size
        )
        queued = False
        acquired = False
        async with self._state_changed:
            try:
                while True:
                    pool = self._pool
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
                    if self.runtime_status != "ready" or pool is None:
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

                    local_prompt_id = None
                    if prompt_id:
                        generation, replica_id, local_prompt_id = self.decode_prompt_id(prompt_id)
                        if generation != self.runtime_generation:
                            raise RuntimeStateError(
                                "PROMPT_NOT_FOUND",
                                f"Voice clone prompt not found: {prompt_id}",
                            )
                        replica = pool.get_replica(replica_id)
                        if replica is None:
                            raise RuntimeStateError(
                                "PROMPT_NOT_FOUND",
                                f"Voice clone prompt not found: {prompt_id}",
                            )
                        if replica.active_jobs == 0:
                            replica.active_jobs += 1
                            acquired = True
                            break
                    else:
                        replica = pool.first_idle_replica()
                        if replica is not None:
                            replica.active_jobs += 1
                            acquired = True
                            break

                    action = "synthesis" if purpose == "synthesize" else "voice prompt preparation"
                    if purpose != "synthesize":
                        raise RuntimeStateError(
                            "SERVER_BUSY",
                            f"No idle capacity is available for {action}. Try again shortly.",
                        )
                    if not queued:
                        if self._waiting_synth_requests >= self.config.max_waiting_synth_requests:
                            raise RuntimeStateError(
                                "SERVER_BUSY",
                                f"No idle capacity is available for {action}. Try again shortly.",
                            )
                        self._waiting_synth_requests += 1
                        queued = True
                    await self._state_changed.wait()
            finally:
                if queued and not acquired:
                    self._waiting_synth_requests = max(0, self._waiting_synth_requests - 1)
                    self._state_changed.notify_all()

            if queued:
                self._waiting_synth_requests = max(0, self._waiting_synth_requests - 1)
                self._state_changed.notify_all()

        try:
            yield EngineLease(replica=replica, local_prompt_id=local_prompt_id)
        finally:
            async with self._state_changed:
                replica.active_jobs = max(0, replica.active_jobs - 1)
                self._state_changed.notify_all()
