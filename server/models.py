from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ALLOWED_AUDIO_FORMATS = frozenset({"wav", "mp3", "flac", "ogg", "m4a", "webm", "opus"})


# --- Client -> Server messages ---

class SynthesizeRequest(BaseModel):
    type: Literal["synthesize"] = "synthesize"
    request_id: str = Field(max_length=128)
    text: str
    language: str = "English"
    model: str | None = None
    voice_clone_prompt_id: str | None = None
    chunk_size: int | None = Field(default=None, ge=1, le=24)


class UploadRefAudioRequest(BaseModel):
    type: Literal["upload_ref_audio"] = "upload_ref_audio"
    request_id: str = Field(max_length=128)
    audio_base64: str
    ref_text: str = ""
    audio_format: str = "wav"
    model: str | None = None


# --- Server -> Client messages ---

class SynthesisStart(BaseModel):
    type: Literal["synthesis_start"] = "synthesis_start"
    request_id: str
    model: str
    sample_rate: int = 24000
    sample_width: int = 2
    channels: int = 1


class SynthesisEnd(BaseModel):
    type: Literal["synthesis_end"] = "synthesis_end"
    request_id: str
    model: str
    total_chunks: int
    total_samples: int
    duration_ms: float
    ttfa_ms: float
    rtf: float


class SynthesisCancelled(BaseModel):
    type: Literal["synthesis_cancelled"] = "synthesis_cancelled"
    request_id: str
    chunks_sent: int


class VoiceClonePromptReady(BaseModel):
    type: Literal["voice_clone_prompt_ready"] = "voice_clone_prompt_ready"
    request_id: str
    model: str
    prompt_id: str
    processing_ms: float


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    request_id: str | None = None
    code: str
    message: str


class PongMessage(BaseModel):
    type: Literal["pong"] = "pong"
