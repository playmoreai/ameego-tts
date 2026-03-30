from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from server.audio_utils import normalize_audio_to_wav


_DEFAULT_VOICE_STORE_ROOT = Path(tempfile.gettempdir()) / "ameego_tts_voices"


@dataclass(frozen=True)
class StoredVoice:
    voice_id: str
    audio_path: Path
    meta_path: Path
    metadata: dict


class VoiceStore:
    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir) if root_dir else _DEFAULT_VOICE_STORE_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_voice(
        self,
        *,
        audio_bytes: bytes,
        audio_format: str,
        display_name: str | None = None,
    ) -> StoredVoice:
        voice_id = str(uuid.uuid4())
        voice_dir = self.root_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=False)
        try:
            audio_path = voice_dir / "source.wav"
            duration_ms = normalize_audio_to_wav(
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                output_path=audio_path,
            )

            metadata = {
                "voice_id": voice_id,
                "audio_format": "wav",
                "display_name": display_name,
                "duration_ms": round(duration_ms, 1),
                "created_at": datetime.now(UTC).isoformat(),
            }
            meta_path = voice_dir / "meta.json"
            meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
            return StoredVoice(voice_id=voice_id, audio_path=audio_path, meta_path=meta_path, metadata=metadata)
        except Exception:
            shutil.rmtree(voice_dir, ignore_errors=True)
            raise

    def get_voice(self, voice_id: str) -> StoredVoice | None:
        normalized_voice_id = self.normalize_voice_id(voice_id)
        if normalized_voice_id is None:
            return None
        voice_dir = self.root_dir / normalized_voice_id
        audio_path = voice_dir / "source.wav"
        meta_path = voice_dir / "meta.json"
        if not audio_path.exists():
            return None
        metadata = {}
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {"voice_id": normalized_voice_id, "audio_format": "wav"}
        return StoredVoice(voice_id=normalized_voice_id, audio_path=audio_path, meta_path=meta_path, metadata=metadata)

    def normalize_voice_id(self, voice_id: str) -> str | None:
        try:
            return str(uuid.UUID(voice_id))
        except (ValueError, TypeError, AttributeError):
            return None
