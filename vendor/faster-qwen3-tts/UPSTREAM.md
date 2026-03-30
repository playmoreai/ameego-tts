Upstream repository: https://github.com/andimarafioti/faster-qwen3-tts
Vendored from commit: 3ee34963f41bc393cacf0f026f5190b4715d78fd

Local modifications in this repository:
- Allow `FasterQwen3TTS.from_pretrained(..., **model_kwargs)` passthrough to qwen-tts/Transformers.
- Preserve `model_name` on the wrapper for health checks and reporting.
- Fall back to nested `speech_tokenizer` access and infer sample rate defensively.

Update process:
1. Replace the vendored source with a newer upstream snapshot.
2. Re-apply the small local patch in `faster_qwen3_tts/model.py`.
3. Re-run the standard bf16 smoke test before using the update in production.
