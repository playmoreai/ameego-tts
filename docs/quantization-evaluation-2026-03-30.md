# Quantization Evaluation

Date: `2026-03-30`

Scope:
- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- Hardware: GCP `g2-standard-4` with NVIDIA L4 24GB
- Mode: streaming
- Chunk size: `2`
- Warmup: `1`
- Iterations: `3`
- Seed: `1234`

## Result

`bf16` is the production choice.

| Mode | Avg TTFA | Avg Wall | Avg Audio | Avg RTF | Verdict |
|---|---:|---:|---:|---:|---|
| `bf16` | `155.1ms` | `2353.8ms` | `3786.7ms` | `0.622` | Best overall |
| `bnb4` | `219.2ms` | `2555.6ms` | `3840.0ms` | `0.666` | Slower TTFA, no clear upside on L4 |
| `bnb8` | `252.4ms` | `86806.3ms` | `110880.0ms` | `0.789` | Rejected |

`RTF` here is `wall_time / audio_duration`, so lower is better.

## Notes

- `bnb8` initially failed CUDA graph capture.
- Setting `llm_int8_threshold=0.0` allowed graph capture, but generation stability was still poor.
- In repeated runs, `bnb8` often failed to stop cleanly and expanded toward `max_new_tokens`, producing extremely long audio.
- `bnb4` worked, but TTFA and overall speed were both worse than `bf16` on the tested setup.

## Decision

- Keep the production path on vendored `faster-qwen3-tts` with `bf16`.
- Do not ship `bnb4` or `bnb8` support in the server, deploy flow, or Docker image.
