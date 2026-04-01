"""Pre-download Qwen3-TTS models during Docker build."""

import argparse

MODEL_MAP = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Tokenizer is always required
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def resolve_model_id(size: str, model_id_0_6b: str, model_id_1_7b: str) -> str:
    if size == "0.6B":
        return model_id_0_6b
    if size == "1.7B":
        return model_id_1_7b
    raise ValueError(f"Unknown model size '{size}'. Available: {list(MODEL_MAP)}")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS models")
    parser.add_argument(
        "--model-sizes",
        default="0.6B",
        help="Comma-separated model sizes to download (default: 0.6B)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--model-id-0-6b",
        default=MODEL_MAP["0.6B"],
        help="HF repo ID for the 0.6B model",
    )
    parser.add_argument(
        "--model-id-1-7b",
        default=MODEL_MAP["1.7B"],
        help="HF repo ID for the 1.7B model",
    )
    parser.add_argument(
        "--voice-design-enabled",
        action="store_true",
        help="Download the VoiceDesign model in addition to Base models.",
    )
    parser.add_argument(
        "--voice-design-model-id",
        default=VOICE_DESIGN_MODEL,
        help="HF repo ID for the VoiceDesign model",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    sizes = [s.strip() for s in args.model_sizes.split(",") if s.strip()]

    print(f"Downloading tokenizer: {TOKENIZER_REPO}")
    snapshot_download(TOKENIZER_REPO, cache_dir=args.cache_dir)

    for size in sizes:
        if size not in MODEL_MAP:
            print(f"WARNING: Unknown model size '{size}', skipping. Available: {list(MODEL_MAP)}")
            continue
        model_id = resolve_model_id(size, args.model_id_0_6b, args.model_id_1_7b)
        print(f"Downloading model: {model_id}")
        snapshot_download(model_id, cache_dir=args.cache_dir)

    if args.voice_design_enabled:
        print(f"Downloading Voice Design model: {args.voice_design_model_id}")
        snapshot_download(args.voice_design_model_id, cache_dir=args.cache_dir)

    print("Download complete.")


if __name__ == "__main__":
    main()
