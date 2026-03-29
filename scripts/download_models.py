"""Pre-download Qwen3-TTS models during Docker build."""

import argparse

from huggingface_hub import snapshot_download

MODEL_MAP = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# Tokenizer is always required
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS models")
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_MAP.keys()),
        default="0.6B",
        help="Model size to download (default: 0.6B)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    model_id = MODEL_MAP[args.model_size]
    print(f"Downloading tokenizer: {TOKENIZER_REPO}")
    snapshot_download(TOKENIZER_REPO, cache_dir=args.cache_dir)

    print(f"Downloading model: {model_id}")
    snapshot_download(model_id, cache_dir=args.cache_dir)

    print("Download complete.")


if __name__ == "__main__":
    main()
