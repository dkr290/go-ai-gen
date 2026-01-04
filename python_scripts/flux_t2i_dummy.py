#!/usr/bin/env python3
"""
Flux1-dev Text-to-Image Python Script

This script would be called by the Go backend to generate images using Flux1-dev.
"""

import argparse
import os
import sys

env_keys = [
    "HF_HOME",
    "DIFFUSERS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_TOKEN",
]
print("=== Hugging Face ENV ===", file=sys.stderr, flush=True)
for key in env_keys:
    print(f"{key}={os.environ.get(key)}", file=sys.stderr, flush=True)
print("************************", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--gguf-file", required=False, help="Path to GGUF file (optional)"
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-file", default="", help="safesensoes file")

    parser.add_argument(
        "--lora-adapter-name",
        type=str,
        default="",
        help="Pass the name of lora adapter name",
    )
    # New argument to accept multiple prompts and their data
    parser.add_argument(
        "--prompts",
        required=True,
        help='JSON string of prompt data, e.g., \'[{"prompt": "a dog", "filename": "dog.png"}]\'',
    )

    # Performance flag
    parser.add_argument(
        "--low-vram",
        type=str,
        default="false",
        help="Enable CPU offload and attention slicing for low VRAM GPUs",
    )
    parser.add_argument(
        "--num-images", default=1, type=int, help="Number of images per prompt"
    )

    parser.add_argument(
        "--static-seed",
        type=str,
        default="false",
        help="Always use seed 42 (Qwen default) if true",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="0",
        help="GPU device ID(s) to use (e.g., '0', '0,1', 'cuda:0,cuda:1')",
    )
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="bf16",
        help="Precision mode",
    )
    parser.add_argument(
        "--custom-encoders",
        type=str,
        default="false",
        help="Use custom text encoders",
    )
    parser.add_argument(
        "--encoder-repo",
        type=str,
        default="comfyanonymous/flux_text_encoders",
        help="HuggingFace repository for custom text encoders",
    )

    # THIS is what you were missing
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
