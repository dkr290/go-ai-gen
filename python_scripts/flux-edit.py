#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch

# Core AI Libraries
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from PIL import Image
from transformers import BitsAndBytesConfig, T5EncoderModel


def save_image(image: Image.Image, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        image.save(f, format="PNG")


def load_pipeline(args):
    # Determine precision based on hardware
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- Loading Components ---", file=sys.stderr)

    # 1. OPTIMIZATION: Manually load T5-XXL in 4-bit (Saves ~7GB VRAM)
    # This will download the T5 weights from the 'model' repo if not cached.
    text_encoder_2 = None
    if args.low_vram.lower() == "true":
        print("✓ Quantizing T5-XXL Text Encoder to 4-bit...", file=sys.stderr)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            args.model,
            subfolder="text_encoder_2",
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
        )

    # 2. TRANSFORMER: Load your local GGUF file
    if args.gguf_file and os.path.exists(args.gguf_file):
        print(f"✓ Loading GGUF Transformer from: {args.gguf_file}", file=sys.stderr)
        transformer = FluxTransformer2DModel.from_single_file(
            args.gguf_file,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )
    else:
        transformer = None  # Let pipeline download default if no GGUF provided

    # 3. PIPELINE: Assemble everything
    # This will auto-download the CLIP encoder and VAE (approx 500MB total)
    pipe = FluxPipeline.from_pretrained(
        args.model,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch_dtype,
    )

    # 4. FINAL MEMORY OPTIMIZATIONS
    if args.low_vram.lower() == "true":
        # Crucial for 8GB-16GB cards
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
        print(
            "✓ Low VRAM optimizations enabled (CPU Offload + Tiling)", file=sys.stderr
        )
    else:
        pipe.to(device)

    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="black-forest-labs/FLUX.1-dev", help="HF model ID"
    )
    parser.add_argument("--gguf-file", required=True, help="Path to your .gguf file")
    parser.add_argument("--prompts", required=True, help="JSON list of prompts")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--low-vram", type=str, default="true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        pipe = load_pipeline(args)
        prompts_data = json.loads(args.prompts)

        for i, data in enumerate(prompts_data):
            prompt_text = data["prompt"]
            filename = data.get("filename", f"output_{i}.png")

            generator = torch.Generator().manual_seed(args.seed)

            print(f"Generating image {i+1}/{len(prompts_data)}...", file=sys.stderr)

            # CRITICAL: Set max_sequence_length=512 for long prompts
            output = pipe(
                prompt=prompt_text,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=512,
            ).images[0]

            save_path = os.path.join(args.output_dir, filename)
            save_image(output, save_path)
            print(f"✓ Saved: {save_path}", file=sys.stderr)

        print(json.dumps({"status": "success"}))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
