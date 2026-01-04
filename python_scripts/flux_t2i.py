#!/usr/bin/env python3
"""
Flux1-dev Text-to-Image Python Script

This script would be called by the Go backend to generate images using Flux1-dev.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.utils import logging as diffusers_logging
from PIL import Image
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import CLIPTextModel, T5EncoderModel

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


def save_image(image: Image.Image, output_path: str) -> None:
    """Save image as PNG, matching Go's png.Encode behavior."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save as PNG with same approach as Go
    with open(output_path, "wb") as f:
        image.save(f, format="PNG", optimize=False)


def debug_tokens(pipe, prompt):
    tokens = pipe.tokenizer(
        prompt, return_tensors="pt", padding=False, truncation=False
    ).input_ids[0]

    print(f"Token count: {len(tokens)}")
    print(f"Tokens (first 50): {tokens[:50]}")


def load_pipeline(args):

    print(f"Loading model: {args.model}", file=sys.stderr, flush=True)
    if args.gguf_file:
        print(f"Using GGUF: {args.gguf_file}", file=sys.stderr, flush=True)

    device_ids = []
    if args.device_id and args.device_id.strip():
        device_str = args.device_id.strip()

        if device_str == "auto":
            # Auto-detect all available GPUs
            if torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
                print(
                    f"✓ Auto-detected {len(device_ids)} GPU(s): {device_ids}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    "⚠ No CUDA devices available for auto-detection",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            # Parse comma-separated device IDs
            for dev_id in device_str.split(","):
                dev_id = dev_id.strip()
                try:
                    # Handle both "cuda:0" and "0" formats
                    if ":" in dev_id:
                        # Get the number after the colon
                        num_part = dev_id.split(":")[-1]
                        device_id = int(num_part)
                    else:
                        device_id = int(dev_id)

                    device_ids.append(device_id)
                except (ValueError, IndexError):
                    print(f"⚠ Invalid device ID: {dev_id}", file=sys.stderr, flush=True)

    print(f"DEBUG: Using device IDs: {device_ids}", file=sys.stderr, flush=True)

    start = time.time()

    # Force simple progress bars for downloads
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "true"
    os.environ["DISABLE_TQDM"] = "true"

    logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    diffusers_logging.set_verbosity_error()

    # Determine device configuration
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    print("--- Loading Components ---", file=sys.stderr)
    # Check if custom encoders are enabled
    use_custom_encoders = args.custom_encoders.lower() == "true"

    if use_custom_encoders:
        encoder_model_id = args.encoder_repo
        print(f"✓ Using custom text encoders from {encoder_model_id}", file=sys.stderr)
    else:
        encoder_model_id = args.model

    # 1. TEXT ENCODER: Load with quantization if requested
    text_encoder_2 = None

    if args.quant_mode == "fp8":
        print("✓ Loading T5 text encoder in 8-bit...", file=sys.stderr)
        quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            encoder_model_id,
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )

    else:
        # Load without quantization
        text_encoder_2 = T5EncoderModel.from_pretrained(
            encoder_model_id,
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
        )

    # Load text_encoder (CLIP-based) only if using custom encoders
    text_encoder = None
    if use_custom_encoders:
        print("✓ Loading custom CLIP text encoder...", file=sys.stderr)
        text_encoder = CLIPTextModel.from_pretrained(
            encoder_model_id,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )

    # 2. TRANSFORMER: Load GGUF or standard with specific precision
    transformer = None
    if args.gguf_file and os.path.exists(args.gguf_file):
        print(f"✓ Loading GGUF: {args.gguf_file}", file=sys.stderr)
        transformer = FluxTransformer2DModel.from_single_file(
            args.gguf_file,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )
    elif args.quant_mode == "fp8":
        print("✓ Loading Transformer in 8-bit...", file=sys.stderr)
        quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        transformer = FluxTransformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
        )
    else:
        # Load transformer without quantization
        transformer = FluxTransformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )

    # 3. PIPELINE ASSEMBLY
    pipeline_kwargs = {
        "text_encoder_2": text_encoder_2,
        "torch_dtype": torch_dtype,
    }

    # Add transformer if we loaded it separately
    if transformer is not None:
        pipeline_kwargs["transformer"] = transformer
    # Only add text_encoder if we loaded a custom one
    if text_encoder is not None:
        pipeline_kwargs["text_encoder"] = text_encoder
        print("✓ Using custom CLIP text encoder in pipeline", file=sys.stderr)

    # Use device_map for multi-GPU if specified
    if args.quant_mode == "fp8":
        pipeline_kwargs["device_map"] = "balanced"
    if len(device_ids) > 1:
        pipeline_kwargs["device_map"] = "balanced"
        print("✓ Using 'balanced' device map for multi-GPU", file=sys.stderr)

    pipe = FluxPipeline.from_pretrained(args.model, **pipeline_kwargs)

    # SIMPLE MULTI-GPU: Use CPU offload for multiple GPUs
    # Handle device placement if not using device_map
    if "device_map" not in pipeline_kwargs:

        if len(device_ids) > 1:
            print(
                f"✓ Using {len(device_ids)} GPUs with CPU offload",
                file=sys.stderr,
                flush=True,
            )
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
            try:
                # Flux uses transformer, not unet
                if hasattr(pipe, "transformer"):
                    pipe.transformer = torch.compile(
                        pipe.transformer, mode="reduce-overhead", fullgraph=True
                    )
                    print(
                        "✓ Transformer compiled with torch.compile()", file=sys.stderr
                    )
            except Exception as e:
                print(f"⚠ torch.compile failed: {e}", file=sys.stderr)

    pipe.set_progress_bar_config(disable=None)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers enabled", file=sys.stderr)
    except Exception:
        pass

    print(f"✓ Loaded Flux model: {args.model}", file=sys.stderr, flush=True)
    # Memory optimizations
    if args.low_vram.lower() == "true":
        if len(device_ids) <= 1 and "device_map" not in pipeline_kwargs:
            pipe.enable_sequential_cpu_offload()
        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        pipe.to(torch.float16)
        print("✓ Low VRAM mode enabled", file=sys.stderr)
    else:
        if torch.cuda.is_available() and len(device_ids) <= 1:
            print("✓ Full GPU mode", file=sys.stderr, flush=True)
        else:
            print("✓ CPU mode", file=sys.stderr, flush=True)

    # Load LoRA if specified
    if args.lora_file:
        print(f"Loading LoRA: {args.lora_file}", file=sys.stderr)
        try:
            # Get the directory containing the lora file
            lora_dir = os.path.dirname(args.lora_file)
            lora_filename = os.path.basename(args.lora_file)

            # Load from local directory instead of HuggingFace
            pipe.load_lora_weights(
                lora_dir,
                weight_name=lora_filename,
                adapter_name=args.lora_adapter_name,
            )
            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None
                print("✓ Safety checker disabled", file=sys.stderr)
        except Exception as e:
            print(f"⚠ LoRA loading failed: {e}", file=sys.stderr)
            print("  Continuing without LoRA...", file=sys.stderr)

    print(f"✓ Model loaded in {time.time() - start:.1f}s", file=sys.stderr)

    # Cache the pipeline
    return pipe


def main():
    # Add diagnostic checks here
    print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
    print(f"CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}", file=sys.stderr)
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}", file=sys.stderr)
            print(
                f"    Memory allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB",
                file=sys.stderr,
            )
            print(
                f"    Memory reserved: {torch.cuda.memory_reserved(i)/1e9:.2f} GB",
                file=sys.stderr,
            )
    else:
        print("WARNING: CUDA not available.", file=sys.stderr)
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

    args = parser.parse_args()
    try:
        pipe = load_pipeline(args)

        # Parse the incoming JSON string for prompts data
        prompts = json.loads(args.prompts)
        all_results = []

        for i, prompt in enumerate(prompts):
            for batch_index in range(args.num_images):
                pr = prompt["prompt"]
                filename = prompt["filename"]
                name_without_ext, ext = os.path.splitext(filename)
                batch_filename = f"{name_without_ext}_{batch_index}{ext}"
                output_path = os.path.join(args.output_dir, batch_filename)
                # Generate
                if args.static_seed.lower() == "true":
                    current_seed = 42
                else:
                    current_seed = args.seed + batch_index
                generator = torch.Generator().manual_seed(current_seed)

                print(f"Generating: {pr[:60]}...", file=sys.stderr)
                print(
                    f" Size: {args.width}x{args.height}, Steps: {args.steps}",
                    f" Guidence Scale: {args.guidance_scale}, Seed: {current_seed}",
                    file=sys.stderr,
                )
                debug_tokens(pipe, pr)
                gen_start = time.time()

                if args.model == "black-forest-labs/FLUX.1-schnell":
                    result = pipe(
                        prompt=pr,
                        negative_prompt=args.negative_prompt,
                        width=args.width,
                        height=args.height,
                        num_inference_steps=args.steps,
                        guidance_scale=0.0,
                        generator=generator,
                        max_sequence_length=256,
                    )

                else:
                    result = pipe(
                        prompt=pr,
                        negative_prompt=args.negative_prompt,
                        width=args.width,
                        height=args.height,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    )

                image = result.images[0]

                # Save image (matching Go's approach)
                save_image(image, output_path)

                elapsed = time.time() - gen_start
                print(f"✓ Saved to {output_path} in {elapsed:.1f}s", file=sys.stderr)

                all_results.append(
                    {
                        "status": "success",
                        "output": output_path,
                        "prompt_index": i,
                        "batch_index": batch_index,
                        "filename": filename,
                    }
                )

        print(json.dumps({"all_status": "success", "generations": all_results}))

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"FULL ERROR TRACEBACK:\n{error_details}", file=sys.stderr, flush=True)
        print(
            json.dumps({"status": "error", "error": str(e), "traceback": error_details})
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
