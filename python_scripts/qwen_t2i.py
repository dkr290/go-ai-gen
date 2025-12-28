import argparse
import json
import os
import sys
import time
import uuid

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import logging as diffusers_logging
from PIL import Image

env_keys = [
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "DIFFUSERS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_TOKEN",
]

print("=== Hugging Face ENV ===")
for key in env_keys:
    print(f"{key}={os.environ.get(key)}")
print("************************")


def save_image(image: Image.Image, output_path: str) -> None:
    """Save image as PNG, matching Go's png.Encode behavior."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save as PNG with same approach as Go
    with open(output_path, "wb") as f:
        image.save(f, format="PNG", optimize=False)


def load_pipeline(args):
    """Load Qwen-t2i pipeline with optional optimizations."""

    print(f"Loading Qwen-t2i model: {args.model}", file=sys.stderr, flush=True)

    start = time.time()
    diffusers_logging.set_verbosity_error()

    # Load Qwen-Image-Edit pipeline
    # Note: Qwen-Image-Edit uses a different pipeline structure
    # We'll use AutoPipelineForImage2Image which should work with Qwen models
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.set_progress_bar_config(disable=None)
    print(f"✓ Loaded Qwen-Image-Edit model: {args.model}", file=sys.stderr, flush=True)

    # Memory optimizations
    if args.low_vram == "true":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing("auto")
        print("✓ Low VRAM mode enabled", file=sys.stderr)
    else:
        if torch.cuda.is_available():
            pipe.to("cuda")
            print("✓ Full GPU mode", file=sys.stderr)
        else:
            print("✓ CPU mode", file=sys.stderr)

    # Try to enable xformers for memory efficiency
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers enabled", file=sys.stderr)
    except Exception:
        print("⚠ xformers not available, using default attention", file=sys.stderr)

    if args.lora_file:
        print(f"Loading LoRA: {args.lora_file}", file=sys.stderr)
        try:
            # Get the directory containing the lora file
            lora_dir = os.path.dirname(args.lora_file)
            lora_filename = os.path.basename(args.lora_file)

            # Load from local directory
            pipe.load_lora_weights(
                lora_dir,
                weight_name=lora_filename,
                adapter_name=args.lora_adapter_name,
            )
            print("✓ LoRA loaded successfully", file=sys.stderr)
        except Exception as e:
            print(f"⚠ LoRA loading failed: {e}", file=sys.stderr)
            print("  Continuing without LoRA...", file=sys.stderr)

    print(f"✓ Pipeline loaded in {time.time() - start:.1f}s", file=sys.stderr)

    return pipe


def main():
    # Add diagnostic checks
    print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
    print(f"CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}", file=sys.stderr)
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}", file=sys.stderr)

    parser = argparse.ArgumentParser(description="Qwen-Image text-to-image generation")
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen-Image)",
    )
    parser.add_argument(
        "--negative-prompt", default="", help="Negative prompt for generation"
    )
    parser.add_argument("--width", type=int, default=1024, help="Output image width")
    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument(
        "--steps", type=int, default=40, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=1.0, help="Guidance scale (CFG)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for generated images"
    )
    parser.add_argument(
        "--num-images", default=1, type=int, help="Number of images per prompt"
    )

    # New argument to accept multiple prompts and their data for img2img
    parser.add_argument(
        "--prompts",
        required=True,
        help='JSON string of prompt data, e.g., \'["prompt1", "prompt2"]\'',
    )

    # Performance flag
    parser.add_argument(
        "--low-vram",
        type=str,
        default="false",
        help="Enable CPU offload and attention slicing for low VRAM GPUs",
    )
    parser.add_argument(
        "--lora-file",
        type=str,
        default="",
        help="Pass lora weights as lora file from the card",
    )
    parser.add_argument(
        "--lora-adapter-name",
        type=str,
        default="",
        help="Pass the name of lora adapter name",
    )

    args = parser.parse_args()

    try:
        pipe = load_pipeline(args)

        # Parse the incoming JSON string for prompts data
        prompts = json.loads(args.prompts)
        all_results = []

        for i, prompt in enumerate(prompts):
            for batch_index in range(args.num_images):

                filename = f"qwen_{uuid.uuid4().hex[:8]}.png"
                output_path = os.path.join(args.output_dir, filename)
                generator = torch.Generator().manual_seed(args.seed + batch_index)

                print(f"Generating: {prompt[:60]}...", file=sys.stderr)
                print(
                    f"  Size: {args.width}x{args.height}, Steps: {args.steps}, CFG Scale: {args.guidance_scale}, Seed: {args.seed + batch_index}",
                    file=sys.stderr,
                )
                gen_start = time.time()

                # Qwen-Image specific parameters
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    true_cfg_scale=args.guidance_scale,
                    generator=generator,
                    num_images_per_prompt=args.num_images,
                )

                image = result.images[0]

                # Save image
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
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
