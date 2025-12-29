import argparse
import json
import logging
import os
import sys
import time
import uuid

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from PIL import Image

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


def load_pipeline(args):
    """Load Qwen-t2i pipeline with optional optimizations."""

    print(f"Loading Qwen-t2i model: {args.model}", file=sys.stderr, flush=True)

    # Parse device IDs
    device_ids = []
    if args.device_id and args.device_id.strip():
        if args.device_id == "auto":
            # Auto-detect all available GPUs
            if torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
                print(
                    f"✓ Auto-detected {len(device_ids)} GPU(s)",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            for dev_id in args.device_id.split(","):
                dev_id = dev_id.strip()
                if dev_id.startswith("cuda:"):
                    device_ids.append(int(dev_id.split(":")[1]))
                elif dev_id.isdigit():
                    device_ids.append(int(dev_id))
    elif args.device_id == "auto" or (not args.device_id and torch.cuda.is_available()):
        # Auto-detect all available GPUs
        device_ids = list(range(torch.cuda.device_count()))
        print(
            f"✓ Auto-detected {len(device_ids)} GPU(s): {device_ids}",
            file=sys.stderr,
            flush=True,
        )
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

    # Load Qwen-Image pipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
    )
    # SIMPLE MULTI-GPU: Use CPU offload for multiple GPUs
    if len(device_ids) > 1:
        print(
            f"✓ Using {len(device_ids)} GPUs with CPU offload",
            file=sys.stderr,
            flush=True,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=None)
    print(f"✓ Loaded Qwen-Image model: {args.model}", file=sys.stderr, flush=True)

    # Memory optimizations
    if args.low_vram == "true":
        # If already using CPU offload for multi-GPU, don't enable again
        if len(device_ids) <= 1:
            pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing("auto")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        print("✓ Low VRAM mode enabled", file=sys.stderr, flush=True)
    else:
        if torch.cuda.is_available() and len(device_ids) <= 1:
            pipe.to("cuda")
            print("✓ Full GPU mode", file=sys.stderr, flush=True)
        else:
            print("✓ CPU mode", file=sys.stderr, flush=True)

    # Try to enable xformers for memory efficiency
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers enabled", file=sys.stderr, flush=True)
    except Exception:
        print(
            "⚠ xformers not available, using default attention",
            file=sys.stderr,
            flush=True,
        )

    if args.lora_file:
        print(f"Loading LoRA: {args.lora_file}", file=sys.stderr, flush=True)
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
                if args.static_seed.lower() == "true":
                    current_seed = 42
                else:
                    current_seed = args.seed + batch_index
                generator = torch.Generator().manual_seed(current_seed)

                print(f"Generating: {prompt[:60]}...", file=sys.stderr)
                print(
                    f"  Size: {args.width}x{args.height}, Steps: {args.steps}, CFG Scale: {args.guidance_scale}, Seed: {current_seed}",
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
