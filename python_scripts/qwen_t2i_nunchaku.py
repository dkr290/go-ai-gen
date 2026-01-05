import argparse
import json
import logging
import os
import sys
import time

import torch
from diffusers import QwenImagePipeline
from diffusers.utils import logging as diffusers_logging
from PIL import Image

# Nunchaku-specific imports
try:
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
    from nunchaku.utils import get_gpu_memory, get_precision
    NUNCHAKU_AVAILABLE = True
except ImportError:
    print("⚠ Nunchaku not installed. Install with: pip install nunchaku-qwen", file=sys.stderr)
    NUNCHAKU_AVAILABLE = False

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


def load_nunchaku_pipeline(args):
    """Load Nunchaku Qwen-Image pipeline with optimizations."""
    
    if not NUNCHAKU_AVAILABLE:
        raise ImportError(
            "Nunchaku is not installed. Please install with: pip install nunchaku-qwen"
        )

    print(f"Loading Nunchaku Qwen model: {args.model}", file=sys.stderr, flush=True)

    start = time.time()
    # Force simple progress bars for downloads
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "true"
    os.environ["DISABLE_TQDM"] = "true"

    logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    diffusers_logging.set_verbosity_error()
    
    # Determine device configuration
    if torch.cuda.is_available():
        device = "cuda:0"  # Nunchaku optimized for single GPU
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr, flush=True)
        
        # Get GPU memory and determine precision
        gpu_memory = get_gpu_memory()
        print(f"✓ GPU Memory: {gpu_memory / 1024:.2f} GB", file=sys.stderr, flush=True)
        
        # Get recommended precision based on GPU memory
        precision = get_precision(gpu_memory)
        print(f"✓ Recommended precision: {precision}", file=sys.stderr, flush=True)
    else:
        device = "cpu"
        precision = "fp32"
        print("⚠ No GPU detected, using CPU with fp32", file=sys.stderr, flush=True)

    # Load the Nunchaku transformer model
    print("✓ Loading Nunchaku quantized transformer...", file=sys.stderr, flush=True)
    
    # Load Nunchaku transformer with automatic precision detection
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )
    
    print("✓ Loading Qwen Image Pipeline...", file=sys.stderr, flush=True)
    
    # Load the full pipeline with Nunchaku transformer
    pipe = QwenImagePipeline.from_pretrained(
        args.model,
        transformer=transformer,
        torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )
    
    pipe = pipe.to(device)
    
    pipe.set_progress_bar_config(disable=None)

    print(f"✓ Loaded Nunchaku model: {args.model}", file=sys.stderr, flush=True)

    # Memory optimizations for low VRAM
    if args.low_vram == "true":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        print("✓ Low VRAM mode enabled (CPU offload + VAE optimizations)", file=sys.stderr, flush=True)
    else:
        print("✓ Full GPU mode", file=sys.stderr, flush=True)

    # Load LoRA if provided
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
            
            # Set LoRA scale if provided
            if args.lora_scale:
                try:
                    scale = float(args.lora_scale)
                    pipe.set_adapters([args.lora_adapter_name], adapter_weights=[scale])
                    print(f"✓ LoRA loaded with scale {scale}", file=sys.stderr, flush=True)
                except ValueError:
                    print(f"⚠ Invalid LoRA scale: {args.lora_scale}, using default 1.0", file=sys.stderr)
                    print("✓ LoRA loaded successfully", file=sys.stderr)
            else:
                print("✓ LoRA loaded successfully", file=sys.stderr)
        except Exception as e:
            print(f"⚠ LoRA loading failed: {e}", file=sys.stderr)
            print("  Continuing without LoRA...", file=sys.stderr)

    print(f"✓ Pipeline loaded in {time.time() - start:.1f}s", file=sys.stderr)
    
    # Disable safety checker if present
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        print("✓ Safety checker disabled", file=sys.stderr)
    
    return pipe


def main():
    # Add diagnostic checks
    print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
    print(f"CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}", file=sys.stderr)
        print(f"  Device 0: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    
    if NUNCHAKU_AVAILABLE:
        print("✓ Nunchaku is available", file=sys.stderr)
    else:
        print("✗ Nunchaku is NOT available - please install", file=sys.stderr)

    parser = argparse.ArgumentParser(description="Nunchaku Qwen-Image text-to-image generation")
    parser.add_argument(
        "--model",
        required=True,
        help="Nunchaku model ID (e.g., nunchaku-tech/nunchaku-qwen-image)",
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
        "--guidance-scale", type=float, default=4.0, help="Guidance scale (CFG)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for generated images"
    )
    parser.add_argument(
        "--num-images", default=1, type=int, help="Number of images per prompt"
    )

    # New argument to accept multiple prompts and their data
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
        help="Path to LoRA weights file",
    )
    parser.add_argument(
        "--lora-adapter-name",
        type=str,
        default="lora",
        help="Name of LoRA adapter",
    )
    parser.add_argument(
        "--lora-scale",
        type=str,
        default="1.0",
        help="LoRA scale/weight (0.0 to 2.0)",
    )
    parser.add_argument(
        "--static-seed",
        type=str,
        default="false",
        help="Always use seed 42 (Qwen default) if true",
    )

    args = parser.parse_args()

    try:
        pipe = load_nunchaku_pipeline(args)

        # Parse the incoming JSON string for prompts data
        prompts = json.loads(args.prompts)
        all_results = []

        for i, prompt in enumerate(prompts):
            for batch_index in range(args.num_images):

                filename = prompt["filename"]
                pr = prompt["prompt"]
                name_without_ext, ext = os.path.splitext(filename)
                batch_filename = f"{name_without_ext}_{batch_index}{ext}"
                output_path = os.path.join(args.output_dir, batch_filename)
                
                if args.static_seed.lower() == "true":
                    current_seed = 42
                else:
                    current_seed = args.seed + batch_index
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(current_seed)

                print(f"Generating: {pr[:60]}...", file=sys.stderr)
                print(
                    f" Size: {args.width}x{args.height}, Steps: {args.steps}",
                    f" CFG Scale: {args.guidance_scale}, Seed: {current_seed}",
                    file=sys.stderr,
                )
                gen_start = time.time()

                # Nunchaku Qwen-Image generation
                # Note: Nunchaku uses standard Qwen pipeline parameters
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
        import traceback

        error_details = traceback.format_exc()
        print(f"FULL ERROR TRACEBACK:\n{error_details}", file=sys.stderr, flush=True)
        print(
            json.dumps({"status": "error", "error": str(e), "traceback": error_details})
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
