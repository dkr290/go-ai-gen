import argparse
import json
import os
import sys
import time

import torch
from diffusers import (
    AutoPipelineForText2Image,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)
from PIL import Image


def save_image(image: Image.Image, output_path: str) -> None:
    """Save image as PNG, matching Go's png.Encode behavior."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        image.save(f, format="PNG", optimize=False)


def load_gguf_pipeline(args):
    """Load Qwen-Image GGUF model using FluxPipeline with GGUF transformer."""
    print("Loading Qwen-Image GGUF model", file=sys.stderr, flush=True)

    start = time.time()

    # Set environment variables for progress bars
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "true"
    os.environ["DISABLE_TQDM"] = "true"

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    try:
        # For Flux models with GGUF support, we need to load the GGUF file
        # and use it with FluxPipeline

        # First, check if GGUF file exists locally
        if not os.path.exists(args.gguf_file):
            raise FileNotFoundError(f"GGUF file not found: {args.gguf_file}")

        print(f"✓ Found GGUF file: {args.gguf_file}", file=sys.stderr, flush=True)

        # Load the pipeline with GGUF transformer
        # Note: This requires a specific version of diffusers that supports GGUF
        # You might need to install: pip install diffusers[gguf]

        print("Loading GGUF transformer...", file=sys.stderr, flush=True)

        try:
            # Load the GGUF‑quantized transformer
            transformer = FluxTransformer2DModel.from_single_file(
                args.gguf_file,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
            )

            # Load the rest of the pipeline
            pipe = FluxPipeline.from_pretrained(
                args.model,
                transformer=transformer,
                torch_dtype=torch_dtype,
            )

        except Exception:
            # Fallback if GGUF loading fails
            # print(f"✗ Failed to load GGUF pipeline: {e}", file=sys.stderr, flush=True)

            pipe = AutoPipelineForText2Image.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
            )

            print("✓ Using default FULL model", file=sys.stderr, flush=True)

        # Move to device
        pipe = pipe.to(device)

        # Apply optimizations
        if args.low_vram == "true":
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            print("✓ Low VRAM mode enabled", file=sys.stderr, flush=True)

        print(f"✓ Pipeline loaded in {time.time() - start:.1f}s", file=sys.stderr)
        return pipe

    except Exception as e:
        print(f"✗ Failed to load GGUF pipeline: {e}", file=sys.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Image GGUF text-to-image generation"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen-Image-2512)",
    )
    parser.add_argument(
        "--gguf-file",
        required=True,
        help="Path to GGUF model file",
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
    parser.add_argument(
        "--prompts",
        required=True,
        help='JSON string of prompt data, e.g., \'[{"prompt": "text", "filename": "name.png"}]\'',
    )
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
        pipe = load_gguf_pipeline(args)

        # Parse the incoming JSON string for prompts data
        prompts = json.loads(args.prompts)
        all_results = []

        for i, prompt_data in enumerate(prompts):
            for batch_index in range(args.num_images):

                filename = prompt_data["filename"]
                pr = prompt_data["prompt"]
                output_path = os.path.join(args.output_dir, filename)
                if args.static_seed.lower() == "true":
                    current_seed = 42
                else:
                    current_seed = args.seed + batch_index

                # For Flux models, we need a generator
                generator = torch.Generator(device="cpu").manual_seed(current_seed)

                print(f"Generating: {pr[:60]}...", file=sys.stderr)
                print(
                    f" Size: {args.width}x{args.height}, Steps: {args.steps}",
                    f" CFG Scale: {args.guidance_scale}, Seed: {current_seed}",
                    file=sys.stderr,
                )
                gen_start = time.time()

                # Generate image using Flux pipeline
                # Note: Flux models might have different parameters

                result = pipe(
                    prompt=pr,
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
        import traceback

        error_details = traceback.format_exc()
        print(f"FULL ERROR TRACEBACK:\n{error_details}", file=sys.stderr, flush=True)
        print(
            json.dumps({"status": "error", "error": str(e), "traceback": error_details})
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
