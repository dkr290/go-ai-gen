# filepath: scripts/qwen_t2i.py
#!/usr/bin/env python3
import json
import sys
import time
import traceback


def log(message, level="INFO"):
    """Log message in a format Go can parse."""
    log_entry = {"timestamp": time.time(), "level": level, "message": message}
    # Print as JSON line for Go to parse
    print(json.dumps(log_entry))
    sys.stdout.flush()  # Important for real-time streaming


def main():
    try:
        # Read JSON from stdin
        data = json.loads(sys.stdin.read())

        log(
            f"Starting Qwen image generation with {len(data.get('prompts', []))} prompts"
        )
        log(
            f"Parameters: {data.get('width')}x{data.get('height')}, {data.get('steps')} steps"
        )

        prompts = data.get("prompts", [])
        batch_size = data.get("batch_size", 1)

        # Simulate image generation
        for i, prompt in enumerate(prompts):
            log(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

            for j in range(batch_size):
                log(f"  Generating image {j+1}/{batch_size} for prompt {i+1}")
                time.sleep(0.5)  # Simulate work

                # Progress update
                progress = (
                    ((i * batch_size) + j + 1) / (len(prompts) * batch_size) * 100
                )
                log(f"Progress: {progress:.1f}%", "PROGRESS")

        # Final result
        result = {
            "status": "success",
            "generated_images": len(prompts) * batch_size,
            "image_paths": [
                f"output_{i}.png" for i in range(len(prompts) * batch_size)
            ],
            "time_taken": 2.5,  # Example
        }

        log("Image generation completed successfully")
        print(json.dumps({"result": result}))

    except Exception as e:
        log(f"Error in Python script: {str(e)}", "ERROR")
        log(traceback.format_exc(), "DEBUG")
        print(json.dumps({"error": str(e), "status": "failed"}))


if __name__ == "__main__":
    main()
