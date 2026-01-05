# Nunchaku Qwen Installation Guide

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support
- Go 1.21 or higher

## Step 1: Install Nunchaku

```bash
# Option 1: Install from PyPI (if available)
pip install nunchaku-qwen

# Option 2: Install from GitHub
pip install git+https://github.com/nunchaku-tech/nunchaku.git

# Option 3: Clone and install locally
git clone https://github.com/nunchaku-tech/nunchaku.git
cd nunchaku
pip install -e .
```

## Step 2: Verify Nunchaku Installation

```bash
# Test imports
python -c "from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel; print('✓ Nunchaku installed successfully')"

python -c "from nunchaku.utils import get_gpu_memory, get_precision; print('✓ Nunchaku utils available')"
```

## Step 3: Install Required Dependencies

```bash
# Install diffusers with Qwen support
pip install diffusers>=0.30.0

# Install transformers
pip install transformers>=4.40.0

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install pillow accelerate safetensors
```

## Step 4: Test the Implementation

### Quick Test Script

Create a file `test_nunchaku.py`:

```python
import torch
from diffusers import QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_memory = get_gpu_memory()
    print(f"GPU Memory: {gpu_memory / 1024:.2f} GB")
    
    precision = get_precision(gpu_memory)
    print(f"Recommended precision: {precision}")
    
    # Test loading transformer
    print("\nLoading Nunchaku transformer...")
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        "nunchaku-tech/nunchaku-qwen-image",
        subfolder="transformer",
        torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )
    print("✓ Transformer loaded successfully")
    
    # Test loading pipeline
    print("\nLoading Qwen Image Pipeline...")
    pipe = QwenImagePipeline.from_pretrained(
        "nunchaku-tech/nunchaku-qwen-image",
        transformer=transformer,
        torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )
    pipe = pipe.to("cuda:0")
    print("✓ Pipeline loaded successfully")
    
    # Test generation
    print("\nGenerating test image...")
    result = pipe(
        prompt="A beautiful sunset over mountains",
        num_inference_steps=20,
        guidance_scale=4.0,
        width=512,
        height=512,
    )
    
    result.images[0].save("test_nunchaku_output.png")
    print("✓ Image generated successfully: test_nunchaku_output.png")
else:
    print("⚠ No CUDA available. Nunchaku is optimized for GPU usage.")
```

Run the test:
```bash
python test_nunchaku.py
```

## Step 5: Build and Run the Go Application

```bash
# Build the application
go build -o go-ai-gen main.go

# Run the application
./go-ai-gen
# or
go run main.go
```

## Step 6: Access the Web Interface

1. Open your browser
2. Navigate to `http://localhost:8080`
3. Click on "Qwen-t2i Nunchaku" in the sidebar
4. Try generating an image!

## Troubleshooting

### Issue: "ImportError: No module named 'nunchaku'"

**Solution:**
```bash
pip install git+https://github.com/nunchaku-tech/nunchaku.git
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Enable "Low VRAM Mode" in the web interface
2. Reduce image resolution
3. Reduce batch size
4. Use a smaller number of inference steps

### Issue: "Model download fails"

**Solutions:**
1. Check your internet connection
2. Set HuggingFace token if needed:
   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxx"
   ```
3. Pre-download the model:
   ```bash
   huggingface-cli download nunchaku-tech/nunchaku-qwen-image
   ```

### Issue: "Slow generation speed"

**Possible causes:**
1. Not using GPU (check CUDA availability)
2. Low VRAM mode enabled (disables some optimizations)
3. High resolution images
4. Many inference steps

**Solutions:**
- Verify GPU is being used (check console output)
- Disable Low VRAM mode if you have enough memory
- Start with smaller resolutions (512x512)
- Use 20-40 steps (Nunchaku converges faster)

## Verify Installation Checklist

- [ ] Python 3.10+ installed
- [ ] CUDA and PyTorch installed
- [ ] Nunchaku package installed
- [ ] Can import `NunchakuQwenImageTransformer2DModel`
- [ ] Can import `get_gpu_memory` and `get_precision`
- [ ] Can load Qwen Image Pipeline
- [ ] Test generation works
- [ ] Go application builds successfully
- [ ] Web interface accessible

## Performance Tips

1. **First Run**: First generation will be slower due to model downloads
2. **GPU Memory**: More GPU memory = better performance
3. **Inference Steps**: 20-40 steps is optimal for Nunchaku
4. **Batch Size**: Use batch size 1-2 for best memory efficiency
5. **Image Size**: Start with 512x512 or 768x768, scale up gradually

## Additional Resources

- [Nunchaku Documentation](https://nunchaku.tech/docs/nunchaku/usage/qwen-image.html)
- [Nunchaku GitHub](https://github.com/nunchaku-tech/nunchaku)
- [Nunchaku Models on HuggingFace](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image)
- [Qwen Image Documentation](https://huggingface.co/Qwen/Qwen-Image)
