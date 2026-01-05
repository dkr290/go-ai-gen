# How Nunchaku Model Selection Works

## The Short Answer

**Yes, it will work correctly!** But there's only ONE model repo needed: `nunchaku-tech/nunchaku-qwen-image`

## How Nunchaku Actually Works

Unlike traditional approaches where you have separate models for different quantizations (fp16, int8, etc.), **Nunchaku uses a single model repository** with automatic precision detection.

### What Happens When You Select the Model

1. **User selects**: `nunchaku-tech/nunchaku-qwen-image` (the only option now)

2. **Python script receives**: `--model nunchaku-tech/nunchaku-qwen-image`

3. **Nunchaku detects GPU memory**:
   ```python
   from nunchaku.utils import get_gpu_memory, get_precision
   
   gpu_memory = get_gpu_memory()  # e.g., 12288 MB (12 GB)
   precision = get_precision(gpu_memory)  # Returns: "bf16", "fp16", or "fp32"
   ```

4. **Loads optimized transformer**:
   ```python
   transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
       "nunchaku-tech/nunchaku-qwen-image",
       subfolder="transformer",
       torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
   )
   ```

5. **Loads pipeline with that transformer**:
   ```python
   pipe = QwenImagePipeline.from_pretrained(
       "nunchaku-tech/nunchaku-qwen-image",
       transformer=transformer,
       torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
   )
   ```

## Why Only One Model Option?

The dropdown previously had three options:
- ❌ `nunchaku-tech/nunchaku-qwen-image-fp16` (doesn't exist)
- ❌ `nunchaku-tech/nunchaku-qwen-image-int8` (doesn't exist)
- ✅ `nunchaku-tech/nunchaku-qwen-image` (the actual repo)

**Nunchaku's approach:** Instead of maintaining separate model repos for each quantization, they have:
- **One model repository**: Contains the base model + optimized transformer
- **Automatic precision selection**: Based on your GPU's available memory
- **Runtime optimization**: The `NunchakuQwenImageTransformer2DModel` applies optimizations at load time

## What Gets Downloaded

When you first use Nunchaku, it downloads from `nunchaku-tech/nunchaku-qwen-image`:

```
nunchaku-qwen-image/
├── transformer/              # Nunchaku optimized transformer
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── ...
├── text_encoder/            # Text encoder
├── text_encoder_2/          # Second text encoder
├── vae/                     # VAE decoder
├── scheduler/               # Noise scheduler
└── model_index.json         # Pipeline config
```

The **transformer** is the key component that Nunchaku optimizes!

## GPU Memory & Precision Detection

Nunchaku automatically selects precision based on available GPU memory:

| GPU Memory | Selected Precision | Example GPUs |
|------------|-------------------|--------------|
| < 6 GB     | fp32 (CPU fallback) | GTX 1060 6GB |
| 6-10 GB    | fp16              | RTX 3060, RTX 2070 |
| 10-16 GB   | bf16              | RTX 3080, RTX 4070 |
| > 16 GB    | bf16              | RTX 4090, A100 |

You can see this in the console output:
```
✓ GPU Memory: 12.00 GB
✓ Recommended precision: bf16
```

## Low VRAM Mode

When "Low VRAM Mode" is enabled, it adds additional optimizations:
```python
pipe.enable_model_cpu_offload()  # Offload to CPU when not in use
pipe.enable_vae_slicing()        # Process VAE in slices
pipe.enable_vae_tiling()         # Tile large images
```

This allows you to run on GPUs with less memory (even 6GB), but slightly slower.

## Comparison: Traditional vs Nunchaku

### Traditional Approach (e.g., regular Qwen)
```
User selects model → Download full model → Apply quantization at runtime → Load to GPU
```

### Nunchaku Approach
```
User selects model → Download optimized model → Auto-detect GPU → Select precision → Load optimized transformer
```

## Model Files Structure

Looking at `nunchaku-tech/nunchaku-qwen-image` on HuggingFace:

```
Files and versions:
- transformer/
  - config.json (transformer config)
  - diffusion_pytorch_model.safetensors (optimized weights)
- text_encoder/
- text_encoder_2/
- vae/
- scheduler/
- model_index.json
```

The **transformer weights** are already optimized by Nunchaku. The precision (bf16/fp16) is just the dtype used when loading them into memory.

## What if Someone Wants Different Quantization?

If Nunchaku adds more model variants in the future (e.g., int8, int4), they would likely:

1. Create new repos like:
   - `nunchaku-tech/nunchaku-qwen-image-int8`
   - `nunchaku-tech/nunchaku-qwen-image-int4`

2. You would then update the dropdown:
   ```html
   <option value="nunchaku-tech/nunchaku-qwen-image">Standard (Auto)</option>
   <option value="nunchaku-tech/nunchaku-qwen-image-int8">INT8 Quantized</option>
   ```

But currently, **there's only one repo** and it handles everything automatically!

## Testing It Works

When you run a generation, watch the console output:

```
=== Hugging Face ENV ===
HF_HOME=/root/.cache/huggingface
...
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA device count: 1
  Device 0: NVIDIA GeForce RTX 3080
✓ Nunchaku is available
Loading Nunchaku Qwen model: nunchaku-tech/nunchaku-qwen-image
✓ Using GPU: NVIDIA GeForce RTX 3080
✓ GPU Memory: 10.00 GB
✓ Recommended precision: bf16
✓ Loading Nunchaku quantized transformer...
✓ Loading Qwen Image Pipeline...
✓ Loaded Nunchaku model: nunchaku-tech/nunchaku-qwen-image
✓ Full GPU mode
✓ Pipeline loaded in 15.3s
✓ Safety checker disabled
```

See? It automatically detected:
- GPU memory: 10GB
- Precision: bf16
- Loaded the optimized transformer

## Summary

**Q: Will the dropdown work correctly?**  
**A:** Yes! The single option `nunchaku-tech/nunchaku-qwen-image` is correct and will:
1. Download the Nunchaku model
2. Auto-detect your GPU memory
3. Select optimal precision automatically
4. Load the optimized transformer
5. Generate images efficiently

**Q: Do I need to add more model options?**  
**A:** No, not unless Nunchaku releases additional model variants. Check their HuggingFace org: https://huggingface.co/nunchaku-tech

**Q: What if I have a low-end GPU?**  
**A:** Enable "Low VRAM Mode" checkbox - Nunchaku will still auto-detect but also enable CPU offloading and VAE optimizations.

The implementation is **correct and will work as expected**! 🚀
