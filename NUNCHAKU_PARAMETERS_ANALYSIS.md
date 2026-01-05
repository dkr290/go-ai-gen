# Nunchaku Parameters Analysis

## Parameters Being Passed from Go Handler

### ✅ **NEEDED - Core Generation Parameters**
These are essential for image generation:

| Parameter | Usage | Nunchaku Support |
|-----------|-------|------------------|
| `--model` | Model repo path | ✅ Required |
| `--width` | Image width | ✅ Standard param |
| `--height` | Image height | ✅ Standard param |
| `--steps` | Inference steps | ✅ Standard param |
| `--guidance-scale` | CFG scale | ✅ Standard param |
| `--seed` | Random seed | ✅ Standard param |
| `--output-dir` | Save location | ✅ Required |
| `--num-images` | Batch size | ✅ Required |
| `--prompts` | JSON prompts | ✅ Required |

### ⚠️ **MAYBE NEEDED - Optional Features**

| Parameter | Usage | Nunchaku Support | Keep? |
|-----------|-------|------------------|-------|
| `--negative-prompt` | Negative guidance | ✅ QwenImagePipeline supports | **YES** |
| `--static-seed` | Use seed 42 always | ✅ Custom logic | **YES** |
| `--lora-file` | LoRA weights path | ✅ Diffusers LoRA support | **YES** |
| `--lora-adapter-name` | LoRA adapter name | ✅ Diffusers LoRA support | **YES** |
| `--lora-scale` | LoRA influence | ✅ Diffusers LoRA support | **YES** |

### 🤔 **QUESTIONABLE - Memory Optimizations**

| Parameter | Usage | Nunchaku Compatibility | Recommendation |
|-----------|-------|------------------------|----------------|
| `--low-vram` | Enable CPU offload + VAE optimizations | ⚠️ May conflict with Nunchaku's auto-optimization | **KEEP but simplify** |

## The Low VRAM Question

### What Low VRAM Currently Does:
```python
if args.low_vram == "true":
    pipe.enable_model_cpu_offload()  # Offload to CPU
    pipe.enable_vae_slicing()        # Slice VAE processing
    pipe.enable_vae_tiling()         # Tile large images
```

### Nunchaku's Auto-Optimization:
```python
gpu_memory = get_gpu_memory()      # Auto-detects GPU memory
precision = get_precision(gpu_memory)  # Auto-selects precision
```

### The Conflict:

**Nunchaku already optimizes** based on GPU memory:
- 6-8 GB GPU → Uses fp16, lighter precision
- 8-12 GB GPU → Uses bf16, optimal precision
- 12+ GB GPU → Uses bf16, full speed

**Low VRAM mode adds:**
- CPU offloading (moves parts to CPU)
- VAE slicing (reduces VRAM usage)
- VAE tiling (for large images)

### Should We Keep Low VRAM?

**YES, but simplified:**

#### Scenario 1: GPU with < 6GB VRAM
- User enables Low VRAM mode
- Nunchaku auto-detects low memory
- Low VRAM mode adds extra optimizations
- **Result**: Can run on 4-6GB GPUs

#### Scenario 2: GPU with 8GB+ VRAM
- User keeps Low VRAM disabled
- Nunchaku auto-detects sufficient memory
- Uses full GPU power
- **Result**: Faster generation

#### Scenario 3: User has 8GB but wants to run other apps
- User enables Low VRAM mode
- Nunchaku uses 8GB-optimized settings
- Low VRAM mode reduces memory footprint
- **Result**: Leaves RAM for other tasks

### Recommendation: **Keep Low VRAM but with Clear Messaging**

## Parameters We DON'T Need (from regular Qwen)

These were in the original `qwen_t2i.py` but **NOT needed** for Nunchaku:

| Parameter | Why Not Needed |
|-----------|----------------|
| `--device-id` | Nunchaku is single-GPU only, always uses cuda:0 |
| `--quant-mode` | Nunchaku handles quantization automatically |
| `--gpu-devices` | Nunchaku doesn't support multi-GPU |

## Recommended Parameter Set

### Minimal Required:
```bash
python3 qwen_t2i_nunchaku.py \
  --model "nunchaku-tech/nunchaku-qwen-image" \
  --prompts '[{"prompt": "...", "filename": "..."}]' \
  --output-dir "downloads/generated" \
  --width 1024 \
  --height 1024 \
  --steps 40 \
  --guidance-scale 4.0 \
  --seed 42 \
  --num-images 1
```

### With Optional Features:
```bash
python3 qwen_t2i_nunchaku.py \
  --model "nunchaku-tech/nunchaku-qwen-image" \
  --prompts '[...]' \
  --output-dir "downloads/generated" \
  --width 1024 \
  --height 1024 \
  --steps 40 \
  --guidance-scale 4.0 \
  --seed 42 \
  --num-images 2 \
  --negative-prompt "blurry, low quality" \
  --static-seed "true" \
  --low-vram "true" \
  --lora-file "downloads/lora/style.safetensors" \
  --lora-adapter-name "mystyle" \
  --lora-scale "1.2"
```

## Updated Low VRAM Behavior

### Current Implementation (Keep):
```python
if args.low_vram == "true":
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
```

### Why It's Still Useful:

1. **Nunchaku's precision selection** optimizes the transformer
2. **Low VRAM mode** optimizes the VAE and memory management
3. **They complement each other** rather than conflict

Example:
- 6GB GPU without Low VRAM: Might OOM on large images
- 6GB GPU with Low VRAM: Can generate 1024x1024 images

## Simplified Logic

```python
# Nunchaku handles transformer optimization automatically
gpu_memory = get_gpu_memory()
precision = get_precision(gpu_memory)

# Load with auto-detected precision
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(...)
pipe = QwenImagePipeline.from_pretrained(...)

# User can OPTIONALLY add memory optimizations
if args.low_vram == "true":
    # These don't conflict with Nunchaku, they help VAE
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
```

## Final Recommendations

### Keep These Parameters:
✅ `--model` (required)
✅ `--width`, `--height` (required)
✅ `--steps`, `--guidance-scale` (required)
✅ `--seed`, `--num-images` (required)
✅ `--prompts`, `--output-dir` (required)
✅ `--negative-prompt` (optional, useful)
✅ `--static-seed` (optional, useful)
✅ `--low-vram` (optional, helpful for low-end GPUs)
✅ `--lora-file`, `--lora-adapter-name`, `--lora-scale` (optional, for LoRA)

### Remove These Parameters:
❌ `--device-id` (not in current code, good!)
❌ `--quant-mode` (not in current code, good!)
❌ `--gpu-devices` (not in current code, good!)

## Current Implementation Status

Looking at your current `qwen_handlers_nunchaku.go`:
- ✅ Only passes needed parameters
- ✅ No multi-GPU params
- ✅ No manual quant mode
- ✅ Includes low-vram as optional
- ✅ Includes LoRA support

**Your current implementation is correct!** 👍

## Summary

**Q: Do we need all the parameters being passed?**  
**A: Yes!** All current parameters are either:
1. Required for generation (model, prompts, dimensions, etc.)
2. Useful optional features (negative prompt, LoRA, low VRAM)

**Q: Does low-vram conflict with Nunchaku?**  
**A: No!** They complement each other:
- Nunchaku optimizes the transformer (automatic)
- Low VRAM optimizes VAE and memory (user choice)

**Q: Should I change anything?**  
**A: No!** Your current parameter set is optimal for Nunchaku.

The only thing you might want to add is better UI messaging to explain that:
- Nunchaku auto-detects optimal settings
- Low VRAM is an optional extra optimization
- LoRA works with Nunchaku models
