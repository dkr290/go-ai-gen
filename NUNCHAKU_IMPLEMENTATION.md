# Nunchaku Qwen Implementation Summary

## Installation Requirements

### Python Dependencies

Before using Nunchaku Qwen, you need to install the Nunchaku library:

```bash
pip install nunchaku-qwen
# or
pip install git+https://github.com/nunchaku-tech/nunchaku.git
```

### Required Imports

The implementation uses Nunchaku-specific components:

```python
from diffusers import QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
```

### How Nunchaku Works

Nunchaku provides optimized transformer models that replace the standard Qwen transformer:

1. **Load Nunchaku Transformer**: First, load the optimized Nunchaku transformer
   ```python
   transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
       "nunchaku-tech/nunchaku-qwen-image",
       subfolder="transformer"
   )
   ```

2. **Load Pipeline with Nunchaku Transformer**: Then load the Qwen pipeline with this transformer
   ```python
   pipe = QwenImagePipeline.from_pretrained(
       "nunchaku-tech/nunchaku-qwen-image",
       transformer=transformer
   )
   ```

3. **Automatic Precision Detection**: Nunchaku automatically detects optimal precision based on GPU memory
   ```python
   gpu_memory = get_gpu_memory()  # Get available GPU memory
   precision = get_precision(gpu_memory)  # Returns 'bf16', 'fp16', or 'fp32'
   ```

## Files Created/Modified

### New Files Created:

1. **templates/qwen_t2i_nunchaku.html**
   - Web UI for Nunchaku Qwen quantized models
   - Features:
     - Dropdown to select Nunchaku quantized models
     - LoRA support with scale slider (0.0 - 2.0)
     - All standard generation parameters (aspect ratio, steps, guidance, etc.)
     - Style presets and camera shot options
     - Low VRAM mode support
     - Single GPU optimized (no multi-GPU options)
     - Green-themed UI to distinguish from regular Qwen

2. **python_scripts/qwen_t2i_nunchaku.py**
   - Python backend for Nunchaku model inference
   - Features:
     - **Nunchaku-specific implementation**:
       - Uses `NunchakuQwenImageTransformer2DModel` for optimized transformer
       - Uses `QwenImagePipeline` from diffusers
       - Automatic GPU memory detection with `get_gpu_memory()`
       - Automatic precision selection with `get_precision()`
     - Single GPU optimized (cuda:0 or cpu)
     - Graceful fallback if Nunchaku not installed
     - LoRA loading with scale/adapter weights
     - Memory optimizations (CPU offload, VAE slicing, VAE tiling)
     - Generator device-aware for CUDA/CPU
     - All generation parameters support

3. **internal/handlers/qwen_handlers_nunchaku.go**
   - Go handler for Nunchaku API routes
   - Features:
     - QwenT2INunchakuHandler - renders the page
     - QwenT2INunchakuAPIHandler - processes generation requests
     - LoRA file download and caching
     - Nunchaku model selection
     - LoRA scale parameter support
     - Output directory: downloads/generated/nunchaku_{timestamp}

### Modified Files:

4. **internal/handlers/configs.go**
   - Added QwenT2INunchakuRequest struct
   - New fields:
     - NunchakuModel: model selection from Nunchaku repo
     - LoraScale: LoRA weight/influence control

5. **main.go**
   - Added routes:
     - GET /qwen-t2i-nunchaku
     - POST /api/qwen-t2i-nunchaku

6. **templates/menu.html**
   - Added menu link for "Qwen-t2i Nunchaku" with rocket icon

## Key Differences from Regular Qwen

| Feature | Regular Qwen | Nunchaku Qwen |
|---------|-------------|---------------|
| Model Source | Qwen/Qwen-Image | nunchaku-tech/nunchaku-qwen-image |
| Quantization | Manual (BitsAndBytes) | Pre-quantized Nunchaku transformer |
| Transformer | Standard Transformer2DModel | NunchakuQwenImageTransformer2DModel |
| Pipeline | DiffusionPipeline | QwenImagePipeline |
| Precision Detection | Manual | Automatic (get_precision) |
| GPU Memory Detection | Manual | Automatic (get_gpu_memory) |
| Multi-GPU | Supported | Not supported (single GPU only) |
| LoRA Scale | Not available | Slider control (0.0 - 2.0) |
| Default Dtype | bfloat16 | Auto-detected based on GPU |
| Model Loading | Standard loading | Two-step: transformer + pipeline |
| Output Dir | generated/{timestamp} | nunchaku_{timestamp} |

## API Endpoints

### GET /qwen-t2i-nunchaku
Returns the Nunchaku Qwen web interface

### POST /api/qwen-t2i-nunchaku
Generates images using Nunchaku models

**Request Parameters:**
- prompt (required): JSON array of prompts
- nunchaku_model: Model selection (default: nunchaku-tech/nunchaku-qwen-image)
- aspect_ratio: Image dimensions
- steps: Inference steps (20-40 recommended for Nunchaku)
- guidance: CFG scale
- seed: Random seed
- batch_size: Images per prompt
- low_vram: Enable memory optimizations
- negative_prompt: What to avoid
- style_preset: Art style
- camera_shot: Camera composition
- lora_enabled: Enable LoRA
- lora_url: URL to download LoRA file
- lora_adapter_name: LoRA adapter name
- lora_scale: LoRA influence (0.0-2.0)
- hf_token: HuggingFace authentication
- static_seed: Use seed 42

**Response:**
```json
{
  "success": true,
  "message": "Will generate X images with Nunchaku",
  "data": {
    "prompts": [...],
    "width": 928,
    "height": 1664,
    "steps": 40,
    "guidance": 4.0,
    "seed": 42,
    "batch_size": 2,
    "total_images": 2,
    "model": "nunchaku-tech/nunchaku-qwen-image",
    "image_urls": [...],
    "lora_enabled": true,
    "lora_scale": "1.0"
  }
}
```

## Available Nunchaku Models

Based on the form dropdown:
1. `nunchaku-tech/nunchaku-qwen-image` (Default/Recommended)
2. `nunchaku-tech/nunchaku-qwen-image-fp16`
3. `nunchaku-tech/nunchaku-qwen-image-int8`

More models available at: https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/tree/main

## LoRA Support

- **Download**: Auto-downloads LoRA files from URLs
- **Caching**: Stores in downloads/lora/ directory
- **Scale Control**: Slider from 0.0 (no effect) to 2.0 (enhanced effect)
- **Adapter Names**: Customizable adapter naming
- **Formats**: Supports .safetensors and .bin files

## Usage Flow

1. User visits `/qwen-t2i-nunchaku`
2. Selects Nunchaku model from dropdown
3. Configures generation parameters
4. (Optional) Enables LoRA and sets scale
5. Submits form to `/api/qwen-t2i-nunchaku`
6. Backend downloads LoRA if needed
7. Calls `python_scripts/qwen_t2i_nunchaku.py`
8. Python script loads Nunchaku model
9. Generates images
10. Returns image URLs and metadata
11. Frontend displays success message

## Testing Steps

### 1. Install Nunchaku
```bash
pip install nunchaku-qwen
# Verify installation
python -c "from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel; print('Nunchaku installed successfully')"
```

### 2. Start the Server
```bash
go run main.go
```

### 3. Test Generation
1. Navigate to `http://localhost:8080/qwen-t2i-nunchaku`
2. Test with a simple prompt: `["A cat on a skateboard"]`
3. Select model: `nunchaku-tech/nunchaku-qwen-image`
4. Click "Generate Images"
5. Check console output for:
   - ✓ GPU Memory detection
   - ✓ Precision auto-selection
   - ✓ Nunchaku transformer loading
   - ✓ Pipeline loading
   - ✓ Image generation

### 4. Test LoRA
1. Enable LoRA checkbox
2. Provide a public LoRA URL
3. Adjust LoRA scale slider (try 0.5, 1.0, 1.5)
4. Generate and compare results

### 5. Test Low VRAM Mode
1. Enable "Low VRAM Mode" checkbox
2. Generate images
3. Verify CPU offload + VAE optimizations in logs

### 6. Verify Output
1. Check `downloads/generated/nunchaku_*` directory
2. Verify image quality and dimensions
3. Compare generation speed vs regular Qwen

## Documentation Links

- Nunchaku Documentation: https://nunchaku.tech/docs/nunchaku/usage/qwen-image.html
- Nunchaku Models: https://huggingface.co/nunchaku-tech/nunchaku-qwen-image
- Model Files: https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/tree/main
