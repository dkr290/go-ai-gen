# AI Model Interface

A web interface for various AI image generation models using Go, HTMX, and Bootstrap.

## Features

- 7 different AI model interfaces:
  - Flux1-dev t2i (Text-to-Image)
  - Flux1-Kontext i2i (Image-to-Image)
  - sd-t2i (Stable Diffusion Text-to-Image)
  - sd-i2i (Stable Diffusion Image-to-Image)
  - Qwen-t2i (Qwen Text-to-Image)
  - Qwen-i2i-single (Single Image-to-Image)
  - Qwen-i2i-multi (Multiple Image-to-Image)

- Modern UI with Bootstrap 5.3
- HTMX for dynamic updates
- Left-side hamburger menu
- Responsive design

## Installation

1. Install Go 1.21 or later
2. Clone the repository
3. Install dependencies:
   ```bash
   go mod download
   ```

## Running the Application

```bash
go run main.go
```

The server will start on `http://localhost:8080`

## Project Structure

- `main.go` - Main application file with routes and handlers
- `templates/` - HTML templates
  - `base.html` - Base layout template
  - `menu.html` - Menu partial
  - `index.html` - Home page
  - `flux1_dev_t2i.html` - Flux1-dev t2i interface
  - `flux1_kontext_i2i.html` - Flux1-Kontext i2i interface
  - `sd_t2i.html` - Stable Diffusion t2i interface
  - `sd_i2i.html` - Stable Diffusion i2i interface
  - `qwen_t2i.html` - Qwen t2i interface
  - `qwen_i2i_single.html` - Qwen single i2i interface
  - `qwen_i2i_multi.html` - Qwen multi i2i interface

## Dependencies

- [chi](https://github.com/go-chi/chi) - Lightweight HTTP router
- Bootstrap 5.3 (CDN)
- HTMX 1.9.10 (CDN)
- Bootstrap Icons (CDN)

## Next Steps

1. Implement Python backend integration for each model
2. Add file upload handling for image-to-image models
3. Implement actual image generation/transformation
4. Add user authentication if needed
5. Add image gallery/history

## Some loras for testing

- starsfriday/Qwen-Image-NSFW/qwen_image_nsfw.safetensors

