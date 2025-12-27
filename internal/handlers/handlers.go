// Package handlers
package handlers

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/dkr290/go-ai-gen/utils"
	"github.com/gofiber/fiber/v2"
)

type Handler struct{}

func New() *Handler {
	return &Handler{}
}

func (h *Handler) HandleHome(c *fiber.Ctx) error {
	return c.Render("templates/index", fiber.Map{"Title": "Home"}, "templates/base")
}

func (h *Handler) Flux1DevT2IHandler(c *fiber.Ctx) error {
	return c.SendString("Flux1-dev t2i API endpoint - will call Python script")
}

func (h *Handler) Flux1KontextI2IHandler(c *fiber.Ctx) error {
	return c.SendString("Flux1-Kontext i2i API endpoint - will call Python script")
}

func (h *Handler) SdT2IHandler(c *fiber.Ctx) error {
	return c.SendString("sd-t2i API endpoint - will call Python script")
}

func (h *Handler) SdI2IHandler(c *fiber.Ctx) error {
	return c.SendString("sd-i2i API endpoint - will call Python script")
}

func (h *Handler) QwenT2IHandler(c *fiber.Ctx) error {
	return c.Render(
		"templates/qwen_t2i",
		fiber.Map{"Title": "Qwen Text to image"},
		"templates/base",
	)
}

func (h *Handler) QwenT2IAPIHandler(c *fiber.Ctx) error {
	type QwenT2IRequest struct {
		Prompt         string  `form:"prompt"`
		Suffix         string  `form:"suffix"`
		QwenModel      string  `form:"qwen_model"`
		AspectRatio    string  `form:"aspect_ratio"`
		Steps          int     `form:"steps"`
		Guidance       float64 `form:"guidance"`
		StylePreset    string  `form:"style_preset"`
		CameraShot     string  `form:"camera_shot"`
		LowVRAM        bool    `form:"low_vram"` // Add this field
		Seed           int64   `form:"seed"`
		BatchSize      int     `form:"batch_size"`
		NegativePrompt string  `form:"negative_prompt"`
	}

	req := new(QwenT2IRequest)
	if err := c.BodyParser(req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request format: " + err.Error(),
		})
	}

	// Parse JSON array of prompts
	var prompts []string
	if err := json.Unmarshal([]byte(req.Prompt), &prompts); err != nil {
		// Try to handle if user entered a single string
		trimmed := strings.TrimSpace(req.Prompt)
		if trimmed != "" {
			if strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]") {
				return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
					"error": "Invalid JSON array. Expected: [\"prompt1\", \"prompt2\"]. Error: " + err.Error(),
				})
			}
			prompts = []string{trimmed}
		} else {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Prompt is required",
			})
		}
	}

	if len(prompts) == 0 {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "At least one prompt is required",
		})
	}

	// Parse dimensions
	width, height, err := utils.ParseDimensions(req.AspectRatio)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid aspect ratio: " + err.Error(),
		})
	}

	// Handle seed
	seed := req.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	// Apply suffix
	var enhancedPrompts []string
	for _, prompt := range prompts {
		enhanced := strings.TrimSpace(prompt)

		if req.CameraShot != "" {
			enhanced = utils.FormatCameraShot(req.CameraShot, enhanced)
		}

		if req.StylePreset != "" && req.StylePreset != "none" {
			styleText := utils.MapStylePreset(req.StylePreset)
			enhanced = enhanced + ", " + styleText
		}

		if req.Suffix != "" {
			enhanced = enhanced + ", " + strings.TrimSpace(req.Suffix)
		}
		enhancedPrompts = append(enhancedPrompts, enhanced)
	}

	// Prepare response
	totalImages := len(enhancedPrompts) * req.BatchSize

	// Create output directory
	outputDir := filepath.Join("images", "generated", fmt.Sprintf("%d", time.Now().Unix()))
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to create output directory: " + err.Error(),
		})
	}

	// Prepare prompts JSON for Python script
	promptsJSON, err := json.Marshal(enhancedPrompts)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to marshal prompts: " + err.Error(),
		})
	}
	// Call Python script
	cmd := exec.Command("python3", "python_scripts/qwen_t2i.py",
		"--model", req.QwenModel,
		"--negative-prompt", req.NegativePrompt,
		"--width", fmt.Sprintf("%d", width),
		"--height", fmt.Sprintf("%d", height),
		"--steps", fmt.Sprintf("%d", req.Steps),
		"--guidance-scale", fmt.Sprintf("%.1f", req.Guidance),
		"--seed", fmt.Sprintf("%d", seed),
		"--output-dir", outputDir,
		"--num-images", fmt.Sprintf("%d", req.BatchSize),
		"--prompts", string(promptsJSON),
		"--low-vram", strconv.FormatBool(req.LowVRAM),
	)
	output, err := cmd.CombinedOutput()
	fmt.Printf("=== PYTHON OUTPUT ===\n%s\n", string(output))
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error":  "Python script failed: " + err.Error(),
			"output": string(output),
		})
	}
	// Parse Python script output
	var pythonResult map[string]any
	if err := json.Unmarshal(output, &pythonResult); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error":      "Failed to parse Python output: " + err.Error(),
			"raw_output": string(output),
		})
	}

	if pythonResult["status"] == "error" {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Generation failed: " + pythonResult["error"].(string),
		})
	}

	// Prepare image URLs for display
	var imageURLs []string
	if generations, ok := pythonResult["generations"].([]any); ok {
		for _, gen := range generations {
			if genMap, ok := gen.(map[string]any); ok {
				if filename, ok := genMap["filename"].(string); ok {
					// Convert file path to URL path
					relPath := strings.TrimPrefix(outputDir, "static")
					imageURLs = append(imageURLs, filepath.Join(relPath, filename))
				}
			}
		}
	}

	return c.JSON(fiber.Map{
		"success": true,
		"message": fmt.Sprintf("Will generate %d images", totalImages),
		"data": fiber.Map{
			"prompts":         enhancedPrompts,
			"width":           width,
			"height":          height,
			"steps":           req.Steps,
			"guidance":        req.Guidance,
			"seed":            seed,
			"batch_size":      req.BatchSize,
			"total_images":    totalImages,
			"qwen_model":      req.QwenModel,
			"negative_prompt": req.NegativePrompt,
			"image_urls":      imageURLs,
		},
	})
}

func (h *Handler) QwenI2ISingleHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-single API endpoint - will call Python script")
}

func (h *Handler) QwenI2IMultiHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-multi API endpoint - will call Python script")
}
