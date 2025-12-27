// Package handlers
package handlers

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

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
		PositiveMagic  string  `form:"positive_magic"`
		QwenModel      string  `form:"qwen_model"`
		AspectRatio    string  `form:"aspect_ratio"`
		Steps          int     `form:"steps"`
		Guidance       float64 `form:"guidance"`
		StylePreset    string  `form:"style_preset"`
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
	width, height, err := parseDimensions(req.AspectRatio)
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

	// Apply positive magic
	var enhancedPrompts []string
	for _, prompt := range prompts {
		enhanced := strings.TrimSpace(prompt)
		if req.PositiveMagic != "" {
			enhanced = enhanced + ", " + strings.TrimSpace(req.PositiveMagic)
		}
		enhancedPrompts = append(enhancedPrompts, enhanced)
	}

	// Prepare response
	totalImages := len(enhancedPrompts) * req.BatchSize

	return c.JSON(fiber.Map{
		"success": true,
		"message": fmt.Sprintf("Will generate %d images", totalImages),
		"data": fiber.Map{
			"prompts":         enhancedPrompts,
			"width":           width,
			"height":          height,
			"steps":           req.Steps,
			"guidance":        req.Guidance,
			"style_preset":    req.StylePreset,
			"seed":            seed,
			"batch_size":      req.BatchSize,
			"total_images":    totalImages,
			"qwen_model":      req.QwenModel,
			"negative_prompt": req.NegativePrompt,
		},
	})
}

// Add this helper function
func parseDimensions(dimStr string) (int, int, error) {
	if dimStr == "" {
		return 0, 0, fmt.Errorf("dimension string is empty")
	}

	parts := strings.Split(dimStr, "x")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("expected 'WIDTHxHEIGHT' format")
	}

	width, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	height, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil {
		return 0, 0, fmt.Errorf("invalid dimensions")
	}

	if width <= 0 || height <= 0 {
		return 0, 0, fmt.Errorf("dimensions must be positive")
	}

	return width, height, nil
}

func (h *Handler) QwenI2ISingleHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-single API endpoint - will call Python script")
}

func (h *Handler) QwenI2IMultiHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-multi API endpoint - will call Python script")
}
