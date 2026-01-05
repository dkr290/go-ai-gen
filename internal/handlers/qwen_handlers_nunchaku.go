package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/dkr290/go-ai-gen/internal/download"
	"github.com/dkr290/go-ai-gen/internal/envs"
	"github.com/dkr290/go-ai-gen/utils"
	"github.com/gofiber/fiber/v2"
)

// QwenT2INunchakuHandler renders the Nunchaku Qwen T2I page
func (h *Handler) QwenT2INunchakuHandler(c *fiber.Ctx) error {
	return c.Render(
		"templates/qwen_t2i_nunchaku",
		fiber.Map{"Title": "Nunchaku Qwen Text to Image"},
		"templates/base",
	)
}

// QwenT2INunchakuAPIHandler handles the API request for Nunchaku Qwen T2I generation
func (h *Handler) QwenT2INunchakuAPIHandler(c *fiber.Ctx) error {
	req := new(QwenT2INunchakuRequest)
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

	// Apply suffix, style, and camera shot
	var promptsData []PromptData
	for i, prompt := range prompts {

		filename := utils.SanitizeFilenameForImage(prompt, i+1)
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

		promptsData = append(promptsData, PromptData{
			Prompt:   enhanced,
			Filename: filename,
		})
	}

	// Prepare response
	totalImages := len(promptsData) * req.BatchSize

	// Create output directory
	outputDir := filepath.Join("downloads", "generated", fmt.Sprintf("nunchaku_%d", time.Now().Unix()))
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to create output directory: " + err.Error(),
		})
	}

	// Handle LoRA download if enabled
	loraFilePath := ""
	if req.LoraEnabled && req.LoraURL != "" {
		// Create lora directory if it doesn't exist
		loraDir := filepath.Join("downloads", "lora")
		if err := os.MkdirAll(loraDir, 0o755); err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed to create lora directory: " + err.Error(),
			})
		}

		// Generate unique filename for lora file
		// Extract filename from URL (remove query parameters and get basename)
		cleanURL := req.LoraURL
		if strings.Contains(cleanURL, "?") {
			cleanURL = strings.Split(cleanURL, "?")[0]
		}
		loraFilename := filepath.Base(cleanURL)
		// If filename is empty or weird, fallback to a safe name
		if loraFilename == "" || loraFilename == "." || loraFilename == "/" {
			loraFilename = fmt.Sprintf("lora_%d.safetensors", time.Now().Unix())
		}

		loraFilePath = filepath.Join(loraDir, loraFilename)
		// Check if file already exists
		if _, err := os.Stat(loraFilePath); os.IsNotExist(err) {

			// Download the lora file
			if err := download.DownloadFile(req.LoraURL, loraFilePath); err != nil {
				return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
					"error": "Failed to download LoRA file: " + err.Error(),
				})
			}
			fmt.Printf("✓ LoRA file downloaded to: %s\n", loraFilePath)
		} else {
			fmt.Printf("✓ LoRA file already exists at: %s (skipping download)\n", loraFilePath)
		}
	}

	// Prepare prompts JSON for Python script
	promptsJSON, err := json.Marshal(promptsData)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to marshal prompts: " + err.Error(),
		})
	}

	// Use the Nunchaku model from the form
	nunchakuModel := req.NunchakuModel
	if nunchakuModel == "" {
		nunchakuModel = "nunchaku-tech/nunchaku-qwen-image" // Default
	}

	// Call Python script for Nunchaku
	args := []string{
		"python3", "python_scripts/qwen_t2i_nunchaku.py",
		"--model", nunchakuModel,
		"--width", fmt.Sprintf("%d", width),
		"--height", fmt.Sprintf("%d", height),
		"--steps", fmt.Sprintf("%d", req.Steps),
		"--guidance-scale", fmt.Sprintf("%.1f", req.Guidance),
		"--seed", fmt.Sprintf("%d", seed),
		"--output-dir", outputDir,
		"--num-images", fmt.Sprintf("%d", req.BatchSize),
		"--prompts", string(promptsJSON),
	}

	// Add low VRAM flag if enabled
	if req.LowVRAM {
		args = append(args, "--low-vram", "true")
	} else {
		args = append(args, "--low-vram", "false")
	}

	// Add negative prompt if provided
	if req.NegativePrompt != "" {
		args = append(args, "--negative-prompt", req.NegativePrompt)
	}

	// Add static seed option
	if req.StaticSeed == "true" {
		args = append(args, "--static-seed", req.StaticSeed)
	}

	// Add LoRA arguments if enabled
	if req.LoraEnabled && loraFilePath != "" {
		args = append(args, "--lora-file", loraFilePath)
		if req.LoraAdapterName != "" {
			args = append(args, "--lora-adapter-name", req.LoraAdapterName)
		} else {
			args = append(args, "--lora-adapter-name", "lora")
		}
		// Add LoRA scale
		if req.LoraScale != "" {
			args = append(args, "--lora-scale", req.LoraScale)
		}
	}

	cmd := exec.Command(args[0], args[1:]...)

	// Set up HuggingFace environment
	if req.HFToken != "" {
		envs.SetHuggingFaceEnv(cmd, req.HFToken)
	} else {
		envs.SetHuggingFaceEnv(cmd, "")
	}

	cmd.Env = append(cmd.Env, "HF_HUB_DISABLE_PROGRESS_BARS=true")
	cmd.Env = append(cmd.Env, "DISABLE_TQDM=true")
	cmd.Env = append(cmd.Env, "PYTHONUNBUFFERED=1")

	// Create stdout pipe for capturing JSON output
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to create stdout pipe: " + err.Error(),
		})
	}

	// Stream stderr directly to terminal for real-time progress
	cmd.Stderr = os.Stderr

	// Start the command
	if err := cmd.Start(); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to start Python script: " + err.Error(),
		})
	}

	// Capture stdout while also printing it
	var stdoutBuffer bytes.Buffer
	stdoutMulti := io.MultiWriter(&stdoutBuffer, os.Stdout)
	if _, err := io.Copy(stdoutMulti, stdoutPipe); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to read Python output: " + err.Error(),
		})
	}

	// Wait for command to complete
	if err := cmd.Wait(); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error":  "Python script failed: " + err.Error(),
			"stdout": stdoutBuffer.String(),
		})
	}

	// Parse Python script output
	var pythonResult map[string]any
	if err := json.Unmarshal(stdoutBuffer.Bytes(), &pythonResult); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error":      "Failed to parse Python output: " + err.Error(),
			"raw_output": stdoutBuffer.String(),
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
		"message": fmt.Sprintf("Will generate %d images with Nunchaku", totalImages),
		"data": fiber.Map{
			"prompts":         promptsData,
			"width":           width,
			"height":          height,
			"steps":           req.Steps,
			"guidance":        req.Guidance,
			"seed":            seed,
			"batch_size":      req.BatchSize,
			"total_images":    totalImages,
			"model":           nunchakuModel,
			"negative_prompt": req.NegativePrompt,
			"image_urls":      imageURLs,
			"lora_enabled":    req.LoraEnabled,
			"lora_scale":      req.LoraScale,
		},
	})
}
