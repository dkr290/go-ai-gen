// Package utils
package utils

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
)

var cameraFormat = map[string]string{
	// Subject-focused (use "of")
	"extreme close-up":       "of",
	"close-up shot":          "of",
	"medium close-up":        "of",
	"medium shot":            "of",
	"medium full shot":       "of",
	"full shot":              "of",
	"detail shot":            "of",
	"insert shot":            "of",
	"macro shot":             "of",
	"profile shot":           "of",
	"front shot":             "of",
	"side shot":              "of",
	"rear shot":              "of",
	"three-quarter view":     "of",
	"intimate portrait shot": "of",
	"cinematic close-up":     "of",

	// Environment-focused (use "showing")
	"wide angle shot":        "showing",
	"ultra wide angle":       "showing",
	"long shot":              "showing",
	"extreme long shot":      "showing",
	"establishing shot":      "showing",
	"epic establishing shot": "showing",
	"wide cinematic shot":    "showing",
	"bird's eye view":        "showing",
	"worm's eye view":        "showing",
	"overhead shot":          "showing",
	"top-down shot":          "showing",
	"bottom-up shot":         "showing",
	"environmental portrait": "showing",

	// Angle / stylistic (comma separation)
	"dutch angle":                  ",",
	"tilted angle":                 ",",
	"cinematic shot":               ",",
	"dynamic shot":                 ",",
	"dramatic shot":                ",",
	"moody shot":                   ",",
	"minimalist shot":              ",",
	"action shot":                  ",",
	"medium shot from a low angle": ",",

	// POV (special prefix)
	"point of view":     "POV",
	"first-person view": "POV",
	"subjective camera": "POV",

	// Neutral append
	"eye-level shot":         " ",
	"high angle shot":        " ",
	"low angle shot":         " ",
	"over-the-shoulder shot": " ",
	"telephoto shot":         " ",
	"fisheye lens":           " ",
	"tilt-shift":             " ",
	"shallow depth of field": " ",
	"deep focus shot":        " ",
	"rack focus":             " ",
	"soft focus":             " ",
	"sharp focus":            " ",
	"bokeh background":       " ",
	"motion blur":            " ",
	"freeze frame":           " ",
	"third-person view":      " ",
}

func ParseDimensions(dimStr string) (int, int, error) {
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

// MapStylePreset map style preset to prompt text
func MapStylePreset(preset string) string {
	switch preset {
	case "photorealistic":
		return "photorealistic, realistic, highly detailed"
	case "anime":
		return "anime style, anime art, Japanese animation style"
	case "digital_art":
		return "digital art, digital painting, concept art"
	case "painting":
		return "oil painting, brush strokes, painterly style"
	case "sketch":
		return "sketch, pencil drawing, line art"
	case "watercolor":
		return "watercolor painting, watercolor art"
	case "cyberpunk":
		return "cyberpunk, neon, futuristic, sci-fi"
	case "fantasy":
		return "fantasy art, magical, mystical"
	default:
		return preset + " style"
	}
}

func FormatCameraShot(cameraShot, prompt string) string {
	if cameraShot == "" {
		return prompt
	}

	if mode, ok := cameraFormat[cameraShot]; ok {
		switch mode {
		case "of":
			return cameraShot + " of " + prompt
		case "showing":
			return cameraShot + " showing " + prompt
		case ",":
			return cameraShot + ", " + prompt
		case "POV":
			return "POV: " + prompt
		default:
			return cameraShot + " " + prompt
		}
	}

	return cameraShot + " " + prompt
}

func SetupLogger() logger.Config {
	logLevel := os.Getenv("LOG_LEVEL")
	logFormat := os.Getenv("LOG_FORMAT")

	var format string

	switch logLevel {
	case "verbose", "debug":
		format = "${time} | ${status} | ${latency} | ${method} | ${path} | ${ip} | ${error}\n"
		if logFormat == "json" {
			format = `{"time":"${time}","status":"${status}","latency":"${latency}","method":"${method}","path":"${path}","ip":"${ip}","error":"${error}"}`
		}
	case "minimal":
		format = "${status} | ${method} | ${path} | ${latency}\n"
	case "none":
		return logger.Config{
			Next: func(c *fiber.Ctx) bool { return true }, // Skip all logging
		}
	default: // "info" or empty
		format = "${time} | ${status} | ${latency} | ${method} | ${path}\n"
	}

	return logger.Config{
		Format:     format,
		TimeFormat: "2006-01-02 15:04:05",
		TimeZone:   "Local",
		Output:     os.Stdout,
	}
}

// FormatFileSize  function to format file size
func FormatFileSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func SanitizeFilename(text string) string {
	text = strings.ReplaceAll(text, "/", "-")
	text = strings.ReplaceAll(text, " ", "-")
	text = strings.ReplaceAll(text, ",", "")
	return strings.ToLower(text)
}

func SanitizeFilenameForImage(prompt string, index int) string {
	// Sanitize the prompt
	sanitized := SanitizeFilename(prompt)

	// Take first 10 characters (or less) for the prompt part
	promptPart := sanitized
	if len(promptPart) > 10 {
		promptPart = promptPart[:10]
	}

	// Remove trailing dash if present
	promptPart = strings.TrimSuffix(promptPart, "-")

	// Format: {index}_{prompt10}.png
	// Max: 2 + 1 + 10 + 4 = 17 chars (well under 20)

	return fmt.Sprintf("%02d_%s.png", index, promptPart)
}

func GetFilenameFromURL(url string) string {
	// Extract the last part of the URL
	parts := strings.Split(url, "/")
	return parts[len(parts)-1]
}
