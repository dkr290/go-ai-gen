// Package utils
package utils

import (
	"fmt"
	"strconv"
	"strings"
)

// Add this helper function
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
	switch cameraShot {
	case "front shot", "close-up shot", "extreme close-up", "medium shot",
		"full shot", "long shot", "low angle shot", "high angle shot":
		return cameraShot + " of " + prompt
	case "wide angle shot", "establishing shot", "overhead shot",
		"bird's eye view", "worm's eye view":
		return cameraShot + " showing " + prompt
	case "dutch angle":
		return prompt + ", " + cameraShot
	case "point of view":
		return "POV: " + prompt
	default:
		return cameraShot + " " + prompt
	}
}
