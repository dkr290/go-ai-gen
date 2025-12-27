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
