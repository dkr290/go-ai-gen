// Package envs set up global important envs
package envs

import (
	"fmt"
	"os"
	"os/exec"
)

func SetHuggingFaceEnv(cmd *exec.Cmd, hfToken string) {
	// Determine cache directory
	cacheDir := "./downloads/models/.cache"

	// Try to get from environment first
	if envCache := os.Getenv("HF_HOME"); envCache != "" {
		cacheDir = envCache
	}

	// Create cache directory if it doesn't exist
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		fmt.Printf("Warning: Failed to create cache directory %s: %v\n", cacheDir, err)
	}

	// Set all relevant Hugging Face environment variables
	env := append(os.Environ(),
		"HF_HOME="+cacheDir,
		"TRANSFORMERS_CACHE="+cacheDir,
		"DIFFUSERS_CACHE="+cacheDir,
		"HUGGINGFACE_HUB_CACHE="+cacheDir,
	)

	if hfToken != "" {
		env = append(env, "HF_TOKEN="+hfToken)
	}
	cmd.Env = env
	fmt.Printf("Set Hugging Face cache to: %s\n", cacheDir)
}

// GetCacheDir returns the Hugging Face cache directory
func GetCacheDir() string {
	if cacheDir := os.Getenv("HF_HOME"); cacheDir != "" {
		return cacheDir
	}
	return "./downloads/models/.cache"
}
