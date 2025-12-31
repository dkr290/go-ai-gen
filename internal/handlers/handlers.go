// Package handlers
package handlers

import (
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

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

func (h *Handler) FileServerHandler(c *fiber.Ctx) error {
	// This will render a file browser template
	return c.Render("templates/file_server", fiber.Map{"Title": "File Server"}, "templates/base")
}

func (h *Handler) QwenI2ISingleHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-single API endpoint - will call Python script")
}

func (h *Handler) QwenI2IMultiHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-multi API endpoint - will call Python script")
}

func (h *Handler) FileServerListHandler(c *fiber.Ctx) error {
	// List files from downloads/generated directory
	baseDir := "downloads/generated"

	// Walk through all subdirectories
	var files []map[string]string

	err := filepath.Walk(baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories and non-image files
		if info.IsDir() {
			return nil
		}

		// Check if it's an image file
		ext := strings.ToLower(filepath.Ext(path))
		imageExts := []string{".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
		isImage := slices.Contains(imageExts, ext)

		if !isImage {
			return nil
		}

		// Get relative path for URL
		relPath, err := filepath.Rel("static", path)
		if err != nil {
			relPath = strings.TrimPrefix(path, "static/")
		}

		// Format file size
		size := utils.FormatFileSize(info.Size())

		// Format modification time
		modified := info.ModTime().Format("2006-01-02 15:04")

		files = append(files, map[string]string{
			"name":     filepath.Base(path),
			"path":     path,
			"url":      "/" + relPath,
			"size":     size,
			"modified": modified,
		})

		return nil
	})
	if err != nil {
		// If directory doesn't exist, return empty list
		if os.IsNotExist(err) {
			return c.JSON(fiber.Map{
				"success": true,
				"files":   []string{},
			})
		}

		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"success": false,
			"error":   "Failed to list files: " + err.Error(),
		})
	}

	// Sort by modification time (newest first)
	sort.Slice(files, func(i, j int) bool {
		return files[i]["modified"] > files[j]["modified"]
	})

	return c.JSON(fiber.Map{
		"success": true,
		"files":   files,
	})
}
