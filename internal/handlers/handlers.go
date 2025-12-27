// Package handlers
package handlers

import (
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

func (h *Handler) QwenI2ISingleHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-single API endpoint - will call Python script")
}

func (h *Handler) QwenI2IMultiHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-multi API endpoint - will call Python script")
}
