package handlers

import "github.com/gofiber/fiber/v2"

func (h *Handler) Flux1DevT2IHandler(c *fiber.Ctx) error {
	return c.Render(
		"templates/flux1_dev_t2i",
		fiber.Map{"Title": "Flux Text to image"},
		"templates/base",
	)
}
