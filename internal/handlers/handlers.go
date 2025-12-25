package handlers

import (
	"github.com/gofiber/fiber/v2"
)

type PageData struct {
	Title   string
	Content string
}

// Handler functions
func HomeHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "Home",
		Content: "Welcome to the AI Model Interface",
	}
	return c.Render("index", data)
}

func Flux1DevT2IHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "Flux1-dev t2i",
		Content: "Flux1-dev Text-to-Image Interface",
	}
	return c.Render("flux1_dev_t2i", data)
}

func Flux1KontextI2IHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "Flux1-Kontext i2i",
		Content: "Flux1-Kontext Image-to-Image Interface",
	}
	return c.Render("flux1_kontext_i2i", data)
}

func SdT2IHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "sd-t2i",
		Content: "Stable Diffusion Text-to-Image Interface",
	}
	return c.Render("sd_t2i", data)
}

func SdI2IHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "sd-i2i",
		Content: "Stable Diffusion Image-to-Image Interface",
	}
	return c.Render("sd_i2i", data)
}

func QwenT2IHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "Qwen-t2i",
		Content: "Qwen Text-to-Image Interface",
	}
	return c.Render("qwen_t2i", data)
}

func QwenI2ISingleHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "Qwen-i2i-single",
		Content: "Qwen Single Image-to-Image Interface",
	}
	return c.Render("qwen_i2i_single", data)
}

func QwenI2IMultiHandler(c *fiber.Ctx) error {
	data := PageData{
		Title:   "Qwen-i2i-multi",
		Content: "Qwen Multi Image-to-Image Interface",
	}
	return c.Render("qwen_i2i_multi", data)
}

// API handler functions (placeholder)
func Flux1DevT2IAPIHandler(c *fiber.Ctx) error {
	return c.SendString("Flux1-dev t2i API endpoint - will call Python script")
}

func Flux1KontextI2IAPIHandler(c *fiber.Ctx) error {
	return c.SendString("Flux1-Kontext i2i API endpoint - will call Python script")
}

func SdT2IAPIHandler(c *fiber.Ctx) error {
	return c.SendString("sd-t2i API endpoint - will call Python script")
}

func SdI2IAPIHandler(c *fiber.Ctx) error {
	return c.SendString("sd-i2i API endpoint - will call Python script")
}

func QwenT2IAPIHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-t2i API endpoint - will call Python script")
}

func QwenI2ISingleAPIHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-single API endpoint - will call Python script")
}

func QwenI2IMultiAPIHandler(c *fiber.Ctx) error {
	return c.SendString("Qwen-i2i-multi API endpoint - will call Python script")
}