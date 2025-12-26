package main

import (
	"log"
	"os"

	"github.com/dkr290/go-ai-gen/internal/handlers"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/template/html/v2"
)

func main() {
	// Create Fiber app with HTML template engine
	engine := html.New("./templates", ".html")

	app := fiber.New(fiber.Config{
		Views: engine,
	})

	h := handlers.New()

	// Routes
	app.Get("/", h.HandleHome)

	// Submenu routes
	app.Get("/flux1-dev-t2i", h.Flux1DevT2IHandler)
	app.Get("/flux1-kontext-i2i", h.Flux1KontextI2IHandler)
	app.Get("/sd-t2i", h.SdT2IHandler)
	app.Get("/sd-i2i", h.SdI2IHandler)
	app.Get("/qwen-t2i", h.QwenT2IHandler)
	app.Get("/qwen-i2i-single", h.QwenI2ISingleHandler)
	app.Get("/qwen-i2i-multi", h.QwenI2IMultiHandler)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("Server starting on :%s", port)
	log.Fatal(app.Listen(":" + port))
}
