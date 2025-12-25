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

	// Static files
	app.Static("/static", "./static")

	// Routes
	app.Get("/", handlers.HomeHandler)

	// Submenu routes
	app.Get("/flux1-dev-t2i", handlers.Flux1DevT2IHandler)
	app.Get("/flux1-kontext-i2i", handlers.Flux1KontextI2IHandler)
	app.Get("/sd-t2i", handlers.SdT2IHandler)
	app.Get("/sd-i2i", handlers.SdI2IHandler)
	app.Get("/qwen-t2i", handlers.QwenT2IHandler)
	app.Get("/qwen-i2i-single", handlers.QwenI2ISingleHandler)
	app.Get("/qwen-i2i-multi", handlers.QwenI2IMultiHandler)

	// API endpoints for HTMX
	app.Post("/api/flux1-dev-t2i", handlers.Flux1DevT2IAPIHandler)
	app.Post("/api/flux1-kontext-i2i", handlers.Flux1KontextI2IAPIHandler)
	app.Post("/api/sd-t2i", handlers.SdT2IAPIHandler)
	app.Post("/api/sd-i2i", handlers.SdI2IAPIHandler)
	app.Post("/api/qwen-t2i", handlers.QwenT2IAPIHandler)
	app.Post("/api/qwen-i2i-single", handlers.QwenI2ISingleAPIHandler)
	app.Post("/api/qwen-i2i-multi", handlers.QwenI2IMultiAPIHandler)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("Server starting on :%s", port)
	log.Fatal(app.Listen(":" + port))
}
