package wrapper

import (
	"html/template"
	"net/http"

	"github.com/dkr290/go-ai-gen/internal/handlers"
)

// Handler is the Fiber-like handler interface that returns an error
type Handler func(w http.ResponseWriter, r *http.Request) error

// Adaptor converts a standard http.HandlerFunc to our Handler type
func Adaptor(h http.HandlerFunc) Handler {
	return func(w http.ResponseWriter, r *http.Request) error {
		h(w, r)
		return nil
	}
}

// TemplateAdaptor converts a template-based handler to our Handler type
func TemplateAdaptor(h func(tmpl *template.Template) http.HandlerFunc, tmpl *template.Template) Handler {
	return func(w http.ResponseWriter, r *http.Request) error {
		h(tmpl)(w, r)
		return nil
	}
}

// Fiber-style handlers that can return errors
func NewFlux1DevT2IHandler(tmpl *template.Template) Handler {
	return TemplateAdaptor(handlers.Flux1DevT2IHandler, tmpl)
}

func NewFlux1KontextI2IHandler(tmpl *template.Template) Handler {
	return TemplateAdaptor(handlers.Flux1KontextI2IHandler, tmpl)
}

func NewSdT2IHandler(tmpl *template.Template) Handler {
	return TemplateAdaptor(handlers.SdT2IHandler, tmpl)
}

func NewSdI2IHandler(tmpl *template.Template) Handler {
	return TemplateAdaptor(handlers.SdI2IHandler, tmpl)
}

func NewQwenT2IHandler(tmpl *template.Template) Handler {
	return TemplateAdaptor(handlers.QwenT2IHandler, tmpl)
}

func NewQwenI2ISingleHandler(tmpl *template.Template) Handler {
	return TemplateAdaptor(handlers.QwenI2ISingleHandler, tmpl)
}

func NewQwenI2IMultiHandler(tmpl *template.Template) Handler {
	return TemplateAdaptor(handlers.QwenI2IMultiHandler, tmpl)
}

// API handlers
func NewFlux1DevT2IAPIHandler() Handler {
	return Adaptor(handlers.Flux1DevT2IAPIHandler)
}

func NewFlux1KontextI2IAPIHandler() Handler {
	return Adaptor(handlers.Flux1KontextI2IAPIHandler)
}

func NewSdT2IAPIHandler() Handler {
	return Adaptor(handlers.SdT2IAPIHandler)
}

func NewSdI2IAPIHandler() Handler {
	return Adaptor(handlers.SdI2IAPIHandler)
}

func NewQwenT2IAPIHandler() Handler {
	return Adaptor(handlers.QwenT2IAPIHandler)
}

func NewQwenI2ISingleAPIHandler() Handler {
	return Adaptor(handlers.QwenI2ISingleAPIHandler)
}

func NewQwenI2IMultiAPIHandler() Handler {
	return Adaptor(handlers.QwenI2IMultiAPIHandler)
}

// ToHTTP converts our Handler to standard http.HandlerFunc with error handling
func ToHTTP(h Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := h(w, r); err != nil {
			// Custom error handling - you can log, send JSON, etc.
			handleError(w, r, err)
		}
	}
}

// handleError provides centralized error handling
func handleError(w http.ResponseWriter, r *http.Request, err error) {
	// You can customize this based on your needs
	// For example, check Accept header to decide JSON vs HTML error response
	
	// Simple implementation for now
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

// Error types for better error handling
type AppError struct {
	Code    int
	Message string
	Err     error
}

func (e *AppError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}
	return e.Message
}

func NewAppError(code int, message string, err error) *AppError {
	return &AppError{
		Code:    code,
		Message: message,
		Err:     err,
	}
}

// Middleware support (Fiber-style)
func Middleware(next Handler) Handler {
	return func(w http.ResponseWriter, r *http.Request) error {
		// Example middleware: log request
		// log.Printf("Request: %s %s", r.Method, r.URL.Path)
		
		// Call the next handler
		return next(w, r)
	}
}

// Chain multiple middlewares
func Chain(middlewares ...func(Handler) Handler) func(Handler) Handler {
	return func(next Handler) Handler {
		for i := len(middlewares) - 1; i >= 0; i-- {
			next = middlewares[i](next)
		}
		return next
	}
}