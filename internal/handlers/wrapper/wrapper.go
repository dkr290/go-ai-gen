package wrapper

import (
	"html/template"
	"net/http"

	"github.com/dkr290/go-ai-gen/internal/handlers"
)

// ErrorHandler is a handler that returns an error
type ErrorHandler func(w http.ResponseWriter, r *http.Request) error

// WrapHandler wraps a standard http.HandlerFunc to return an error
func WrapHandler(h http.HandlerFunc) ErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		h(w, r)
		return nil
	}
}

// WrapTemplateHandler wraps a template-based handler to return an error
func WrapTemplateHandler(h func(tmpl *template.Template) http.HandlerFunc, tmpl *template.Template) ErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		h(tmpl)(w, r)
		return nil
	}
}

// Wrapped handlers that return errors (like Fiber style)
func Flux1DevT2IHandler(tmpl *template.Template) ErrorHandler {
	return WrapTemplateHandler(handlers.Flux1DevT2IHandler, tmpl)
}

func Flux1KontextI2IHandler(tmpl *template.Template) ErrorHandler {
	return WrapTemplateHandler(handlers.Flux1KontextI2IHandler, tmpl)
}

func SdT2IHandler(tmpl *template.Template) ErrorHandler {
	return WrapTemplateHandler(handlers.SdT2IHandler, tmpl)
}

func SdI2IHandler(tmpl *template.Template) ErrorHandler {
	return WrapTemplateHandler(handlers.SdI2IHandler, tmpl)
}

func QwenT2IHandler(tmpl *template.Template) ErrorHandler {
	return WrapTemplateHandler(handlers.QwenT2IHandler, tmpl)
}

func QwenI2ISingleHandler(tmpl *template.Template) ErrorHandler {
	return WrapTemplateHandler(handlers.QwenI2ISingleHandler, tmpl)
}

func QwenI2IMultiHandler(tmpl *template.Template) ErrorHandler {
	return WrapTemplateHandler(handlers.QwenI2IMultiHandler, tmpl)
}

// API handlers
func Flux1DevT2IAPIHandler() ErrorHandler {
	return WrapHandler(handlers.Flux1DevT2IAPIHandler)
}

func Flux1KontextI2IAPIHandler() ErrorHandler {
	return WrapHandler(handlers.Flux1KontextI2IAPIHandler)
}

func SdT2IAPIHandler() ErrorHandler {
	return WrapHandler(handlers.SdT2IAPIHandler)
}

func SdI2IAPIHandler() ErrorHandler {
	return WrapHandler(handlers.SdI2IAPIHandler)
}

func QwenT2IAPIHandler() ErrorHandler {
	return WrapHandler(handlers.QwenT2IAPIHandler)
}

func QwenI2ISingleAPIHandler() ErrorHandler {
	return WrapHandler(handlers.QwenI2ISingleAPIHandler)
}

func QwenI2IMultiAPIHandler() ErrorHandler {
	return WrapHandler(handlers.QwenI2IMultiAPIHandler)
}

// Middleware to convert ErrorHandler to http.HandlerFunc
func ToHTTPHandler(h ErrorHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := h(w, r); err != nil {
			// Handle the error - you can customize this based on your needs
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}