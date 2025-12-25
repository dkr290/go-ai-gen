package wrapper

import (
	"html/template"
	"net/http"

	"github.com/dkr290/go-ai-gen/internal/handlers"
)

// EnhancedErrorHandler is a handler that returns an error with enhanced capabilities
type EnhancedErrorHandler func(w http.ResponseWriter, r *http.Request) error

// Enhanced wrapper that provides error handling capabilities similar to Fiber
type EnhancedWrapper struct {
	tmpl *template.Template
}

// NewEnhancedWrapper creates a new enhanced wrapper
func NewEnhancedWrapper(tmpl *template.Template) *EnhancedWrapper {
	return &EnhancedWrapper{tmpl: tmpl}
}

// Enhanced template handler with error propagation
func (ew *EnhancedWrapper) Flux1DevT2IHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		// Example of error handling - you could add validation, database calls, etc.
		if r.Method != http.MethodGet {
			return &HTTPError{
				Code:    http.StatusMethodNotAllowed,
				Message: "Method not allowed",
			}
		}
		
		// Call the original handler
		handlers.Flux1DevT2IHandler(ew.tmpl)(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) Flux1KontextI2IHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.Flux1KontextI2IHandler(ew.tmpl)(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) SdT2IHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.SdT2IHandler(ew.tmpl)(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) SdI2IHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.SdI2IHandler(ew.tmpl)(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) QwenT2IHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.QwenT2IHandler(ew.tmpl)(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) QwenI2ISingleHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.QwenI2ISingleHandler(ew.tmpl)(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) QwenI2IMultiHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.QwenI2IMultiHandler(ew.tmpl)(w, r)
		return nil
	}
}

// API handlers with error handling
func (ew *EnhancedWrapper) Flux1DevT2IAPIHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		// Example: Validate request
		if err := r.ParseForm(); err != nil {
			return &HTTPError{
				Code:    http.StatusBadRequest,
				Message: "Invalid form data",
			}
		}
		
		prompt := r.FormValue("prompt")
		if prompt == "" {
			return &HTTPError{
				Code:    http.StatusBadRequest,
				Message: "Prompt is required",
			}
		}
		
		handlers.Flux1DevT2IAPIHandler(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) Flux1KontextI2IAPIHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.Flux1KontextI2IAPIHandler(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) SdT2IAPIHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.SdT2IAPIHandler(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) SdI2IAPIHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.SdI2IAPIHandler(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) QwenT2IAPIHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.QwenT2IAPIHandler(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) QwenI2ISingleAPIHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.QwenI2ISingleAPIHandler(w, r)
		return nil
	}
}

func (ew *EnhancedWrapper) QwenI2IMultiAPIHandler() EnhancedErrorHandler {
	return func(w http.ResponseWriter, r *http.Request) error {
		handlers.QwenI2IMultiAPIHandler(w, r)
		return nil
	}
}

// HTTPError represents an HTTP error
type HTTPError struct {
	Code    int
	Message string
}

func (e *HTTPError) Error() string {
	return e.Message
}

// ToHTTPHandler converts EnhancedErrorHandler to http.HandlerFunc with error handling
func (ew *EnhancedWrapper) ToHTTPHandler(h EnhancedErrorHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := h(w, r); err != nil {
			if httpErr, ok := err.(*HTTPError); ok {
				http.Error(w, httpErr.Message, httpErr.Code)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		}
	}
}

// Middleware support similar to Fiber
func (ew *EnhancedWrapper) Use(middleware func(EnhancedErrorHandler) EnhancedErrorHandler) {
	// This shows how you could implement middleware chaining
	// In practice, you'd store middleware and apply it when creating handlers
}

// Group support similar to Fiber (conceptual)
func (ew *EnhancedWrapper) Group(prefix string) *EnhancedWrapper {
	// This would return a new wrapper with route prefix
	// For simplicity, we return the same wrapper
	return ew
}