package wrapper

import (
	"html/template"
	"net/http"

	"github.com/dkr290/go-ai-gen/internal/handlers"
)

// ExtendedHandler provides enhanced error-returning handlers with validation
type ExtendedHandler struct {
	tmpl *template.Template
}

// NewExtendedHandler creates a new extended handler
func NewExtendedHandler(tmpl *template.Template) *ExtendedHandler {
	return &ExtendedHandler{tmpl: tmpl}
}

// Flux1DevT2IHandlerWithValidation adds validation before calling the original handler
func (eh *ExtendedHandler) Flux1DevT2IHandlerWithValidation() Handler {
	return func(w http.ResponseWriter, r *http.Request) error {
		// Example validation - check if user is authenticated
		// token := r.Header.Get("Authorization")
		// if token == "" {
		//     return NewAppError(http.StatusUnauthorized, "Authentication required", nil)
		// }
		
		// Call the original handler
		handlers.Flux1DevT2IHandler(eh.tmpl)(w, r)
		return nil
	}
}

// Flux1DevT2IAPIHandlerWithProcessing adds processing and error handling
func (eh *ExtendedHandler) Flux1DevT2IAPIHandlerWithProcessing() Handler {
	return func(w http.ResponseWriter, r *http.Request) error {
		// Parse form data
		if err := r.ParseForm(); err != nil {
			return NewAppError(http.StatusBadRequest, "Failed to parse form data", err)
		}
		
		// Validate required fields
		prompt := r.FormValue("prompt")
		if prompt == "" {
			return NewAppError(http.StatusBadRequest, "Prompt is required", nil)
		}
		
		// Validate prompt length
		if len(prompt) > 1000 {
			return NewAppError(http.StatusBadRequest, "Prompt too long (max 1000 characters)", nil)
		}
		
		// You could add database calls, external API calls, etc. here
		// All of which could return errors
		
		// Call the original handler
		handlers.Flux1DevT2IAPIHandler(w, r)
		return nil
	}
}

// Example of a handler that calls a Python script and handles errors
func (eh *ExtendedHandler) Flux1DevT2IAPIHandlerWithPython() Handler {
	return func(w http.ResponseWriter, r *http.Request) error {
		// Parse form data
		if err := r.ParseForm(); err != nil {
			return NewAppError(http.StatusBadRequest, "Failed to parse form data", err)
		}
		
		prompt := r.FormValue("prompt")
		if prompt == "" {
			return NewAppError(http.StatusBadRequest, "Prompt is required", nil)
		}
		
		// Example: Call Python script (this would be your actual integration)
		// cmd := exec.Command("python3", "python_scripts/flux1_dev_t2i.py", 
		//     "--prompt", prompt,
		//     "--steps", r.FormValue("steps"),
		//     "--guidance_scale", r.FormValue("guidance_scale"))
		// 
		// output, err := cmd.CombinedOutput()
		// if err != nil {
		//     return NewAppError(http.StatusInternalServerError, 
		//         "Failed to generate image", 
		//         fmt.Errorf("python script error: %v, output: %s", err, output))
		// }
		// 
		// // Parse Python script output
		// var result map[string]interface{}
		// if err := json.Unmarshal(output, &result); err != nil {
		//     return NewAppError(http.StatusInternalServerError, 
		//         "Failed to parse Python script output", err)
		// }
		// 
		// // Return JSON response
		// w.Header().Set("Content-Type", "application/json")
		// json.NewEncoder(w).Encode(result)
		
		// For now, call the placeholder handler
		handlers.Flux1DevT2IAPIHandler(w, r)
		return nil
	}
}

// Middleware example: Logging middleware
func LoggingMiddleware(next Handler) Handler {
	return func(w http.ResponseWriter, r *http.Request) error {
		log.Printf("Request started: %s %s", r.Method, r.URL.Path)
		
		// Create a custom response writer to capture status code
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		err := next(rw, r)
		
		if err != nil {
			log.Printf("Request error: %s %s - Error: %v", r.Method, r.URL.Path, err)
		} else {
			log.Printf("Request completed: %s %s - Status: %d", r.Method, r.URL.Path, rw.statusCode)
		}
		
		return err
	}
}

// Middleware example: Recovery middleware (like Fiber)
func RecoveryMiddleware(next Handler) Handler {
	return func(w http.ResponseWriter, r *http.Request) (err error) {
		defer func() {
			if r := recover(); r != nil {
				err = NewAppError(http.StatusInternalServerError, 
					"Internal server error", 
					r.(error))
			}
		}()
		
		return next(w, r)
	}
}

// Custom response writer to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Helper to create handlers with middleware chain
func (eh *ExtendedHandler) CreateHandler(baseHandler Handler, middlewares ...func(Handler) Handler) http.HandlerFunc {
	// Chain all middlewares
	handler := baseHandler
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	
	return ToHTTP(handler)
}

// Example usage in main.go would be:
// eh := wrapper.NewExtendedHandler(tmpl)
// r.Get("/flux1-dev-t2i", eh.CreateHandler(
//     eh.Flux1DevT2IHandlerWithValidation(),
//     wrapper.LoggingMiddleware,
//     wrapper.RecoveryMiddleware,
// ))