package handlers

type HandlersPythonParams struct {
	Prompt          string  `form:"prompt"`
	Suffix          string  `form:"suffix"`
	Model           string  `form:"model"`
	AspectRatio     string  `form:"aspect_ratio"`
	Steps           int     `form:"steps"`
	Guidance        float64 `form:"guidance"`
	StylePreset     string  `form:"style_preset"`
	CameraShot      string  `form:"camera_shot"`
	LowVRAM         bool    `form:"low_vram"` // Add this field
	Seed            int64   `form:"seed"`
	BatchSize       int     `form:"batch_size"`
	NegativePrompt  string  `form:"negative_prompt"`
	LoraEnabled     bool    `form:"lora_enabled"`
	LoraURL         string  `form:"lora_url"`
	LoraAdapterName string  `form:"lora_adapter_name"`
	HFToken         string  `form:"hf_token"` // Add this field
	StaticSeed      string  `form:"static_seed"`
	GPUDevices      string  `form:"gpu_devices"` // Add this field
}

type PromptData struct {
	Prompt   string `json:"prompt"`
	Filename string `json:"filename"`
}

type QwenT2IRequest struct {
	HandlersPythonParams
	QuantMode string `json:"quant_mode"`
}
type FluxT2IRequest struct {
	GGUFEnabled  bool   `form:"gguf_enabled"`      // Add this field
	GGUFURL      string `form:"gguf_url"`          // Add this field
	GGUFNGLayers int    `form:"gguf_n_gpu_layers"` // Add this field
	GGUFNThreads int    `form:"gguf_n_threads"`    // Add this field
	QuantMode    string `json:"quant_mode"`

	HandlersPythonParams
}
