# OpenLLM - Local AI for Apple Silicon

<p align="center">
  <img src="assets/dragon.png" width="200" alt="OpenLLM">
</p>

Run **Qwen3-4B** locally on your Mac with full privacy. Your data never leaves your machine - no API calls, no cloud dependencies, just you and your hardware.

This project uses **MLX** with **INT4 quantization** for efficient inference on Apple Silicon, requiring only ~2GB of memory.

## Features

- **Apple Silicon Native** - Built with MLX for optimal M1/M2/M3/M4/M5 performance
- **INT4 Quantization** - 4-bit quantization reduces memory from ~8GB to ~2GB
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API clients
- **Tool Calling** - Web search support via Brave Search API (free tier)
- **Streaming Support** - Real-time token streaming via Server-Sent Events
- **Beautiful TUI** - Terminal user interface for interactive chat

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4/M5) - Intel Macs are not supported
- **macOS 13.3+** (Ventura or later)
- **8GB+ RAM** (16GB+ recommended)
- **Python 3.9+**

## Quick Start

### Installation

```bash
git clone https://github.com/LabinotCurroja/OpenLLM.git
cd OpenLLM
pip install -r requirements.txt
```

### Run the Chat TUI

```bash
python tui.py
```

This launches an interactive terminal interface for chatting with the model.

### Run the API Server

```bash
python server/server.py
```

This starts an OpenAI-compatible API server on `http://localhost:8000`.

## Tool Calling (Web Search)

The model supports tool calling with web search capabilities. When enabled, the model can search the web for current information.

### Setup (Free)

1. Get a free Brave Search API key (2,000 queries/month, no credit card):
   - Go to https://brave.com/search/api/
   - Sign up and create an API key

2. Set the environment variable:
   ```bash
   export BRAVE_API_KEY="your-api-key-here"
   ```

3. Start the server - tool calling is enabled automatically:
   ```bash
   python server/server.py
   ```

### Available Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for current information |
| `get_current_time` | Get the current date and time |

## API Usage

```bash
# Chat completion (streaming)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Chat completion with tool calling
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "What is the weather like today?"}],
    "stream": true,
    "use_tools": true
  }'
```

### Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Memory Usage

| Configuration | Memory | Notes |
|--------------|--------|-------|
| INT4 (default) | ~2GB | MLX with 4-bit quantization |

The model uses pre-quantized INT4 weights from the MLX Community:
- `mlx-community/Qwen3-4B-4bit` - Base model
- `mlx-community/Qwen3-4B-Thinking-2507-4bit` - Thinking variant (default)

## How It Works

MLX is Apple's framework for machine learning on Apple Silicon. It provides:

- **Native Metal support** - Direct GPU acceleration without adapters
- **Unified memory** - Efficient data sharing between CPU and GPU
- **INT4 quantization** - 4-bit integer weights for 4x memory reduction

The model runs entirely on your Mac's GPU via Metal, providing fast inference with minimal memory footprint.

## Project Structure

```
OpenLLM/
├── tui.py             # Terminal UI (run this)
├── requirements.txt   # Python dependencies
├── server/
│   ├── server.py      # API server (run this)
│   ├── backend.py     # MLX backend with system checks
│   └── tools.py       # Tool calling (web search)
└── inference/
    └── qwen3_mlx.py   # MLX inference implementation
```

## Troubleshooting

### "This application requires Apple Silicon"

This project only works on Apple Silicon Macs (M1/M2/M3/M4/M5). Intel Macs are not supported because MLX requires Apple Silicon.

### "MLX is not installed"

Install MLX with:
```bash
pip install mlx mlx-lm
```

### "Insufficient memory"

The model requires at least 8GB of RAM. Close other applications to free up memory. 16GB+ is recommended for best performance.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the Qwen3 model
- [MLX Community](https://huggingface.co/mlx-community) for quantized models
- [Apple MLX](https://github.com/ml-explore/mlx) for the ML framework
- [Hugging Face](https://huggingface.co/) for model hosting

## License

MIT License - feel free to use, modify, and distribute.

---

**Why OpenLLM?** Run AI locally with complete privacy. No API calls, no data leaving your machine, no vendor lock-in. Just pure local inference on Apple Silicon.
