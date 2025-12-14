# OpenLLM - Open AI Usage for Everyone

<p align="center">
  <img src="assets/dragon.png" width="200" alt="OpenLLM">
</p>

Run large language models locally with full privacy. Your data never leaves your machine - no API calls, no cloud dependencies, just you and your hardware.

Currently supports **Qwen3-4B**, optimized for Apple Silicon Macs but also works on CUDA GPUs and CPU.

## Features

- **Pure PyTorch** - No llama.cpp, vLLM, or other inference frameworks required
- **Apple Silicon Optimized** - First-class MPS (Metal Performance Shaders) support
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API clients
- **Tool Calling** - Web search support via Brave Search API (free tier)
- **Streaming Support** - Real-time token streaming via Server-Sent Events
- **Beautiful TUI** - Terminal user interface for interactive chat
- **Optimized Inference** - Pre-allocated KV cache, torch.compile, and more

## Architecture

The implementation includes all core transformer components built from scratch:

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3ForCausalLM                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Embedding Layer (151K vocab)            │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           36x Transformer Decoder Layers             │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  RMSNorm → Grouped Query Attention (GQA)    │    │   │
│  │  │           32 Q heads, 8 KV heads            │    │   │
│  │  │           + Rotary Position Embeddings      │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  RMSNorm → SwiGLU MLP (2560 → 9728 → 2560)  │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           RMSNorm → LM Head (weight tied)            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **RMSNorm** | Root Mean Square Layer Normalization (faster than LayerNorm) |
| **RoPE** | Rotary Position Embeddings with 1M base frequency for 32K context |
| **GQA** | Grouped Query Attention - 4:1 ratio (32 Q heads share 8 KV heads) |
| **SwiGLU** | Gated activation: `SiLU(gate) * up` for better gradient flow |
| **KV Cache** | Pre-allocated buffers for O(1) token generation |

## Quick Start

### Prerequisites

- Python 3.9+
- ~10GB RAM (8GB for model + overhead)
- macOS with Apple Silicon (M1/M2/M3/M4/M5) OR NVIDIA GPU with CUDA

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
python server.py
```

This starts an OpenAI-compatible API server on `http://localhost:5001`.

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
   python server.py
   ```

### Using Tool Calling

The model will automatically search when it needs current information:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5001/v1", api_key="not-needed")

# Ask something that needs current info
response = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "What are the latest developments in AI?"}],
    extra_body={"use_tools": True}  # Enable tool calling
)
```

### Available Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for current information |
| `get_current_time` | Get the current date and time |

### Check Tool Status

```bash
curl http://localhost:5001/v1/tools
```

## API Usage

```bash
# Chat completion (streaming)
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Chat completion with tool calling
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "What is the weather like today?"}],
    "stream": true,
    "use_tools": true
  }'

# Chat completion (non-streaming)
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

#### Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5001/v1",
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

### Run Direct Inference

```bash
python qwen3_pytorch.py
```

This runs the model directly with an interactive prompt.

## Performance Optimizations

### 1. Pre-allocated KV Cache

The naive approach uses `torch.cat` every token, which is O(n²):

```python
# Slow: O(n²) memory operations
k = torch.cat([cached_k, k], dim=2)  # Copies ALL previous tokens
```

Our implementation pre-allocates the entire cache upfront:

```python
# Fast: O(1) per token
class KVCache:
    def __init__(self, batch_size, max_seq_len, ...):
        # Allocate once for all layers
        self.k_cache = torch.zeros(num_layers, batch, heads, max_seq, dim)
        self.v_cache = torch.zeros(num_layers, batch, heads, max_seq, dim)
    
    def update(self, layer_idx, k, v):
        # Just write to the next slot - no copying!
        self.k_cache[layer_idx, :, :, pos:pos+seq_len, :] = k
```

**Result**: 2-3x faster for long generations (500+ tokens)

### 2. torch.compile (Experimental)

On supported hardware, the model is compiled for additional speedup:

```python
torch._dynamo.config.suppress_errors = True  # Graceful fallback
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
```

- First inference has ~30-60s compilation overhead
- Subsequent inferences are faster
- Falls back gracefully if ops aren't supported on MPS

### 3. Grouped Query Attention (GQA)

Instead of full multi-head attention (32 KV heads), Qwen3 uses only 8 KV heads shared across 32 query heads:

- **4x less KV cache memory** (8 heads vs 32)
- **Faster attention computation**
- Minimal quality impact

### 4. Flash Attention via PyTorch SDPA

We use PyTorch's `scaled_dot_product_attention` which automatically uses the most efficient implementation:

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

On MPS, this uses Metal-optimized kernels. On CUDA, it can use Flash Attention 2.

### 5. Weight Tying

The LM head shares weights with the embedding layer, saving ~400MB of memory:

```python
self.lm_head = lambda x: F.linear(x, self.model.embed_tokens.weight)
```

## Memory Usage

| Configuration | Memory | Notes |
|--------------|--------|-------|
| MLX INT4 (recommended on Mac) | ~2GB | Apple Silicon only, fastest |
| bfloat16 (default) | ~8GB | Recommended for M1 Pro+ |
| float16 | ~8GB | Slightly faster on some hardware |
| float32 | ~16GB | Won't fit on most Macs |

The KV cache adds additional memory during generation:
- ~50MB per 1000 tokens of context
- Pre-allocated for `max_new_tokens` (default 2048)

## MLX Backend (Apple Silicon)

For Apple Silicon Macs (M1/M2/M3/M4), we provide an MLX backend with INT4 quantization that offers significant improvements:

### Benefits

| Aspect | PyTorch (bfloat16) | MLX (INT4) |
|--------|-------------------|------------|
| **Memory** | ~8GB | ~2GB |
| **Speed** | Good (via MPS) | 2-3x faster |
| **Native** | Via Metal adapter | Native Apple Silicon |

### Installation

```bash
# Install MLX dependencies (Apple Silicon only)
pip install mlx mlx-lm
```

### Usage

The backend is auto-selected based on your platform. On Apple Silicon with MLX installed, it will automatically use the MLX backend.

```bash
# Auto-select best backend
python server.py

# Force a specific backend
OPENLLM_BACKEND=mlx python server.py
OPENLLM_BACKEND=pytorch python server.py

# Run MLX directly
python qwen3_mlx.py
```

### How It Works

MLX uses pre-quantized INT4 models from the MLX Community:
- `mlx-community/Qwen3-4B-4bit` - Base model (~2GB)
- `mlx-community/Qwen3-4B-Thinking-2507-4bit` - Thinking variant

The quantization is done at the weight level using 4-bit integers, reducing memory by 4x while maintaining quality through careful calibration.

## Configuration

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 2048 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Top-k sampling |

### Model Configuration (Qwen3-4B)

| Parameter | Value |
|-----------|-------|
| Hidden size | 2560 |
| Layers | 36 |
| Attention heads | 32 Q / 8 KV |
| Head dimension | 128 |
| Intermediate size | 9728 |
| Vocab size | 151,936 |
| Max context | 32,768 |
| RoPE base | 1,000,000 |

## Project Structure

```
OpenLLM/
├── qwen3_pytorch.py   # PyTorch model implementation
├── qwen3_mlx.py       # MLX model implementation (Apple Silicon)
├── backend.py         # Backend abstraction (auto-selects MLX/PyTorch)
├── server.py          # OpenAI-compatible API server
├── tui.py             # Terminal UI for interactive chat
├── tools.py           # Tool calling (web search, etc.)
├── requirements.txt   # Python dependencies
├── assets/
│   └── dragon.png     # Logo
└── README.md
```

## Extending to Other Models

The implementation is designed to be adaptable. To add a new model:

1. **Update `Qwen3Config`** with the new model's configuration
2. **Adjust layer implementations** if architecture differs
3. **Update weight mapping** in `load_weights()` if needed
4. **Update tokenizer loading** for the new model

The core components (RMSNorm, RoPE, GQA, SwiGLU, KVCache) are reusable across most modern LLMs.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the Qwen3 model
- [Hugging Face](https://huggingface.co/) for model hosting and tokenizers
- [PyTorch](https://pytorch.org/) for the amazing framework

## License

MIT License - feel free to use, modify, and distribute.

---

**Why OpenLLM?** This implementation gives you **full control** over your LLM inference with ~80% of the efficiency of state-of-the-art frameworks like vLLM or TensorRT-LLM. It's production-ready for many use cases: local development, privacy-sensitive applications, edge deployment, and anywhere you need to **keep your data on your own hardware**. No API calls, no data leaving your machine, no vendor lock-in.
