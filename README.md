# OpenLLM - Pure PyTorch LLM Inference

<p align="center">
  <img src="assets/dragon.png" width="200" alt="OpenLLM">
</p>

A **pure PyTorch implementation** for running large language models locally. No dependencies on heavy inference frameworks - just PyTorch and your hardware.

Currently supports **Qwen3-4B**, optimized for Apple Silicon Macs but also works on CUDA GPUs and CPU.

## ✨ Features

- **Pure PyTorch** - No llama.cpp, vLLM, or other inference frameworks required
- **Apple Silicon Optimized** - First-class MPS (Metal Performance Shaders) support
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API clients
- **Streaming Support** - Real-time token streaming via Server-Sent Events
- **Beautiful TUI** - Terminal user interface for interactive chat
- **Optimized Inference** - Pre-allocated KV cache, torch.compile, and more

## 🏗️ Architecture

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

## 🚀 Quick Start

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
python chat_tui.py
```

This launches an interactive terminal interface for chatting with the model.

### Run the API Server

```bash
python server.py
```

This starts an OpenAI-compatible API server on `http://localhost:5001`.

#### API Usage

```bash
# Chat completion (streaming)
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
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

## ⚡ Performance Optimizations

### 1. Pre-allocated KV Cache

The naive approach uses `torch.cat` every token, which is O(n²):

```python
# ❌ Slow: O(n²) memory operations
k = torch.cat([cached_k, k], dim=2)  # Copies ALL previous tokens
```

Our implementation pre-allocates the entire cache upfront:

```python
# ✅ Fast: O(1) per token
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

## 📊 Memory Usage

| Configuration | Memory | Notes |
|--------------|--------|-------|
| bfloat16 (default) | ~8GB | Recommended for M1 Pro+ |
| float16 | ~8GB | Slightly faster on some hardware |
| float32 | ~16GB | Won't fit on most Macs |

The KV cache adds additional memory during generation:
- ~50MB per 1000 tokens of context
- Pre-allocated for `max_new_tokens` (default 2048)

## 🔧 Configuration

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

## 📁 Project Structure

```
OpenLLM/
├── qwen3_pytorch.py   # Core model implementation
├── server.py          # OpenAI-compatible API server
├── chat_tui.py        # Terminal UI for interactive chat
├── tui.py             # Alternative TUI implementation
├── requirements.txt   # Python dependencies
├── assets/
│   └── dragon.png     # Logo
└── README.md
```

## 🛠️ Extending to Other Models

The implementation is designed to be adaptable. To add a new model:

1. **Update `Qwen3Config`** with the new model's configuration
2. **Adjust layer implementations** if architecture differs
3. **Update weight mapping** in `load_weights()` if needed
4. **Update tokenizer loading** for the new model

The core components (RMSNorm, RoPE, GQA, SwiGLU, KVCache) are reusable across most modern LLMs.

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the Qwen3 model
- [Hugging Face](https://huggingface.co/) for model hosting and tokenizers
- [PyTorch](https://pytorch.org/) for the amazing framework

## 📄 License

MIT License - feel free to use, modify, and distribute.

---

**Note**: This is an educational implementation focused on clarity and understanding. For production workloads with maximum performance, consider using optimized inference frameworks like vLLM, TensorRT-LLM, or llama.cpp.
