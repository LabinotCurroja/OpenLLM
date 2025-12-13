"""
Qwen3-4B OpenAI-Compatible API Server
======================================
A simple HTTP server that provides an OpenAI-compatible /v1/chat/completions endpoint.
Supports streaming via Server-Sent Events (SSE).
"""

import json
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Generator

import torch
import torch.nn.functional as F
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

from qwen3_pytorch import (
    Qwen3Config,
    Qwen3ForCausalLM,
    KVCache,
    download_model,
    load_weights,
)
from transformers import AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "qwen3-4b"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50

# ============================================================================
# Model Loading
# ============================================================================

print("=" * 60)
print("🚀 Qwen3-4B API Server")
print("=" * 60)
print()

def load_model():
    """Load the model."""
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"📱 Using device: {device}")
    dtype = torch.bfloat16
    print(f"📊 Using dtype: {dtype}")
    print()
    
    # Download model
    print("⬇️  Downloading model (if needed)...")
    model_path = download_model("Qwen/Qwen3-4B")
    
    # Load config
    print("⚙️  Loading configuration...")
    config = Qwen3Config.from_pretrained(model_path)
    
    # Create model
    print("🏗️  Creating model architecture...")
    model = Qwen3ForCausalLM(config)
    
    # Load weights
    print("💾 Loading weights (~8GB)...")
    model = load_weights(model, model_path, device, dtype)
    model.eval()
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print()
    print("✅ Model loaded successfully!")
    print()
    
    return model, tokenizer, device


MODEL, TOKENIZER, DEVICE = load_model()

# ============================================================================
# Token Generation
# ============================================================================

def generate_tokens(
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    stop: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """Generator that yields tokens one by one for streaming."""
    model = MODEL
    tokenizer = TOKENIZER
    device = DEVICE
    config = model.config
    dtype = next(model.parameters()).dtype
    
    # Use an incremental decoder to properly handle multi-byte UTF-8 characters
    # that may be split across multiple tokens
    token_buffer = []
    
    with torch.inference_mode():
        # Format as chat using the tokenizer's chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        
        # Pre-allocate KV cache for the entire generation
        total_len = prompt_len + max_tokens
        kv_cache = KVCache(
            batch_size=1,
            max_seq_len=total_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            num_layers=config.num_hidden_layers,
            device=device,
            dtype=dtype,
        )
        
        # Prefill: process the entire prompt
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        logits = model(input_ids, position_ids=position_ids, kv_cache=kv_cache)
        kv_cache.advance(prompt_len)
        
        # Get next token from last position
        next_token_logits = logits[:, -1, :]
        
        current_pos = prompt_len
        generated_text = ""
        
        for _ in range(max_tokens):
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_id = next_token.item()
            
            # Check for EOS
            if token_id == tokenizer.eos_token_id:
                # Flush any remaining tokens in buffer before ending
                if token_buffer:
                    final_text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                    if final_text:
                        generated_text += final_text
                        yield final_text
                break
            
            # Add token to buffer and decode incrementally
            # This handles multi-byte UTF-8 characters that span multiple tokens
            token_buffer.append(token_id)
            
            # Try to decode the buffer - if it produces valid text, yield it
            # Keep a sliding window to handle incomplete sequences
            decoded_text = tokenizer.decode(token_buffer, skip_special_tokens=True)
            
            # Check if the decoded text ends with a replacement character (incomplete UTF-8)
            # If so, buffer more tokens before yielding
            if decoded_text and not decoded_text.endswith('\ufffd') and not decoded_text.endswith('�'):
                # Successfully decoded - yield and clear buffer
                generated_text += decoded_text
                yield decoded_text
                token_buffer = []
                
                # Check for stop sequences
                if stop:
                    for stop_seq in stop:
                        if stop_seq in generated_text:
                            return
            elif len(token_buffer) > 10:
                # Safety limit - if buffer gets too large, force decode and yield
                # This prevents infinite buffering
                generated_text += decoded_text
                yield decoded_text
                token_buffer = []
            
            # Decode step: only process the new token with KV cache
            position_ids = torch.tensor([[current_pos]], device=device)
            logits = model(next_token, position_ids=position_ids, kv_cache=kv_cache)
            kv_cache.advance(1)
            next_token_logits = logits[:, -1, :]
            current_pos += 1
        
        # Flush any remaining tokens in buffer after loop ends (max_tokens reached)
        if token_buffer:
            final_text = tokenizer.decode(token_buffer, skip_special_tokens=True)
            if final_text:
                yield final_text


def generate_full(
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    stop: Optional[List[str]] = None,
) -> str:
    """Generate complete response."""
    tokens = list(generate_tokens(messages, max_tokens, temperature, top_p, top_k, stop))
    return "".join(tokens)


# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)
CORS(app)


def create_chat_completion_response(
    content: str,
    model: str = MODEL_NAME,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """Create a non-streaming chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": -1,  # Not tracked
            "completion_tokens": -1,  # Not tracked
            "total_tokens": -1,  # Not tracked
        },
    }


def create_chat_completion_chunk(
    content: str,
    model: str = MODEL_NAME,
    finish_reason: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a streaming chat completion chunk."""
    delta = {"content": content} if content else {}
    if finish_reason is None and content == "":
        delta = {"role": "assistant", "content": ""}
    
    return {
        "id": chunk_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def stream_response(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
) -> Generator[str, None, None]:
    """Stream SSE response."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    # Send initial chunk with role
    initial_chunk = create_chat_completion_chunk("", chunk_id=chunk_id)
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    
    # Stream tokens
    for token in generate_tokens(messages, max_tokens, temperature, top_p, stop=stop):
        chunk = create_chat_completion_chunk(token, chunk_id=chunk_id)
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send final chunk
    final_chunk = create_chat_completion_chunk("", finish_reason="stop", chunk_id=chunk_id)
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        
        # Extract parameters
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = data.get("temperature", DEFAULT_TEMPERATURE)
        top_p = data.get("top_p", DEFAULT_TOP_P)
        stop = data.get("stop")
        
        if isinstance(stop, str):
            stop = [stop]
        
        if not messages:
            return jsonify({"error": "messages is required"}), 400
        
        if stream:
            # Streaming response
            return Response(
                stream_response(messages, max_tokens, temperature, top_p, stop),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            content = generate_full(messages, max_tokens, temperature, top_p, stop=stop)
            response = create_chat_completion_response(content)
            return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/ping", methods=["GET"])
def ping():
    """Ping endpoint to check if server is ready."""
    try:
        # Check if model is loaded and ready
        if MODEL is not None and TOKENIZER is not None:
            return jsonify({"status": "success", "message": "Server is ready"})
        else:
            return jsonify({"status": "error", "message": "Model not loaded"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """Index page."""
    return jsonify({
        "message": "Qwen3-4B API Server",
        "endpoints": {
            "/v1/chat/completions": "OpenAI-compatible chat completions",
            "/v1/models": "List available models",
            "/health": "Health check",
            "/ping": "Check if server is ready",
        },
    })


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-4B API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    print(f"📚 API docs: http://{args.host}:{args.port}/")
    print()
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
