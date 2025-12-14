"""
Qwen3-4B OpenAI-Compatible API Server (Mac Only)
=================================================
A simple HTTP server that provides an OpenAI-compatible /v1/chat/completions endpoint.
Supports streaming via Server-Sent Events (SSE).
Includes tool calling support with web search.

Requirements:
- Apple Silicon Mac (M1/M2/M3/M4/M5)
- macOS 13.3+
- MLX framework (pip install mlx mlx-lm)
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

# Add inference folder to path
_server_dir = Path(__file__).parent
_project_root = _server_dir.parent
_inference_dir = _project_root / "inference"
if str(_inference_dir) not in sys.path:
    sys.path.insert(0, str(_inference_dir))

from flask import Flask, request, Response, jsonify
from flask_cors import CORS

from backend import check_system_requirements, print_system_info
from tools import (
    AVAILABLE_TOOLS,
    get_tools_system_prompt,
    parse_tool_calls,
    has_tool_calls,
    execute_tool,
    format_tool_result,
)

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "qwen3-4b-thinking"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 20
ENABLE_TOOLS = True  # Enable tool calling by default

# ============================================================================
# System Requirements Check
# ============================================================================

print("=" * 60)
print("🚀 Qwen3-4B API Server (Apple Silicon)")
print("=" * 60)
print()

# Check system requirements before loading model
success, message = check_system_requirements()
if not success:
    print(message)
    sys.exit(1)

print(message)
print()

# ============================================================================
# Model Loading (MLX Only)
# ============================================================================

from qwen3_mlx import Qwen3MLX, generate_tokens_mlx

def load_model():
    """Load the MLX model."""
    print("🍎 Using MLX backend (Apple Silicon)")
    print("📊 INT4 quantization (~2GB memory)")
    print()
    
    model = Qwen3MLX.from_pretrained(use_thinking=True)
    
    print()
    print("✅ Model loaded successfully!")
    print()
    
    return model

MODEL = load_model()
TOKENIZER = MODEL.tokenizer
DEVICE = "mlx"


def generate_tokens(
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    stop: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """Generator that yields tokens one by one for streaming (MLX)."""
    yield from generate_tokens_mlx(
        MODEL,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )


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


def generate_with_tools(
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    stop: Optional[List[str]] = None,
    max_tool_rounds: int = 3,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate response with tool calling support.
    
    Yields events:
    - {"type": "token", "content": str} - Regular token
    - {"type": "tool_call", "name": str, "arguments": dict} - Tool being called
    - {"type": "tool_result", "name": str, "result": dict} - Tool result
    - {"type": "done"} - Generation complete
    """
    import re
    
    # Inject tools system prompt into the first system message or prepend it
    augmented_messages = list(messages)
    tools_prompt = get_tools_system_prompt()
    
    has_system = any(m.get("role") == "system" for m in augmented_messages)
    if has_system:
        for i, m in enumerate(augmented_messages):
            if m.get("role") == "system":
                augmented_messages[i] = {
                    "role": "system",
                    "content": m["content"] + "\n\n" + tools_prompt
                }
                break
    else:
        augmented_messages.insert(0, {"role": "system", "content": tools_prompt})
    
    tool_round = 0
    
    while tool_round < max_tool_rounds:
        # Generate response
        tool_stop = list(stop) if stop else []
        if "</tool_call>" not in tool_stop:
            tool_stop.append("</tool_call>")
        
        full_response = ""
        in_tool_call = False
        in_json_tool_call = False
        pending_yield = ""
        json_brace_depth = 0
        
        for token in generate_tokens(augmented_messages, max_tokens, temperature, top_p, top_k, tool_stop):
            full_response += token
            
            if in_tool_call or in_json_tool_call:
                if in_json_tool_call:
                    for c in token:
                        if c == '{':
                            json_brace_depth += 1
                        elif c == '}':
                            json_brace_depth -= 1
                            if json_brace_depth == 0:
                                break
                continue
            
            pending_yield += token
            
            if "<tool_call>" in pending_yield:
                in_tool_call = True
                before_tool = pending_yield.split("<tool_call>")[0]
                if before_tool:
                    yield {"type": "token", "content": before_tool}
                pending_yield = ""
                continue
            
            json_tool_match = re.search(r'\{"name"\s*:\s*"(web_search|get_current_time)"', pending_yield)
            if json_tool_match:
                in_json_tool_call = True
                json_brace_depth = 1
                before_json = pending_yield[:json_tool_match.start()]
                if before_json:
                    yield {"type": "token", "content": before_json}
                pending_yield = ""
                continue
            
            potential_tag_starts = ["<", "<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call"]
            potential_json_starts = ["{", '{"', '{"n', '{"na', '{"nam', '{"name', '{"name"', '{"name":']
            
            should_buffer = False
            for partial in potential_tag_starts + potential_json_starts:
                if pending_yield.endswith(partial):
                    should_buffer = True
                    break
            
            if not should_buffer:
                if pending_yield:
                    yield {"type": "token", "content": pending_yield}
                    pending_yield = ""
        
        if pending_yield and not in_tool_call and not in_json_tool_call:
            yield {"type": "token", "content": pending_yield}
        
        if "<tool_call>" in full_response and "</tool_call>" not in full_response:
            full_response += "</tool_call>"
        
        if has_tool_calls(full_response):
            tool_calls = parse_tool_calls(full_response)
            
            if not tool_calls:
                break
            
            augmented_messages.append({
                "role": "assistant",
                "content": full_response
            })
            
            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call["arguments"]
                
                yield {"type": "tool_call", "name": tool_name, "arguments": tool_args}
                
                result = execute_tool(tool_name, tool_args)
                
                yield {"type": "tool_result", "name": tool_name, "result": result}
                
                formatted_result = format_tool_result(tool_name, result)
                augmented_messages.append({
                    "role": "user",
                    "content": f"Here are the tool results. Please use this information to provide a helpful response to my original question:\n{formatted_result}"
                })
            
            tool_round += 1
        else:
            break
    
    yield {"type": "done"}


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
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
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
    use_tools: bool = False,
) -> Generator[str, None, None]:
    """Stream SSE response."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    initial_chunk = create_chat_completion_chunk("", chunk_id=chunk_id)
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    
    if use_tools and ENABLE_TOOLS:
        for event in generate_with_tools(messages, max_tokens, temperature, top_p, stop=stop):
            if event["type"] == "token":
                chunk = create_chat_completion_chunk(event["content"], chunk_id=chunk_id)
                yield f"data: {json.dumps(chunk)}\n\n"
            elif event["type"] == "tool_call":
                tool_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "function": {
                                    "name": event["name"],
                                    "arguments": json.dumps(event["arguments"])
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(tool_chunk)}\n\n"
            elif event["type"] == "tool_result":
                query = ""
                if isinstance(event.get("result"), dict):
                    query = event["result"].get("query", "")
                if query:
                    result_text = f"\n[Searching: {query}]\n"
                else:
                    result_text = f"\n[Searching: {event['name']}]\n"
                chunk = create_chat_completion_chunk(result_text, chunk_id=chunk_id)
                yield f"data: {json.dumps(chunk)}\n\n"
                thinking_marker = "<think>"
                thinking_chunk = create_chat_completion_chunk(thinking_marker, chunk_id=chunk_id)
                yield f"data: {json.dumps(thinking_chunk)}\n\n"
    else:
        for token in generate_tokens(messages, max_tokens, temperature, top_p, stop=stop):
            chunk = create_chat_completion_chunk(token, chunk_id=chunk_id)
            yield f"data: {json.dumps(chunk)}\n\n"
    
    final_chunk = create_chat_completion_chunk("", finish_reason="stop", chunk_id=chunk_id)
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = data.get("temperature", DEFAULT_TEMPERATURE)
        top_p = data.get("top_p", DEFAULT_TOP_P)
        stop = data.get("stop")
        
        tools = data.get("tools")
        use_tools = data.get("use_tools", False)
        enable_tools = bool(tools) or use_tools
        
        if isinstance(stop, str):
            stop = [stop]
        
        if not messages:
            return jsonify({"error": "messages is required"}), 400
        
        if stream:
            return Response(
                stream_response(messages, max_tokens, temperature, top_p, stop, use_tools=enable_tools),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            if enable_tools:
                full_content = ""
                for event in generate_with_tools(messages, max_tokens, temperature, top_p, stop=stop):
                    if event["type"] == "token":
                        full_content += event["content"]
                response = create_chat_completion_response(full_content)
            else:
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
    return jsonify({"status": "ok", "model": MODEL_NAME, "tools_enabled": ENABLE_TOOLS})


@app.route("/v1/tools", methods=["GET"])
def list_tools():
    """List available tools."""
    brave_configured = bool(os.environ.get("BRAVE_API_KEY", ""))
    return jsonify({
        "tools": AVAILABLE_TOOLS,
        "tools_enabled": ENABLE_TOOLS,
        "search_configured": brave_configured,
        "setup_instructions": {
            "brave_search": "Set BRAVE_API_KEY environment variable. Get free key at https://brave.com/search/api/"
        }
    })


@app.route("/ping", methods=["GET"])
def ping():
    """Ping endpoint to check if server is ready."""
    try:
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
        "message": "Qwen3-4B API Server (Apple Silicon)",
        "backend": "MLX (INT4 quantized)",
        "endpoints": {
            "/v1/chat/completions": "OpenAI-compatible chat completions",
            "/v1/models": "List available models",
            "/v1/tools": "List available tools",
            "/health": "Health check",
            "/ping": "Check if server is ready",
        },
    })


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-4B API Server (Apple Silicon)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    print(f"📚 API docs: http://{args.host}:{args.port}/")
    print()
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
