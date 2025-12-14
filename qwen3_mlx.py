"""
Qwen3-4B MLX Implementation (INT4 Quantized)
=============================================
This uses Apple's MLX framework for optimized inference on Apple Silicon.
INT4 quantization reduces memory from ~8GB to ~2GB and improves speed.

Benefits over PyTorch:
- Native Apple Silicon support (no MPS adapter layer)
- INT4 quantization built-in (~4x memory reduction)
- Unified memory architecture optimization
- 2-3x faster token generation

Requirements:
- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13.3+
- pip install mlx mlx-lm
"""

import platform
from typing import Optional, List, Dict, Generator, Any
from pathlib import Path

# Check if we're on Apple Silicon
def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return (
        platform.system() == "Darwin" and 
        platform.machine() == "arm64"
    )

if not is_apple_silicon():
    raise ImportError(
        "MLX requires Apple Silicon (M1/M2/M3/M4). "
        "Use qwen3_pytorch.py for other platforms."
    )

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# ============================================================================
# Configuration
# ============================================================================

# Pre-quantized 4-bit model from MLX community
# This is Qwen3-4B quantized to INT4 (~2GB instead of ~8GB)
DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"

# For the thinking variant (if available)
THINKING_MODEL = "mlx-community/Qwen3-4B-Thinking-2507-4bit"


# ============================================================================
# Model Wrapper (Compatible with PyTorch interface)
# ============================================================================

class MLXConfig:
    """Configuration wrapper to match PyTorch interface."""
    def __init__(self, model_config):
        self._config = model_config
        # Extract key config values for compatibility
        self.vocab_size = getattr(model_config, 'vocab_size', 151936)
        self.hidden_size = getattr(model_config, 'hidden_size', 2560)
        self.num_hidden_layers = getattr(model_config, 'num_hidden_layers', 36)
        self.num_attention_heads = getattr(model_config, 'num_attention_heads', 32)
        self.num_key_value_heads = getattr(model_config, 'num_key_value_heads', 8)
        self.head_dim = getattr(model_config, 'head_dim', 128)
        self.max_position_embeddings = getattr(model_config, 'max_position_embeddings', 32768)


class Qwen3MLX:
    """
    MLX-based Qwen3 model wrapper.
    Provides similar interface to the PyTorch implementation.
    """
    
    def __init__(self, model, tokenizer, config: MLXConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name: str = DEFAULT_MODEL,
        use_thinking: bool = True
    ) -> "Qwen3MLX":
        """
        Load a pre-quantized INT4 model from MLX community.
        
        Args:
            model_name: HuggingFace model ID (should be MLX-compatible)
            use_thinking: Whether to use the thinking variant
        
        Returns:
            Qwen3MLX instance ready for inference
        """
        # Try thinking model first if requested
        if use_thinking:
            try:
                print(f"Loading {THINKING_MODEL}...")
                model, tokenizer = load(THINKING_MODEL)
                config = MLXConfig(model.args)
                print(f"✅ Loaded thinking model: {THINKING_MODEL}")
                return cls(model, tokenizer, config)
            except Exception as e:
                print(f"Thinking model not available ({e}), falling back to base model")
        
        # Fall back to base model
        print(f"Loading {model_name}...")
        model, tokenizer = load(model_name)
        config = MLXConfig(model.args)
        print(f"✅ Loaded model: {model_name}")
        
        return cls(model, tokenizer, config)


# ============================================================================
# Download Function (Compatible interface)
# ============================================================================

def download_model(model_name: str = DEFAULT_MODEL) -> str:
    """
    Download model (MLX handles this automatically during load).
    Returns model name for compatibility.
    """
    print(f"Model will be downloaded on first load: {model_name}")
    return model_name


# ============================================================================
# Text Generation (Streaming)
# ============================================================================

def generate_tokens_mlx(
    model: Qwen3MLX,
    messages: List[Dict[str, str]],
    max_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    stop: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """
    Generate tokens one by one for streaming output.
    
    Args:
        model: Qwen3MLX instance
        messages: Chat messages in OpenAI format
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        repetition_penalty: Penalty for repeated tokens
        stop: Stop sequences
    
    Yields:
        Generated text chunks
    """
    # Apply chat template
    prompt = model.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    generated_text = ""
    
    # Create sampler with temperature and top_p
    sampler = make_sampler(temp=temperature, top_p=top_p)
    
    # Use mlx_lm's stream_generate for efficient streaming
    for response in stream_generate(
        model.model,
        model.tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        # response.text contains the newly generated text segment
        chunk = response.text
        
        if chunk:
            generated_text += chunk
            yield chunk
            
            # Check stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        return


def generate_full_mlx(
    model: Qwen3MLX,
    messages: List[Dict[str, str]],
    max_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None,
) -> str:
    """Generate complete response (non-streaming)."""
    tokens = list(generate_tokens_mlx(model, messages, max_tokens, temperature, top_p, stop=stop))
    return "".join(tokens)


# ============================================================================
# Simple Generate Function (for quick use)
# ============================================================================

def generate_simple(
    model: Qwen3MLX,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """
    Simple text generation with a prompt.
    
    Args:
        model: Qwen3MLX instance
        prompt: User prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: System prompt
    
    Returns:
        Generated text
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return generate_full_mlx(model, messages, max_tokens, temperature)


# ============================================================================
# Streaming Generate (prints to console)
# ============================================================================

def generate_streaming(
    model: Qwen3MLX,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant developed by Mathematica.",
):
    """
    Generate text with streaming output to console.
    Compatible with PyTorch interface.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    for chunk in generate_tokens_mlx(model, messages, max_tokens, temperature):
        print(chunk, end="", flush=True)
    
    print()  # Final newline


# ============================================================================
# Main (Interactive Chat)
# ============================================================================

def main():
    print("=" * 60)
    print("Qwen3-4B MLX Implementation (INT4 Quantized)")
    print("=" * 60)
    print()
    print("🍎 Running on Apple Silicon with MLX")
    print("📊 Using INT4 quantization (~2GB memory)")
    print()
    
    # Load model
    print("Loading model...")
    model = Qwen3MLX.from_pretrained(use_thinking=True)
    
    print()
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Attention heads: {model.config.num_attention_heads} Q, {model.config.num_key_value_heads} KV")
    
    # Interactive chat
    print()
    print("=" * 60)
    print("Chat with Qwen3-4B (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nQwen3: ", end="", flush=True)
            generate_streaming(model, user_input)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
