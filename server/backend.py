"""
Qwen3-4B Backend Abstraction
=============================
Provides a unified interface for both MLX (Apple Silicon) and PyTorch backends.
Automatically selects the best available backend.
"""

import platform
import sys
from pathlib import Path
from typing import Optional, List, Dict, Generator, Any
import os

# Add inference folder to path (sibling directory)
_server_dir = Path(__file__).parent
_project_root = _server_dir.parent
_inference_dir = _project_root / "inference"
if str(_inference_dir) not in sys.path:
    sys.path.insert(0, str(_inference_dir))

# ============================================================================
# Backend Detection
# ============================================================================

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return (
        platform.system() == "Darwin" and 
        platform.machine() == "arm64"
    )


def get_available_backend() -> str:
    """
    Determine the best available backend.
    
    Returns:
        'mlx' for Apple Silicon with MLX installed
        'pytorch' for CUDA or CPU
    """
    # Check for environment variable override
    force_backend = os.environ.get("OPENLLM_BACKEND", "").lower()
    if force_backend in ("mlx", "pytorch"):
        return force_backend
    
    # Auto-detect based on platform
    if is_apple_silicon():
        try:
            import mlx.core
            import mlx_lm
            return "mlx"
        except ImportError:
            print("⚠️  MLX not installed. Using PyTorch backend.")
            print("   For better performance, install MLX: pip install mlx mlx-lm")
            return "pytorch"
    
    return "pytorch"


# ============================================================================
# Backend Interface
# ============================================================================

class LLMBackend:
    """Abstract interface for LLM backends."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self.backend_name = "unknown"
    
    def load(self) -> None:
        """Load the model and tokenizer."""
        raise NotImplementedError
    
    def generate_tokens(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 20,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Generate tokens for the given messages (streaming)."""
        raise NotImplementedError
    
    def generate_full(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 20,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate complete response (non-streaming)."""
        return "".join(self.generate_tokens(messages, max_tokens, temperature, top_p, top_k, stop))


# ============================================================================
# MLX Backend (Apple Silicon)
# ============================================================================

class MLXBackend(LLMBackend):
    """MLX backend for Apple Silicon with INT4 quantization."""
    
    def __init__(self, use_thinking: bool = True):
        super().__init__()
        self.backend_name = "mlx"
        self.use_thinking = use_thinking
    
    def load(self) -> None:
        """Load the MLX model."""
        from qwen3_mlx import Qwen3MLX, DEFAULT_MODEL, THINKING_MODEL
        
        print("🍎 Loading MLX backend (INT4 quantized)")
        
        qwen = Qwen3MLX.from_pretrained(use_thinking=self.use_thinking)
        self.model = qwen.model
        self.tokenizer = qwen.tokenizer
        self.config = qwen.config
        self.device = "mlx"
        
        print(f"✅ MLX model loaded (~2GB memory)")
    
    def generate_tokens(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 20,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Generate tokens using MLX."""
        from qwen3_mlx import Qwen3MLX, generate_tokens_mlx
        
        qwen = Qwen3MLX(self.model, self.tokenizer, self.config)
        
        yield from generate_tokens_mlx(
            qwen,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )


# ============================================================================
# PyTorch Backend (CUDA, MPS, CPU)
# ============================================================================

class PyTorchBackend(LLMBackend):
    """PyTorch backend for CUDA, MPS, or CPU."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        super().__init__()
        self.backend_name = "pytorch"
        self.model_name = model_name
        self.dtype = None
    
    def load(self) -> None:
        """Load the PyTorch model."""
        import torch
        from qwen3_pytorch import (
            Qwen3Config,
            Qwen3ForCausalLM,
            download_model,
            load_weights,
        )
        from transformers import AutoTokenizer
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.dtype = torch.bfloat16
        
        print(f"🔥 Loading PyTorch backend on {self.device}")
        
        # Download model
        model_path = download_model(self.model_name)
        
        # Load config
        config = Qwen3Config.from_pretrained(model_path)
        self.config = config
        
        # Create and load model
        model = Qwen3ForCausalLM(config)
        model = load_weights(model, model_path, self.device, self.dtype)
        model.eval()
        self.model = model
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"✅ PyTorch model loaded (~8GB memory)")
    
    def generate_tokens(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 20,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Generate tokens using PyTorch."""
        import torch
        import torch.nn.functional as F
        from qwen3_pytorch import KVCache
        
        model = self.model
        tokenizer = self.tokenizer
        device = self.device
        config = self.config
        dtype = self.dtype
        
        token_buffer = []
        
        with torch.inference_mode():
            # Format as chat
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            prompt_len = input_ids.shape[1]
            
            # Pre-allocate KV cache
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
            
            # Prefill
            position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
            logits = model(input_ids, position_ids=position_ids, kv_cache=kv_cache)
            kv_cache.advance(prompt_len)
            
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
                
                # Apply top-p
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
                    if token_buffer:
                        final_text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                        if final_text:
                            yield final_text
                    break
                
                # Add token to buffer and decode incrementally
                token_buffer.append(token_id)
                decoded_text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                
                if decoded_text and not decoded_text.endswith('\ufffd') and not decoded_text.endswith('�'):
                    generated_text += decoded_text
                    yield decoded_text
                    token_buffer = []
                    
                    # Check stop sequences
                    if stop:
                        for stop_seq in stop:
                            if stop_seq in generated_text:
                                return
                elif len(token_buffer) > 10:
                    generated_text += decoded_text
                    yield decoded_text
                    token_buffer = []
                
                # Decode step
                position_ids = torch.tensor([[current_pos]], device=device)
                logits = model(next_token, position_ids=position_ids, kv_cache=kv_cache)
                kv_cache.advance(1)
                next_token_logits = logits[:, -1, :]
                current_pos += 1
            
            # Flush remaining tokens
            if token_buffer:
                final_text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                if final_text:
                    yield final_text


# ============================================================================
# Factory Function
# ============================================================================

def create_backend(use_thinking: bool = True) -> LLMBackend:
    """
    Create the appropriate backend based on the platform.
    
    Args:
        use_thinking: Whether to use the thinking variant
    
    Returns:
        LLMBackend instance (MLX or PyTorch)
    """
    backend_type = get_available_backend()
    
    if backend_type == "mlx":
        return MLXBackend(use_thinking=use_thinking)
    else:
        model_name = "Qwen/Qwen3-4B-Thinking-2507" if use_thinking else "Qwen/Qwen3-4B"
        return PyTorchBackend(model_name=model_name)


# ============================================================================
# Global Backend Instance
# ============================================================================

_backend: Optional[LLMBackend] = None


def get_backend() -> LLMBackend:
    """Get the global backend instance, loading if necessary."""
    global _backend
    if _backend is None:
        _backend = create_backend()
        _backend.load()
    return _backend


def load_backend(use_thinking: bool = True) -> LLMBackend:
    """Load and return the backend."""
    global _backend
    _backend = create_backend(use_thinking=use_thinking)
    _backend.load()
    return _backend
