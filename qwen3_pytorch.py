"""
Qwen3-4B Pure PyTorch Implementation
=====================================
This implements the model architecture from scratch and loads HuggingFace weights.

Architecture:
- 36 transformer layers
- Grouped Query Attention (32 Q heads, 8 KV heads)
- SwiGLU MLP activation
- RMSNorm
- Rotary Position Embeddings (RoPE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

# ============================================================================
# Model Configuration
# ============================================================================

class Qwen3Config:
    """Configuration for Qwen3-4B model."""
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        head_dim: int = 128,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.head_dim = head_dim

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3Config":
        """Load config from HuggingFace model directory."""
        config_path = Path(model_path) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cos/sin cache if needed."""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for the given positions.
        x: (batch, num_heads, seq_len, head_dim)
        position_ids: (batch, seq_len)
        """
        seq_len = position_ids.max().item() + 1
        self._update_cache(seq_len, x.device, x.dtype)
        
        cos = self._cos_cached[position_ids]  # (batch, seq_len, head_dim)
        sin = self._sin_cached[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K."""
    # q, k: (batch, num_heads, seq_len, head_dim)
    # cos, sin: (batch, seq_len, head_dim)
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# KV Cache with Pre-allocation
# ============================================================================

class KVCache:
    """
    Pre-allocated KV cache for efficient autoregressive generation.
    Avoids O(n²) memory copies from torch.cat by using fixed buffers.
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_seq_len = max_seq_len
        self.current_len = 0
        
        # Pre-allocate buffers for all layers
        # Shape: (batch, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer and return the full cached K, V.
        k, v: (batch, num_kv_heads, seq_len, head_dim)
        """
        seq_len = k.shape[2]
        
        # Write new K, V to the cache at the current position
        start_pos = self.current_len
        end_pos = start_pos + seq_len
        
        self.k_cache[layer_idx, :, :, start_pos:end_pos, :] = k
        self.v_cache[layer_idx, :, :, start_pos:end_pos, :] = v
        
        # Return view of valid cached data
        return (
            self.k_cache[layer_idx, :, :, :end_pos, :],
            self.v_cache[layer_idx, :, :, :end_pos, :],
        )
    
    def advance(self, seq_len: int = 1):
        """Advance the position counter after processing tokens."""
        self.current_len += seq_len
    
    def reset(self):
        """Reset cache for new generation."""
        self.current_len = 0
        # No need to zero - we only read up to current_len


# ============================================================================
# Attention (with Grouped Query Attention)
# ============================================================================

class Qwen3Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA).
    32 Q heads, 8 KV heads -> 4:1 ratio
    """
    
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.hidden_size = config.hidden_size
        
        # Q, K, V projections (no bias in Qwen3)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # For Q and K normalization (Qwen3 uses this)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape: (batch, seq, num_heads * head_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply Q and K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV cache handling - use pre-allocated cache if available
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
        
        # Repeat KV heads to match Q heads (for GQA)
        k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Use scaled_dot_product_attention for better performance
        # This is much faster than manual attention computation
        # Note: When using attn_mask, is_causal must be False
        if attention_mask is not None:
            # Ensure mask dtype matches q dtype to avoid MPS errors
            if attention_mask.dtype != q.dtype:
                attention_mask = attention_mask.to(q.dtype)
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale
            )
        else:
            # No mask - use is_causal for efficiency during prefill
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=(seq_len > 1),
                scale=self.scale
            )
        
        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


# ============================================================================
# MLP with SwiGLU Activation
# ============================================================================

class Qwen3MLP(nn.Module):
    """MLP with SwiGLU activation: SiLU(gate) * up, then down projection."""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate(x)) * up(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================================
# Transformer Block (Decoder Layer)
# ============================================================================

class Qwen3DecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids, attention_mask, kv_cache
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================================
# Full Qwen3 Model
# ============================================================================

class Qwen3Model(nn.Module):
    """Qwen3 transformer model (without LM head)."""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Create causal mask - use same dtype as hidden_states to avoid MPS dtype mismatch
        if attention_mask is None and seq_len > 1:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device, dtype=hidden_states.dtype),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Forward through all layers (KVCache is shared, updated in-place)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_ids, attention_mask, kv_cache
            )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 with language modeling head."""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        # lm_head shares weights with embed_tokens (weight tying)
        # We don't create a separate lm_head - we'll use embed_tokens.weight
        self.lm_head = None  # Will be tied after weight loading
    
    def tie_weights(self):
        """Tie lm_head to embed_tokens (they share the same weight matrix)."""
        # Create lm_head that shares weight with embeddings
        self.lm_head = lambda x: F.linear(x, self.model.embed_tokens.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, position_ids, attention_mask, kv_cache
        )
        logits = self.lm_head(hidden_states)
        return logits


# ============================================================================
# Weight Loading
# ============================================================================

def download_model(model_name: str = "Qwen/Qwen3-4B") -> str:
    """Download model from HuggingFace Hub."""
    print(f"Downloading {model_name} from HuggingFace...")
    model_path = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )
    print(f"Model downloaded to: {model_path}")
    return model_path


def load_weights(model: Qwen3ForCausalLM, model_path: str, device: torch.device, dtype: torch.dtype):
    """Load weights from safetensors files into our model."""
    print("Loading weights...")
    
    model_dir = Path(model_path)
    safetensor_files = list(model_dir.glob("*.safetensors"))
    
    # Mapping from HF names to our names
    def map_key(hf_key: str) -> str:
        """Map HuggingFace weight names to our model's names."""
        # Direct mappings
        key = hf_key
        # The HF model uses "model." prefix which we also use
        return key
    
    # Load all safetensor files
    state_dict = {}
    for sf_file in safetensor_files:
        print(f"  Loading {sf_file.name}...")
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[map_key(key)] = f.get_tensor(key)
    
    # Load into model (lm_head.weight will be "missing" because we tie it to embeddings)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Filter out expected missing keys (lm_head is tied to embeddings)
    missing = [k for k in missing if k != 'lm_head.weight']
    
    if missing:
        print(f"  Warning: Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  Warning: Unexpected keys: {unexpected[:5]}...")
    
    # Move to device and dtype
    model.to(device=device, dtype=dtype)
    
    # Tie lm_head weights to embeddings (weight sharing)
    model.tie_weights()
    
    print("Weights loaded successfully!")
    
    return model


# ============================================================================
# Text Generation
# ============================================================================

@torch.inference_mode()
def generate_streaming(
    model: Qwen3ForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    device: torch.device = None,
):
    """
    Generate text autoregressively with pre-allocated KV caching.
    Yields tokens as they are generated for streaming output.
    """
    if device is None:
        device = next(model.parameters()).device
    
    dtype = next(model.parameters()).dtype
    config = model.config
    
    # Format as chat
    messages = [{ "role": "system", "content": "You are a helpful assistant developed by Mathematica." },{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    
    # Pre-allocate KV cache for the entire generation
    # This avoids O(n²) memory copies from torch.cat
    total_len = prompt_len + max_new_tokens
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
    kv_cache.advance(prompt_len)  # Mark prompt tokens as cached
    
    # Get next token from last position
    next_token_logits = logits[:, -1, :]
    
    current_pos = prompt_len
    
    for _ in range(max_new_tokens):
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
            break
        
        # Decode and stream the new token
        all_tokens = tokenizer.decode([token_id], skip_special_tokens=True)
        if all_tokens:
            print(all_tokens, end="", flush=True)
        
        # Decode step: only process the new token with KV cache
        position_ids = torch.tensor([[current_pos]], device=device)
        logits = model(next_token, position_ids=position_ids, kv_cache=kv_cache)
        kv_cache.advance(1)  # Advance cache position by 1 token
        next_token_logits = logits[:, -1, :]
        current_pos += 1
    
    print()  # Final newline


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Qwen3-4B Pure PyTorch Implementation")
    print("=" * 60)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"\nUsing device: {device}")
    
    # Dtype selection:
    # - bfloat16: Native dtype, ~8GB memory (recommended for Apple Silicon M1 Pro+)
    # - float16: ~8GB memory, slightly faster on some hardware
    # - float32: ~16GB memory (won't fit on most Macs)
    # Note: True INT8/INT4 quantization requires additional libraries not yet MPS-compatible
    dtype = torch.bfloat16
    print(f"Using dtype: {dtype} (~8GB memory)")
    
    # Download model
    model_path = download_model("Qwen/Qwen3-4B")
    
    # Load config
    print("\nLoading configuration...")
    config = Qwen3Config.from_pretrained(model_path)
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads} Q, {config.num_key_value_heads} KV")
    print(f"  Vocab size: {config.vocab_size}")
    
    # Create model
    print("\nCreating model...")
    model = Qwen3ForCausalLM(config)
    
    # Load weights
    model = load_weights(model, model_path, device, dtype)
    model.eval()
    
    # Try torch.compile for potential speedup
    # Note: On MPS (Apple Silicon), support is still maturing
    # The first inference will be slow due to compilation
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True  # Fallback to eager on unsupported ops
        
        print("\nCompiling model with torch.compile (this may take 30-60s on first run)...")
        # Use reduce-overhead mode which works better on MPS
        # fullgraph=False allows partial compilation if some ops aren't supported
        model = torch.compile(
            model,
            mode="reduce-overhead",  # Optimizes for inference latency
            fullgraph=False,         # Allow fallback for unsupported ops
        )
        print("Model compiled successfully!")
    except Exception as e:
        print(f"torch.compile not available or failed: {e}")
        print("Continuing with eager mode (still fast!)")
    
    # Load tokenizer (we still use HF for tokenizer - it's complex)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Interactive chat
    print("\n" + "=" * 60)
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
            # Streaming generation - prints tokens as they're generated
            generate_streaming(model, tokenizer, user_input, device=device)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
