"""
Qwen3-4B MLX Backend (Mac Only)
================================
Provides the MLX backend for Apple Silicon Macs.
Uses INT4 quantization for efficient inference (~2GB memory).
"""

import platform
import subprocess
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Generator

# Add inference folder to path (sibling directory)
_server_dir = Path(__file__).parent
_project_root = _server_dir.parent
_inference_dir = _project_root / "inference"
if str(_inference_dir) not in sys.path:
    sys.path.insert(0, str(_inference_dir))


# ============================================================================
# System Requirements Check
# ============================================================================

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return (
        platform.system() == "Darwin" and 
        platform.machine() == "arm64"
    )


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback for macOS without psutil
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
        except Exception:
            pass
    return 0.0


def get_gpu_memory_usage_gb() -> Optional[float]:
    """
    Get current GPU/unified memory usage on Apple Silicon via ioreg.
    Returns the memory in GB, or None if unavailable.
    """
    if not is_macos():
        return None
    
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            match = re.search(r'"In use system memory"=(\d+)', result.stdout)
            if match:
                mem_bytes = int(match.group(1))
                return mem_bytes / (1024 ** 3)
    except Exception:
        pass
    return None


def check_system_requirements() -> tuple[bool, str]:
    """
    Check if the system meets requirements for running the model.
    
    Returns:
        Tuple of (success, message)
    """
    # Check macOS
    if not is_macos():
        return False, (
            "❌ This application requires macOS.\n"
            "   OpenLLM with MLX backend only runs on Apple Silicon Macs."
        )
    
    # Check Apple Silicon
    if not is_apple_silicon():
        return False, (
            "❌ This application requires Apple Silicon (M1/M2/M3/M4/M5).\n"
            "   Intel Macs are not supported. MLX requires Apple Silicon."
        )
    
    # Check memory (need at least 8GB, recommend 16GB)
    total_memory = get_system_memory_gb()
    if total_memory < 8:
        return False, (
            f"❌ Insufficient memory: {total_memory:.1f}GB available.\n"
            "   At least 8GB of RAM is required to run Qwen3-4B.\n"
            "   The model uses ~2GB + overhead for inference."
        )
    
    # Check MLX installation
    try:
        import mlx.core
        import mlx_lm
    except ImportError:
        return False, (
            "❌ MLX is not installed.\n"
            "   Please install MLX: pip install mlx mlx-lm"
        )
    
    # All checks passed
    memory_note = ""
    if total_memory < 16:
        memory_note = f"\n⚠️  Note: {total_memory:.0f}GB RAM detected. 16GB+ recommended for best performance."
    
    return True, f"✅ System requirements met.{memory_note}"


def print_system_info():
    """Print system information for debugging."""
    print("=" * 60)
    print("🍎 System Information")
    print("=" * 60)
    print(f"   Platform: {platform.system()} {platform.machine()}")
    print(f"   macOS Version: {platform.mac_ver()[0]}")
    print(f"   Python: {platform.python_version()}")
    
    total_mem = get_system_memory_gb()
    print(f"   Total Memory: {total_mem:.1f}GB")
    
    gpu_mem = get_gpu_memory_usage_gb()
    if gpu_mem is not None:
        print(f"   GPU Memory In Use: {gpu_mem:.2f}GB")
    
    print("=" * 60)


# ============================================================================
# MLX Backend
# ============================================================================

class MLXBackend:
    """MLX backend for Apple Silicon with INT4 quantization."""
    
    def __init__(self, use_thinking: bool = True):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = "mlx"
        self.backend_name = "mlx"
        self.use_thinking = use_thinking
    
    def load(self) -> None:
        """Load the MLX model."""
        from qwen3_mlx import Qwen3MLX
        
        print("🍎 Loading MLX backend (INT4 quantized)")
        
        qwen = Qwen3MLX.from_pretrained(use_thinking=self.use_thinking)
        self.model = qwen.model
        self.tokenizer = qwen.tokenizer
        self.config = qwen.config
        
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
# Factory Function
# ============================================================================

def create_backend(use_thinking: bool = True) -> MLXBackend:
    """
    Create the MLX backend after checking system requirements.
    
    Args:
        use_thinking: Whether to use the thinking variant
    
    Returns:
        MLXBackend instance
    
    Raises:
        SystemExit: If system requirements are not met
    """
    # Check requirements first
    success, message = check_system_requirements()
    if not success:
        print(message)
        sys.exit(1)
    
    print(message)
    print()
    
    return MLXBackend(use_thinking=use_thinking)


# ============================================================================
# Global Backend Instance
# ============================================================================

_backend: Optional[MLXBackend] = None


def get_backend() -> MLXBackend:
    """Get the global backend instance, loading if necessary."""
    global _backend
    if _backend is None:
        _backend = create_backend()
        _backend.load()
    return _backend


def load_backend(use_thinking: bool = True) -> MLXBackend:
    """Load and return the backend."""
    global _backend
    _backend = create_backend(use_thinking=use_thinking)
    _backend.load()
    return _backend
