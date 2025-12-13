"""
Qwen3-4B Chat TUI
==================
A beautiful terminal user interface for chatting with Qwen3-4B.
Built with Textual.
"""

import sys
import os
import psutil
from datetime import datetime
from typing import Optional

# Must import torch and load model BEFORE textual to avoid fork issues
print("=" * 60)
print("🚀 Qwen3-4B Chat TUI")
print("=" * 60)
print()

import torch
from qwen3_pytorch import (
    Qwen3Config,
    Qwen3ForCausalLM,
    download_model,
    load_weights,
)
from transformers import AutoTokenizer
import torch.nn.functional as F

# ============================================================================
# Load Model First (before any TUI/threading)
# ============================================================================

MODEL_INFO = {
    "name": "Qwen3-4B",
    "params": "4B",
    "dtype": "bfloat16",
    "layers": 36,
    "context": "32K",
}

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process()
    mem_gb = process.memory_info().rss / (1024 ** 3)
    return f"{mem_gb:.1f}GB"

def get_gpu_memory():
    """Get GPU/MPS memory usage if available."""
    try:
        if torch.backends.mps.is_available():
            # MPS doesn't have direct memory query, estimate from process
            return get_memory_usage()
        elif torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            return f"{allocated:.1f}GB"
    except:
        pass
    return "N/A"

def get_device_info(device):
    """Get device info string."""
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    elif device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU"

def load_model():
    """Load the model synchronously before starting TUI."""
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


# Load model before importing textual
MODEL, TOKENIZER, DEVICE = load_model()
DEVICE_INFO = get_device_info(DEVICE)

# Now import textual (after model is loaded)
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.widgets import (
    Static,
    Input,
    Rule,
    Markdown,
)
from textual.worker import Worker, WorkerState
from rich.spinner import Spinner
import re


# ============================================================================
# Token Generation
# ============================================================================

TOKEN_COUNT = {"generated": 0}

def generate_tokens(prompt: str, max_new_tokens: int = 2048):
    """Generator that yields tokens one by one for streaming."""
    model = MODEL
    tokenizer = TOKENIZER
    device = DEVICE
    
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    
    TOKEN_COUNT["generated"] = 0
    
    with torch.inference_mode():
        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        
        # Prefill: process the entire prompt
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        logits, kv_caches = model(input_ids, position_ids=position_ids)
        
        # Get next token from last position
        next_token_logits = logits[:, -1, :]
        
        current_pos = input_ids.shape[1]
        
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
            
            TOKEN_COUNT["generated"] += 1
            
            # Decode and yield this token
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            if token_text:
                yield token_text
            
            # Decode step: only process the new token with KV cache
            position_ids = torch.tensor([[current_pos]], device=device)
            logits, kv_caches = model(next_token, position_ids=position_ids, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]
            current_pos += 1


# ============================================================================
# Custom Widgets
# ============================================================================

class InfoBar(Static):
    """Top info bar with model, memory, and GPU info."""
    
    DEFAULT_CSS = """
    InfoBar {
        width: 100%;
        height: 3;
        background: #0d1a14;
        border-bottom: solid #0d2a1d;
        layout: horizontal;
        align: center middle;
    }
    
    InfoBar .info-panel {
        width: 1fr;
        height: 100%;
        align: center middle;
    }
    
    InfoBar .info-label {
        color: #4a8c6a;
        height: auto;
        layout: horizontal;
        align: center middle;

    }
    
    InfoBar .info-value {
        color: #00d492;
        text-style: bold;
        height: auto;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="info-panel"):
            yield Static("MODEL: ", classes="info-label")
            yield Static("Qwen3-4B", id="info-model", classes="info-value")
        with Horizontal(classes="info-panel"):
            yield Static("MEMORY: ", classes="info-label")
            yield Static("--", id="info-memory", classes="info-value")
        with Horizontal(classes="info-panel"):
            yield Static("GPU: ", classes="info-label")
            yield Static("--", id="info-gpu", classes="info-value")
    
    def on_mount(self) -> None:
        self._update_info()
        self.set_interval(2.0, self._update_info)
    
    def _update_info(self) -> None:
        try:
            self.query_one("#info-memory", Static).update(get_memory_usage())
            self.query_one("#info-gpu", Static).update(get_gpu_memory())
        except NoMatches:
            pass
    
    def refresh_info(self) -> None:
        self._update_info()


class ThinkingSpinner(Static):
    """A spinner widget that animates while the model is thinking."""
    
    DEFAULT_CSS = """
    ThinkingSpinner {
        color: #4a8c6a;
        padding: 0;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._spinner = Spinner("dots", style="#4a8c6a")
    
    def on_mount(self) -> None:
        self.auto_refresh = 1 / 12  # 12 fps for smooth animation
    
    def render(self):
        return self._spinner


def is_thinking(content: str) -> bool:
    """Check if the model is currently in thinking mode."""
    # In thinking mode if we have <think> but no closing </think>
    return '<think>' in content and '</think>' not in content


def get_thinking_text(content: str) -> str:
    """Extract the thinking text from content."""
    match = re.search(r'<think>(.*?)(?:</think>|$)', content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def format_content(content: str) -> str:
    """Format content, removing think tags and converting LaTeX."""
    # Remove <think>...</think> blocks entirely from display
    formatted = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    # Remove incomplete <think> tags (still thinking)
    formatted = re.sub(r'<think>.*$', '', formatted, flags=re.DOTALL)
    # Escape LaTeX for display (Textual Markdown doesn't support LaTeX)
    formatted = re.sub(r'\$\$(.+?)\$\$', r'`\1`', formatted, flags=re.DOTALL)
    formatted = re.sub(r'\$(.+?)\$', r'`\1`', formatted)
    return formatted.strip()


class ChatMessage(Static):
    """A clean output display - just the model response with Markdown support."""
    
    DEFAULT_CSS = """
    ChatMessage {
        layout: vertical;
    }
    
    ChatMessage .thinking-container {
        layout: vertical;
        height: auto;
        padding: 0 0 1 0;
    }
    
    ChatMessage .thinking-header {
        layout: horizontal;
        height: auto;
    }
    
    ChatMessage .thinking-text {
        color: #666666;
        padding: 0 0 0 2;
        height: auto;
    }
    """
    
    def __init__(
        self,
        content: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.msg_content = content
        self._is_thinking = False
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="thinking-container", id="thinking-container"):
            with Horizontal(classes="thinking-header"):
                yield ThinkingSpinner(id="spinner")
                yield Static(" Thinking...", classes="thinking-label")
            yield Static("", id="thinking-text", classes="thinking-text")
        yield Markdown("", classes="msg-text")
    
    def on_mount(self) -> None:
        # Initially hide thinking container
        self.query_one("#thinking-container").display = False
    
    def update_content(self, new_content: str):
        """Update the message content."""
        self.msg_content = new_content
        thinking = is_thinking(new_content)
        
        try:
            thinking_container = self.query_one("#thinking-container")
            thinking_container.display = thinking
            
            if thinking:
                # Update thinking text
                think_text = get_thinking_text(new_content)
                self.query_one("#thinking-text", Static).update(think_text)
            
            # Format and display the non-thinking content
            display_content = format_content(new_content)
            self.query_one(".msg-text", Markdown).update(display_content)
        except NoMatches:
            pass


class ChatContainer(ScrollableContainer):
    """Scrollable container for model output."""
    
    def add_message(self, content: str) -> ChatMessage:
        """Add a new message to the output."""
        message = ChatMessage(content)
        self.mount(message)
        self.scroll_end(animate=False)
        return message


# ============================================================================
# Main TUI App
# ============================================================================

class Qwen3ChatApp(App):
    """A clean TUI for chatting with Qwen3-4B."""
    
    CSS = """
    /* Color scheme - Neon Green */
    $bg: #0a0a0f;
    $bg-light: #0d1a14;
    $accent: #00d492;
    $accent-light: #00d492;
    $text: #e0fff0;
    $text-dim: #4a8c6a;
    $user-color: #00d492;
    $border: #0d2a1d;
    
    Screen {
        background: $bg;
        layout: vertical;
    }
    
    /* Top Info Bar */
    #info-bar {
        width: 100%;
        height: 3;
        background: #0d1a14;
        border-bottom: solid $border;
        layout: horizontal;
        align: center middle;
    }
    
    #info-bar .info-panel {
        width: 1fr;
        height: 100%;
        content-align: center middle;
        align: center middle;
    }
    
    #info-bar .info-label {
        color: $text-dim;
    }
    
    #info-bar .info-value {
        color: $accent;
        text-style: bold;
    }
    
    /* Chat Area */
    #chat-area {
        width: 100%;
        height: 1fr;
        padding: 1 2;
        background: $bg;
    }
    
    ChatMessage {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0;
        background: transparent;
    }
    
    .msg-text {
        color: $text;
        padding: 0 0 1 0;
    }
    
    /* Welcome Message */
    .welcome {
        color: $text-dim;
        text-align: center;
        padding: 4 0;
    }
    
    /* Input Area */
    #input-area {
        width: 100%;
        height: auto;
        background: $bg;
        border-top: solid $border;
        padding: 1 2;
    }
    
    #prompt-input {
        width: 100%;
        background: transparent;
        border: round $border;
        padding: 0 1;
    }
    
    #prompt-input:focus {
        border: round $accent;
    }
    
    #prompt-input.-generating {
        color: $text-dim;
    }
    
    /* Help Text */
    #help-text {
        width: 100%;
        height: 1;
        color: $text-dim;
        text-align: center;
        padding: 0;
        margin-top: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_chat", "Clear", show=False),
        Binding("escape", "cancel_generation", "Cancel", show=False),
    ]
    
    def __init__(self):
        super().__init__()
        self.is_generating = False
        self.cancel_generation = False
        self.current_response_widget = None
        self.full_response = ""
        self.generation_start_time = None
    
    def compose(self) -> ComposeResult:
        # Top info bar
        yield InfoBar(id="info-bar")
        
        # Main chat area
        yield ChatContainer(id="chat-area")
        
        # Bottom input area
        with Vertical(id="input-area"):
            yield Input(
                placeholder="Message Qwen3... (Enter to send, Esc to cancel)",
                id="prompt-input"
            )
            yield Static(
                "[dim]Ctrl+L[/] clear  •  [dim]Esc[/] cancel  •  [dim]Ctrl+C[/] quit",
                id="help-text"
            )
    
    def on_mount(self) -> None:
        """Initialize when app mounts."""
        # Add welcome message
        chat = self.query_one("#chat-area", ChatContainer)
        welcome = Static(
            "[bold #00d492]Welcome to Qwen3-4B Chat[/]\n\n"
            "[#4a8c6a]A pure PyTorch implementation running locally.\n"
            "Type a message below to start chatting.[/]",
            classes="welcome"
        )
        chat.mount(welcome)
        
        # Focus input
        self.query_one("#prompt-input").focus()
        
        # Refresh info bar periodically
        self.set_interval(2.0, self._refresh_info)
    
    def _refresh_info(self) -> None:
        """Refresh the info bar."""
        try:
            self.query_one("#info-bar", InfoBar).refresh_info()
        except NoMatches:
            pass
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        if event.input.id == "prompt-input":
            await self._send_message()
    
    async def _send_message(self) -> None:
        """Send the current message."""
        if self.is_generating:
            return
        
        input_widget = self.query_one("#prompt-input", Input)
        message = input_widget.value.strip()
        
        if not message:
            return
        
        # Clear input
        input_widget.value = ""
        input_widget.add_class("-generating")
        
        # Remove welcome message if present
        try:
            welcome = self.query_one(".welcome")
            welcome.remove()
        except NoMatches:
            pass
        
        # Add response message with cursor (model output only, no user message shown)
        chat = self.query_one("#chat-area", ChatContainer)
        self.current_response_widget = chat.add_message("▌")
        self.full_response = ""
        self.generation_start_time = datetime.now()
        
        # Disable input during generation
        self.is_generating = True
        self.cancel_generation = False
        
        # Start background generation
        self.run_generation(message)
    
    @work(thread=True)
    def run_generation(self, prompt: str) -> None:
        """Run token generation in a background thread."""
        try:
            for token in generate_tokens(prompt):
                if self.cancel_generation:
                    break
                self.full_response += token
                # Update UI from thread
                self.call_from_thread(self._update_streaming_response)
        except Exception as e:
            self.call_from_thread(self._show_error, str(e))
        finally:
            self.call_from_thread(self._finish_generation)
    
    def _update_streaming_response(self) -> None:
        """Update the response widget with current text."""
        if self.current_response_widget:
            self.current_response_widget.update_content(self.full_response + "▌")
            # Scroll to bottom
            try:
                chat = self.query_one("#chat-area", ChatContainer)
                chat.scroll_end(animate=False)
            except NoMatches:
                pass
    
    def _show_error(self, error: str) -> None:
        """Show an error message."""
        if self.current_response_widget:
            self.current_response_widget.update_content(f"[red]Error: {error}[/red]")
    
    def _finish_generation(self) -> None:
        """Clean up after generation completes."""
        # Calculate stats
        elapsed = (datetime.now() - self.generation_start_time).total_seconds()
        tokens = TOKEN_COUNT["generated"]
        tps = tokens / elapsed if elapsed > 0 else 0
        
        # Update final response (remove cursor)
        if self.current_response_widget:
            final_text = self.full_response if self.full_response else "[dim]No response generated[/dim]"
            self.current_response_widget.update_content(final_text)
        
        # Re-enable input
        self.is_generating = False
        self.current_response_widget = None
        
        try:
            input_widget = self.query_one("#prompt-input", Input)
            input_widget.remove_class("-generating")
            input_widget.focus()
            
            # Scroll to bottom
            chat = self.query_one("#chat-area", ChatContainer)
            chat.scroll_end(animate=False)
        except NoMatches:
            pass
    
    def action_clear_chat(self) -> None:
        """Clear all chat messages."""
        if self.is_generating:
            return
        
        try:
            chat = self.query_one("#chat-area", ChatContainer)
            chat.remove_children()
            
            # Add welcome message back
            welcome = Static(
                "[bold #00d492]Chat Cleared[/]\n\n"
                "[#4a8c6a]Type a message below to start a new conversation.[/]",
                classes="welcome"
            )
            chat.mount(welcome)
        except NoMatches:
            pass
    
    def action_cancel_generation(self) -> None:
        """Cancel ongoing generation."""
        if self.is_generating:
            self.cancel_generation = True


# ============================================================================
# Entry Point
# ============================================================================

def main():
    print("🎨 Starting TUI...")
    print()
    app = Qwen3ChatApp()
    app.run()


if __name__ == "__main__":
    main()
