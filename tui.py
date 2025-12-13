import psutil
import os
import json
import requests
import subprocess
import platform
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Container, Horizontal
from textual.widgets import Input, Static, Label, Markdown, LoadingIndicator
from textual.reactive import reactive

try:
    import torch
    HAS_TORCH = torch.backends.mps.is_available() or torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False


def get_memory_usage() -> str:
    """Get system memory usage percentage"""
    mem = psutil.virtual_memory()
    return f"{mem.percent:.0f}%"


def get_gpu_usage() -> str:
    """Get GPU/unified memory usage on Apple Silicon via ioreg"""
    try:
        if platform.system() == "Darwin":  # macOS with Apple Silicon
            # Use ioreg to get actual GPU memory from AGXAccelerator
            result = subprocess.run(
                ["ioreg", "-r", "-c", "AGXAccelerator"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                import re
                # Look for "In use system memory"=NNNNNN in the output
                match = re.search(r'"In use system memory"=(\d+)', result.stdout)
                if match:
                    mem_bytes = int(match.group(1))
                    mem_gb = mem_bytes / 1024 / 1024 / 1024
                    if mem_gb >= 1:
                        return f"{mem_gb:.1f}GB"
                    else:
                        return f"{mem_bytes / 1024 / 1024:.0f}MB"
            return "N/A"
        
        elif HAS_TORCH and torch.cuda.is_available():
            # NVIDIA GPU - use nvidia-smi for accurate GPU utilization
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                gpu_util = result.stdout.strip().split("\n")[0]
                return f"{gpu_util}%"
            except Exception:
                # Fallback to memory-based reporting
                total = torch.cuda.get_device_properties(0).total_memory
                used = torch.cuda.memory_allocated()
                percent = (used / total) * 100
                return f"{percent:.0f}%"
    except Exception:
        pass
    
    return "N/A"


class StatusIndicator(Static):
    """Status indicator on the left"""
    pass


class StatsDisplay(Static):
    """Stats display on the right"""
    pass


class TopBar(Horizontal):
    """Top menu/info bar with dynamic content"""
    
    memory = reactive("0%")
    gpu = reactive("N/A")
    server_ready = reactive(False)
    
    def compose(self):
        yield StatusIndicator("", id="status")
        yield StatsDisplay("", id="stats")
    
    def on_mount(self) -> None:
        self.update_stats()
        self.set_interval(1, self.update_stats)
        self.check_server()
        self.set_interval(60, self.check_server)
    
    def update_stats(self) -> None:
        self.memory = get_memory_usage()
        self.gpu = get_gpu_usage()
    
    def check_server(self) -> None:
        try:
            response = requests.get("http://localhost:5000/ping", timeout=2.0)
            self.server_ready = response.status_code == 200 and response.json().get("status") == "success"
        except Exception:
            self.server_ready = False
    
    def watch_memory(self, memory: str) -> None:
        self._refresh_display()
    
    def watch_gpu(self, gpu: str) -> None:
        self._refresh_display()
    
    def watch_server_ready(self, ready: bool) -> None:
        self._refresh_display()
    
    def _refresh_display(self) -> None:
        if self.server_ready:
            status = "[green]●[/green] Ready"
        else:
            status = "[red]●[/red] Offline"
        
        try:
            self.query_one("#status", StatusIndicator).update(status)
            self.query_one("#stats", StatsDisplay).update(f"RAM: {self.memory}  │  GPU Mem: {self.gpu}")
        except Exception:
            pass





class UserMessage(Static):
    """A user chat message with background"""
    def __init__(self, text: str):
        super().__init__(text, classes="message user")


class ThinkingMessage(Horizontal):
    """A thinking message with spinner and small text"""
    def __init__(self):
        super().__init__(classes="thinking-container")
        self._thinking_text = ""
        self._is_done = False
    
    def compose(self) -> ComposeResult:
        yield LoadingIndicator(id="thinking-spinner")
        yield Static("", id="thinking-text", classes="thinking-text")
    
    def update_thinking(self, text: str) -> None:
        """Update the thinking text"""
        self._thinking_text = text
        try:
            self.query_one("#thinking-text", Static).update(text)
        except Exception:
            pass
    
    def finish_thinking(self) -> None:
        """Stop the spinner and mark thinking as complete"""
        self._is_done = True
        try:
            spinner = self.query_one("#thinking-spinner", LoadingIndicator)
            spinner.remove()
        except Exception:
            pass


class BotMessage(Markdown):
    """A bot chat message with markdown support"""
    def __init__(self, text: str = ""):
        super().__init__(text, classes="message bot")
    
    def append_text(self, text: str) -> None:
        """Append text to the message (for streaming)"""
        current = self._markdown if hasattr(self, '_markdown') else ""
        self._markdown = current + text
        self.update(self._markdown)


class ChatApp(App):
    def __init__(self):
        super().__init__()
        self.conversation_history: list[dict] = []

    CSS = """
    Screen {
        background: #131313;
        color: #e0e0e0;
        layout: vertical;
        width: 100%;
        height: 100%;
        padding: 0;
        margin: 0;
    }

    /* ===== TOP BAR ===== */
    TopBar {
        height: 2;
        width: 100%;
        background: #131313;
        margin: 1 2;
    }

    #status {
        width: auto;
        height: auto;
    }

    #stats {
        width: 1fr;
        height: auto;
        text-align: right;
    }

    /* ===== MESSAGES AREA ===== */
    #messages {
        height: 1fr;
        width: 100%;
        padding: 1 2;
        background: #131313;
    }

    .message {
        margin-bottom: 1;
        padding: 0 1;
        width: 100%;
    }

    .user {
        color: #e0e0e0;
        background: #2a2a3a;
        padding: 1;
    }

    .bot {
        color: #e0e0e0;
        padding: 0;
    }

    .bot > * {
        margin: 0;
    }

    /* Markdown code blocks */
    Markdown {
        margin: 0;
        padding: 0;
    }

    MarkdownFence {
        background: #1e1e1e;
        color: #d4d4d4;
        margin: 1 0;
        padding: 1;
    }

    MarkdownH1, MarkdownH2, MarkdownH3 {
        color: #7aa2f7;
        margin: 1 0;
    }

    MarkdownBlockQuote {
        border-left: thick #7aa2f7;
        padding-left: 1;
        color: #888;
    }

    MarkdownBulletList, MarkdownOrderedList {
        margin: 0 0 0 2;
    }

    /* ===== THINKING CONTAINER ===== */
    .thinking-container {
        height: auto;
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
    }

    #thinking-spinner {
        width: 4;
        height: 1;
        padding: 0;
        margin: 0;
        color: #7aa2f7;
    }

    .thinking-text {
        width: 1fr;
        height: auto;
        color: #888;
        text-style: italic;
        padding-left: 1;
    }

    /* ===== INPUT CONTAINER ===== */
    #input-container {
        height: auto;
        width: 100%;
        padding: 1 2;
        background: #131313;
    }

    Input {
        border: round #7aa2a7;
        background: transparent;
        color: #e0e0e0;
        padding: 0 1;
    }

    Input:focus {
        border: round #7aa2f7;
    }

    Input > .input--placeholder {
        color: #666;
    }
    """

    def compose(self) -> ComposeResult:
        yield TopBar()
        yield VerticalScroll(id="messages")

        with Container(id="input-container"):
            yield Input(placeholder="Type a message...")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""

        messages_container = self.query_one("#messages")
        await messages_container.mount(UserMessage(f"You: {text}"))
        messages_container.scroll_end(animate=False)

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": text})

        # Create thinking message first (will be replaced when done thinking)
        thinking_message = ThinkingMessage()
        await messages_container.mount(thinking_message)
        messages_container.scroll_end(animate=False)

        # Create bot message placeholder (hidden until thinking is done)
        bot_message = BotMessage()

        # Make request to server in background worker
        self.run_worker(
            lambda: self._do_llm_request(text, thinking_message, bot_message, messages_container),
            thread=True,
        )

    def _do_llm_request(self, text: str, thinking_message: ThinkingMessage, bot_message: BotMessage, messages_container: VerticalScroll) -> None:
        """Send message to LLM server and stream response (runs in thread)"""
        payload = {
            "messages": list(self.conversation_history),
            "stream": True,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        
        try:
            response = requests.post(
                "http://localhost:5000/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=120,
            )
            
            if response.status_code != 200:
                self.call_from_thread(thinking_message.remove)
                self.call_from_thread(self._mount_widget, bot_message, messages_container)
                self.call_from_thread(bot_message.update, f"*Error: Server returned {response.status_code}*")
                return
            
            full_response = ""
            thinking_text = ""
            in_thinking = False
            thinking_done = False
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_response += content
                                
                                # Check for thinking tags
                                if "<think>" in full_response and not in_thinking:
                                    in_thinking = True
                                
                                if "</think>" in full_response and in_thinking:
                                    in_thinking = False
                                    thinking_done = True
                                    # Stop spinner but keep thinking text visible
                                    # Extract final thinking text (between <think> and </think>)
                                    think_content = full_response.split("<think>", 1)[-1].split("</think>", 1)[0].strip()
                                    self.call_from_thread(thinking_message.update_thinking, think_content)
                                    self.call_from_thread(thinking_message.finish_thinking)
                                    # Mount bot message after thinking
                                    self.call_from_thread(self._mount_widget, bot_message, messages_container)
                                    # Extract response after </think>
                                    response_text = full_response.split("</think>", 1)[-1].strip()
                                    if response_text:
                                        self.call_from_thread(bot_message.update, response_text)
                                    self.call_from_thread(messages_container.scroll_end)
                                elif in_thinking:
                                    # Update thinking text (strip <think> tag)
                                    thinking_text = full_response.replace("<think>", "").strip()
                                    self.call_from_thread(thinking_message.update_thinking, thinking_text)
                                    self.call_from_thread(messages_container.scroll_end)
                                elif thinking_done:
                                    # After thinking, update bot message
                                    response_text = full_response.split("</think>", 1)[-1].strip()
                                    self.call_from_thread(bot_message.update, response_text)
                                    self.call_from_thread(messages_container.scroll_end)
                                else:
                                    # No thinking tags - direct response
                                    if not thinking_done:
                                        self.call_from_thread(thinking_message.remove)
                                        self.call_from_thread(self._mount_widget, bot_message, messages_container)
                                        thinking_done = True
                                    self.call_from_thread(bot_message.update, full_response)
                                    self.call_from_thread(messages_container.scroll_end)
                        except json.JSONDecodeError:
                            pass
            
            # If we never got out of thinking mode, clean up
            if not thinking_done:
                self.call_from_thread(thinking_message.remove)
                self.call_from_thread(self._mount_widget, bot_message, messages_container)
                if full_response:
                    # Remove thinking tags from response
                    clean_response = full_response.replace("<think>", "").replace("</think>", "").strip()
                    self.call_from_thread(bot_message.update, clean_response if clean_response else "*No response from server*")
                    # Add assistant response to history (clean version without thinking tags)
                    if clean_response:
                        self.conversation_history.append({"role": "assistant", "content": clean_response})
                else:
                    self.call_from_thread(bot_message.update, "*No response from server*")
            else:
                # Add the final response to history (content after </think>)
                response_text = full_response.split("</think>", 1)[-1].strip() if "</think>" in full_response else full_response.replace("<think>", "").strip()
                if response_text:
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                
        except requests.exceptions.ConnectionError:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._mount_widget, bot_message, messages_container)
            self.call_from_thread(bot_message.update, "*Error: Could not connect to server. Is it running?*")
        except requests.exceptions.Timeout:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._mount_widget, bot_message, messages_container)
            self.call_from_thread(bot_message.update, "*Error: Request timed out*")
        except Exception as e:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._mount_widget, bot_message, messages_container)
            self.call_from_thread(bot_message.update, f"*Error: {str(e)}*")

    def _mount_widget(self, widget, container) -> None:
        """Helper to mount a widget from thread"""
        container.mount(widget)

if __name__ == "__main__":
    ChatApp().run()
