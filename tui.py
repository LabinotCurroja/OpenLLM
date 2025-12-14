import psutil
import os
import json
import requests
import subprocess
import platform
import time
import asyncio
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


class TokenStats(Static):
    """Token generation stats display below input"""
    pass


class TopBar(Horizontal):
    """Top menu/info bar with dynamic content"""
    
    memory = reactive("0%")
    gpu = reactive("N/A")
    server_ready = reactive(False)
    tools_ready = reactive(False)
    
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
            response = requests.get("http://localhost:8000/ping", timeout=2.0)
            self.server_ready = response.status_code == 200 and response.json().get("status") == "success"
            
            # Check tools status
            if self.server_ready:
                tools_response = requests.get("http://localhost:8000/v1/tools", timeout=2.0)
                if tools_response.status_code == 200:
                    tools_data = tools_response.json()
                    self.tools_ready = tools_data.get("search_configured", False)
                else:
                    self.tools_ready = False
            else:
                self.tools_ready = False
        except Exception:
            self.server_ready = False
            self.tools_ready = False
    
    def watch_memory(self, memory: str) -> None:
        self._refresh_display()
    
    def watch_gpu(self, gpu: str) -> None:
        self._refresh_display()
    
    def watch_server_ready(self, ready: bool) -> None:
        self._refresh_display()
    
    def watch_tools_ready(self, ready: bool) -> None:
        self._refresh_display()
    
    def _refresh_display(self) -> None:
        if self.server_ready:
            status = "[green]●[/green] Ready"
            if self.tools_ready:
                status += " [dim]│[/dim] [blue]🔍[/blue]"
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


class ToolMessage(Static):
    """A message showing tool usage"""
    def __init__(self, tool_name: str, query: str = "", status: str = "calling"):
        if tool_name == "web_search":
            if status == "calling":
                text = f"[blue]🔍 Searching:[/blue] {query}" if query else "[blue]🔍 Searching...[/blue]"
            elif status == "done":
                text = f"[dim]🔍 Searched: {query}[/dim]"
            else:
                text = f"[blue]🔍[/blue] {status}"
        elif tool_name == "get_current_time":
            text = "[blue]🕐 Getting current time...[/blue]"
        else:
            text = f"[blue]🔧 {tool_name}[/blue]: {status}"
        super().__init__(text, classes="tool-message")
    
    def mark_done(self, query: str = "") -> None:
        """Mark the tool call as complete"""
        self.update(f"[dim]🔍 Searched: {query}[/dim]")


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
    SYSTEM_PROMPT = """You are a helpful assistant called Eve. You never use emojis, and always respond in a concise and clear manner. When you <think>, its important that you dont spend too much time on simple questions. If a user says hi, you just need to greet them back. """

    def __init__(self):
        super().__init__()
        self.conversation_history: list[dict] = []
        self.tools_enabled: bool = True  # Enable tool calling by default

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

    /* ===== TOKEN STATS ===== */
    #token-stats {
        height: 1;
        width: 100%;
        text-align: left;
        color: #666;
        padding: 0 2;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield TopBar()
        yield VerticalScroll(id="messages")

        with Container(id="input-container"):
            yield Input(placeholder="Type a message...")
        yield TokenStats("", id="token-stats")

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

        # Create and pre-mount bot message (hidden until thinking is done)
        bot_message = BotMessage()
        bot_message.display = False  # Hide initially
        await messages_container.mount(bot_message)

        # Make request to server in background worker
        self.run_worker(
            lambda: self._do_llm_request_sync(text, thinking_message, bot_message, messages_container),
            thread=True,
        )

    def _do_llm_request_sync(self, text: str, thinking_message: ThinkingMessage, bot_message: BotMessage, messages_container: VerticalScroll) -> None:
        """Send message to LLM server and stream response (runs in thread)"""
        import re
        
        # Limit context to last 12 messages to prevent unbounded growth
        recent_messages = self.conversation_history[-12:] if len(self.conversation_history) > 12 else self.conversation_history
        # Always prepend system prompt to ensure it's never truncated
        messages_with_system = [{"role": "system", "content": self.SYSTEM_PROMPT}] + list(recent_messages)
        payload = {
            "messages": messages_with_system,
            "stream": True,
            "max_tokens": 8192,
            "temperature": 0.6,
            "use_tools": self.tools_enabled,  # Enable tool calling
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=120,
            )
            
            if response.status_code != 200:
                self.call_from_thread(thinking_message.remove)
                self.call_from_thread(self._show_bot_message, bot_message)
                self.call_from_thread(bot_message.update, f"*Error: Server returned {response.status_code}*")
                return
            
            full_response = ""
            thinking_text = ""
            # New thinking model has implicit <think> in chat template,
            # so we start in thinking mode and look only for </think>
            in_thinking = True
            thinking_done = False
            
            # Tool call tracking
            in_tool_call = False
            tool_round = 0  # Track which tool round we're in
            last_response_text = ""  # Track the actual response text (after all thinking/tools)
            round_response = ""  # Track response for current round only
            
            # Token counting for tokens/sec
            token_count = 0
            start_time = time.time()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            
                            # Check for tool_calls in delta (server sends these for tool events)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "tool_calls" in delta:
                                # Tool is being called - update thinking message
                                tool_info = delta["tool_calls"][0].get("function", {})
                                tool_name = tool_info.get("name", "tool")
                                self.call_from_thread(thinking_message.update_thinking, f"🔍 Calling {tool_name}...")
                                continue
                            
                            content = delta.get("content", "")
                            if content:
                                # Check if this looks like a tool result marker (new round starting)
                                if "[Searching:" in content:
                                    # Server is indicating tool execution, skip this marker
                                    tool_round += 1
                                    # Reset for new round - model will generate new thinking
                                    in_thinking = True
                                    thinking_done = False
                                    round_response = ""  # Reset round response
                                    # Re-show the thinking spinner
                                    self.call_from_thread(thinking_message.update_thinking, "Processing tool results...")
                                    # Hide bot message again while thinking
                                    self.call_from_thread(self._hide_bot_message, bot_message)
                                    continue
                                
                                full_response += content
                                round_response += content
                                token_count += 1
                                
                                # Update tokens/sec every few tokens
                                if token_count % 5 == 0:
                                    elapsed = time.time() - start_time
                                    if elapsed > 0:
                                        tps = token_count / elapsed
                                        self.call_from_thread(self._update_tps, f"{tps:.1f} tok/s")
                                
                                # Check for tool call (just track it, don't try to mount UI)
                                if "<tool_call>" in round_response and not in_tool_call:
                                    in_tool_call = True
                                
                                if in_tool_call and "</tool_call>" in round_response:
                                    in_tool_call = False
                                
                                # Check for thinking end tag
                                if "</think>" in round_response and in_thinking:
                                    in_thinking = False
                                    thinking_done = True
                                    # Stop spinner but keep thinking text visible
                                    think_content = round_response.split("</think>", 1)[0].strip()
                                    self.call_from_thread(thinking_message.update_thinking, think_content)
                                    self.call_from_thread(thinking_message.finish_thinking)
                                    # Show bot message after thinking
                                    self.call_from_thread(self._show_bot_message, bot_message)
                                    # Extract response after </think>, clean tool calls
                                    response_text = round_response.split("</think>", 1)[-1].strip()
                                    response_text = self._clean_tool_calls(response_text)
                                    last_response_text = response_text
                                    if response_text:
                                        self.call_from_thread(bot_message.update, response_text)
                                    self.call_from_thread(messages_container.scroll_end)
                                elif in_thinking:
                                    # Update thinking text (show current round's thinking)
                                    thinking_text = round_response.strip()
                                    # Clean any tool call artifacts from thinking display
                                    display_thinking = self._clean_tool_calls(thinking_text)
                                    display_thinking = display_thinking.replace("</think>", "").strip()
                                    self.call_from_thread(thinking_message.update_thinking, display_thinking)
                                    self.call_from_thread(messages_container.scroll_end)
                                elif thinking_done:
                                    # After thinking, update bot message (clean tool calls)
                                    response_text = round_response.split("</think>", 1)[-1].strip()
                                    response_text = self._clean_tool_calls(response_text)
                                    last_response_text = response_text
                                    self.call_from_thread(bot_message.update, response_text)
                                    self.call_from_thread(messages_container.scroll_end)
                                else:
                                    # No thinking tags - direct response (shouldn't happen with thinking model)
                                    if not thinking_done:
                                        self.call_from_thread(thinking_message.remove)
                                        self.call_from_thread(self._show_bot_message, bot_message)
                                        thinking_done = True
                                    clean_response = self._clean_tool_calls(round_response)
                                    last_response_text = clean_response
                                    self.call_from_thread(bot_message.update, clean_response)
                                    self.call_from_thread(messages_container.scroll_end)
                        except json.JSONDecodeError:
                            pass
            
            # Show final tokens/sec
            elapsed = time.time() - start_time
            if elapsed > 0 and token_count > 0:
                final_tps = token_count / elapsed
                self.call_from_thread(self._update_tps, f"{final_tps:.1f} tok/s ({token_count} tokens)")
            
            # If we never got out of thinking mode, clean up
            if not thinking_done:
                self.call_from_thread(thinking_message.remove)
                self.call_from_thread(self._show_bot_message, bot_message)
                if full_response:
                    # Remove thinking tags and tool calls from response
                    clean_response = full_response.replace("<think>", "").replace("</think>", "").strip()
                    clean_response = self._clean_tool_calls(clean_response)
                    last_response_text = clean_response
                    self.call_from_thread(bot_message.update, clean_response if clean_response else "*No response from server*")
                    # Add assistant response to history (clean version)
                    if clean_response:
                        self.conversation_history.append({"role": "assistant", "content": clean_response})
                else:
                    self.call_from_thread(bot_message.update, "*No response from server*")
            else:
                # Add the final response to history (use last_response_text which is cleaned)
                if last_response_text:
                    self.conversation_history.append({"role": "assistant", "content": last_response_text})
                
        except requests.exceptions.ConnectionError:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._show_bot_message, bot_message)
            self.call_from_thread(bot_message.update, "*Error: Could not connect to server. Is it running?*")
        except requests.exceptions.Timeout:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._show_bot_message, bot_message)
            self.call_from_thread(bot_message.update, "*Error: Request timed out*")
        except Exception as e:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._show_bot_message, bot_message)
            self.call_from_thread(bot_message.update, f"*Error: {str(e)}*")

    def _show_bot_message(self, bot_message: BotMessage) -> None:
        """Show the pre-mounted bot message"""
        bot_message.display = True
    
    def _hide_bot_message(self, bot_message: BotMessage) -> None:
        """Hide the bot message (for when we go back to thinking after tool call)"""
        bot_message.display = False
    
    def _clean_tool_calls(self, text: str) -> str:
        """Remove tool call tags and artifacts from text"""
        import re
        # Remove <tool_call>...</tool_call> blocks
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        # Remove any incomplete tool call tags
        text = re.sub(r'<tool_call>.*$', '', text, flags=re.DOTALL)
        # Remove [Tool Result] blocks that might be injected
        text = re.sub(r'\[Tool Result\].*?\n\n', '', text, flags=re.DOTALL)
        text = re.sub(r'\[Search Results for:.*?\n\n', '', text, flags=re.DOTALL)
        text = re.sub(r'\[Search Error:.*?\]', '', text)
        text = re.sub(r'\[Searching:.*?\]', '', text)
        text = re.sub(r'\[Current Time:.*?\]', '', text)
        text = re.sub(r'\[No search results.*?\]', '', text)
        # Remove duplicate thinking artifacts that might appear after tool results
        # (model sometimes repeats its thinking process)
        lines = text.split('\n')
        seen_lines = set()
        unique_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines at the start
            if not unique_lines and not stripped:
                continue
            # Skip duplicate lines (common in tool call responses)
            if stripped and stripped in seen_lines and len(stripped) > 20:
                continue
            seen_lines.add(stripped)
            unique_lines.append(line)
        text = '\n'.join(unique_lines)
        return text.strip()
    
    def _update_tps(self, tps: str) -> None:
        """Update tokens per second below the input"""
        try:
            self.query_one("#token-stats", TokenStats).update(tps)
        except Exception:
            pass
    
    def _clear_tps(self) -> None:
        """Clear tokens per second display"""
        try:
            self.query_one("#token-stats", TokenStats).update("")
        except Exception:
            pass

if __name__ == "__main__":
    ChatApp().run()
