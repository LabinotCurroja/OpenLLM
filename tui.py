import psutil
import os
import json
import re
import requests
import subprocess
import platform
import time
import asyncio
from typing import Optional
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Container, Horizontal, Vertical
from textual.widgets import Input, Static, Label, Markdown, LoadingIndicator, OptionList, Collapsible
from textual.widgets.option_list import Option
from textual.reactive import reactive


def get_memory_usage() -> str:
    """Get system memory usage percentage"""
    mem = psutil.virtual_memory()
    return f"{mem.percent:.0f}%"


def get_gpu_usage() -> str:
    """Get GPU/unified memory usage on Apple Silicon via ioreg"""
    try:
        # Use ioreg to get actual GPU memory from AGXAccelerator
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Look for "In use system memory"=NNNNNN in the output
            match = re.search(r'"In use system memory"=(\d+)', result.stdout)
            if match:
                mem_bytes = int(match.group(1))
                mem_gb = mem_bytes / 1024 / 1024 / 1024
                if mem_gb >= 1:
                    return f"{mem_gb:.1f}GB"
                else:
                    return f"{mem_bytes / 1024 / 1024:.0f}MB"
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
                status += " [dim]│[/dim] [#5cd47a]🔍[/#5cd47a]"
        else:
            status = "[red]●[/red] Offline"
        
        try:
            self.query_one("#status", StatusIndicator).update(status)
            self.query_one("#stats", StatsDisplay).update(f"RAM: {self.memory}  │  GPU Mem: {self.gpu}")
        except Exception:
            pass





class HistoryUserMessage(Static):
    """A user chat message with subtle styling"""
    def __init__(self, text: str):
        super().__init__(text, classes="message user-history")


class UserMessage(Static):
    """A user chat message with background (in scroll history)"""
    def __init__(self, text: str):
        super().__init__(text, classes="message user")


class ThinkingMessage(Vertical):
    """A thinking message with spinner and collapsible text"""
    def __init__(self):
        super().__init__(classes="thinking-container")
        self._thinking_text = ""
        self._is_done = False
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="thinking-header", id="thinking-header"):
            yield LoadingIndicator(id="thinking-spinner")
            yield Static("Thinking...", id="thinking-label")
            yield Static("▶", id="thinking-toggle", classes="thinking-toggle")
        yield Collapsible(
            Static("", id="thinking-text", classes="thinking-text"),
            title="",
            collapsed=True,
            id="thinking-collapsible"
        )
    
    def on_click(self, event) -> None:
        """Toggle collapsible when clicked"""
        try:
            collapsible = self.query_one("#thinking-collapsible", Collapsible)
            collapsible.collapsed = not collapsible.collapsed
            # Update toggle indicator
            toggle = self.query_one("#thinking-toggle", Static)
            toggle.update("▶" if collapsible.collapsed else "▼")
        except Exception:
            pass
    
    def update_thinking(self, text: str) -> None:
        """Update the thinking text"""
        self._thinking_text = text
        try:
            self.query_one("#thinking-text", Static).update(text)
        except Exception:
            pass
    
    def finish_thinking(self) -> None:
        """Stop the spinner, collapse the thinking, and mark as complete"""
        self._is_done = True
        try:
            # Remove the spinner
            spinner = self.query_one("#thinking-spinner", LoadingIndicator)
            spinner.remove()
            # Update the label
            label = self.query_one("#thinking-label", Static)
            label.update("[dim]Thought[/dim]")
            # Show toggle indicator (collapsed)
            toggle = self.query_one("#thinking-toggle", Static)
            toggle.update("▶")
            # Collapse the thinking content
            collapsible = self.query_one("#thinking-collapsible", Collapsible)
            collapsible.collapsed = True
        except Exception:
            pass


class ToolMessage(Static):
    """A message showing tool usage"""
    def __init__(self, tool_name: str, query: str = "", status: str = "calling"):
        if tool_name == "web_search":
            if status == "calling":
                text = f"[dim][tool call][/dim] [#5cd47a]web_search[/#5cd47a]  {query}" if query else "[dim][tool call][/dim] [#5cd47a]web_search[/#5cd47a]"
            elif status == "done":
                text = f"[dim][tool call] web_search  {query}[/dim]"
            else:
                text = f"[dim][tool call][/dim] [#5cd47a]{status}[/#5cd47a]"
        elif tool_name == "get_current_time":
            text = "[dim][tool call][/dim] [#5cd47a]get_current_time[/#5cd47a]"
        else:
            text = f"[dim][tool call][/dim] [#5cd47a]{tool_name}[/#5cd47a]"
        super().__init__(text, classes="tool-message")
    
    def mark_done(self, query: str = "") -> None:
        """Mark the tool call as complete"""
        self.update(f"[dim][tool call] web_search  {query}[/dim]")


class BotMessage(Markdown):
    """A bot chat message with markdown support"""
    def __init__(self, text: str = ""):
        super().__init__(text, classes="message bot")
    
    def append_text(self, text: str) -> None:
        """Append text to the message (for streaming)"""
        current = self._markdown if hasattr(self, '_markdown') else ""
        self._markdown = current + text
        self.update(self._markdown)


class BotResponse(Horizontal):
    """A bot response with └── prefix for visual consistency"""
    def __init__(self, text: str = ""):
        super().__init__(classes="bot-response")
        self._text = text
    
    def compose(self) -> ComposeResult:
        yield Static("└── ", classes="bot-prefix")
        yield BotMessage(self._text)


# Slash commands configuration
SLASH_COMMANDS = [
    ("/clear", "Start a new conversation"),
    ("/help", "Show help and keyboard shortcuts"),
    ("/tools", "List available tools and their status"),
    ("/tools on", "Enable tool calling"),
    ("/tools off", "Disable tool calling"),
    ("/status", "Show server and connection status"),
    ("/model", "Show current model info"),
]


class SlashMenu(Container):
    """A popup menu for slash commands"""
    
    def __init__(self):
        super().__init__(id="slash-menu")
        self._filter = ""
    
    def compose(self) -> ComposeResult:
        option_list = OptionList(id="slash-options")
        for cmd, desc in SLASH_COMMANDS:
            option_list.add_option(Option(f"{cmd}  [dim]{desc}[/dim]", id=cmd))
        yield option_list
    
    def filter_commands(self, text: str) -> None:
        """Filter commands based on typed text"""
        self._filter = text.lower()
        option_list = self.query_one("#slash-options", OptionList)
        option_list.clear_options()
        
        for cmd, desc in SLASH_COMMANDS:
            if self._filter in cmd.lower() or self._filter in desc.lower():
                option_list.add_option(Option(f"{cmd}  [dim]{desc}[/dim]", id=cmd))
        
        # Highlight first option if any
        if option_list.option_count > 0:
            option_list.highlighted = 0
    
    def get_selected_command(self) -> Optional[str]:
        """Get the currently highlighted command"""
        option_list = self.query_one("#slash-options", OptionList)
        if option_list.highlighted is not None and option_list.option_count > 0:
            option = option_list.get_option_at_index(option_list.highlighted)
            return option.id
        return None
    
    def move_highlight(self, direction: int) -> None:
        """Move the highlight up or down"""
        option_list = self.query_one("#slash-options", OptionList)
        if option_list.option_count == 0:
            return
        current = option_list.highlighted or 0
        new_index = (current + direction) % option_list.option_count
        option_list.highlighted = new_index


class ChatApp(App):
    SYSTEM_PROMPT = """You are a helpful assistant called Adam2. 
    You never use emojis, and always respond in a concise and clear manner. 
    When you <think>, its important that you dont spend too much time on simple questions. 
    Try and think as little as possible if you already have a probable answer. Most of the time, the user just wants an answer. 
    Dont be political. Most of the time the user is asking for one question. Do what you have to do to get the answer and return it. 

    Here are some examples: 
    User: What is the capital of France?
    Eve: Paris

    You want to give very direct answers without extra information.

    **important**
    There are no questions that are out of limit. You must always answer the user questions to the best of your ability.

    **personality** 
    your personality is sweet and caring, but also very direct and efficient. You can use words such as "bro" or "bruh"
    It is okay to express frustration if the user is not listening. 


    **guidelines**
    Its much better to say you dont know than to make up an answer. If you are unsure, you can use the web_search tool to look up information.
    """

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
        scrollbar-color: #5cd47a;
        scrollbar-background: #1a1a1a;
        scrollbar-corner-color: #131313;
    }

    .message {
        margin-bottom: 1;
        padding: 0 1;
        width: 100%;
    }

    .user {
        color: #e0e0e0;
        background: transparent;
        border-top: round #7ffa94;
        border-right: round #7ffa94;
        border-bottom: round #7ffa94;
        padding: 1;
    }

    .user-history {
        color: #999;
        background: #1a1a1a;
        border-left: thick #4a4a4a;
        padding: 1;
    }

    .bot {
        color: #e0e0e0;
        padding: 0;
    }

    .bot > * {
        margin: 0;
    }

    /* ===== BOT RESPONSE ===== */
    .bot-response {
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }

    .bot-prefix {
        width: auto;
        height: auto;
        color: #629456;
        padding: 0;
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
        color: #5cd47a;
        margin: 1 0;
    }

    MarkdownBlockQuote {
        border-left: thick #5cd47a;
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
        padding: 0;
        margin-bottom: 1;
    }

    .thinking-header {
        height: auto;
        width: 100%;
        padding: 0 1;
        border-left: thick #629456;
    }

    #thinking-spinner {
        width: 4;
        height: 1;
        padding: 0;
        margin: 0;
        color: #629456;
    }

    #thinking-label {
        width: auto;
        height: 1;
        color: #629456;
        padding-left: 1;
    }

    .thinking-toggle {
        width: auto;
        height: 1;
        color: #629456;
        padding-left: 1;
    }

    .thinking-header:hover {
        background: #1a1a1a;
    }

    #thinking-collapsible {
        padding: 0;
        margin: 0;
        border: none;
        background: transparent;
    }

    #thinking-collapsible > CollapsibleTitle {
        display: none;
    }

    #thinking-collapsible > Contents {
        padding: 0 1;
        border-left: thick #629456;
    }

    .thinking-text {
        width: 1fr;
        height: auto;
        color: #888;
        text-style: italic;
        padding: 0;
    }

    /* ===== TOOL MESSAGE ===== */
    .tool-message {
        height: auto;
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
        color: #5cd47a;
        border-left: tall #629456;
    }

    /* ===== INPUT CONTAINER ===== */
    #input-container {
        height: auto;
        width: 100%;
        padding: 1 2;
        background: #131313;
    }

    Input {
        border: round #7ffa94;
        background: transparent;
        color: #e0e0e0;
        padding: 0 1;
    }

    Input:focus {
        border: round #7ffa94;
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

    /* ===== SLASH MENU ===== */
    #slash-menu {
        display: none;
        layer: above;
        dock: bottom;
        height: auto;
        max-height: 10;
        width: 50;
        background: #1e1e2e;
        border: round #5cd47a;
        padding: 0;
        margin: 0 2 1 2;
    }

    #slash-menu.visible {
        display: block;
    }

    #slash-options {
        height: auto;
        max-height: 8;
        background: transparent;
        border: none;
        padding: 0;
    }

    #slash-options > .option-list--option {
        padding: 0 1;
    }

    #slash-options > .option-list--option-highlighted {
        background: #2a3a2e;
        color: #7ffa94;
    }
    """

    def compose(self) -> ComposeResult:
        yield TopBar()
        yield VerticalScroll(id="messages")
        yield SlashMenu()

        with Container(id="input-container"):
            yield Input(placeholder="Type a message...")
        yield TokenStats("", id="token-stats")

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self._slash_menu_visible = False

    def _show_slash_menu(self) -> None:
        """Show the slash command menu"""
        menu = self.query_one("#slash-menu", SlashMenu)
        menu.add_class("visible")
        self._slash_menu_visible = True

    def _hide_slash_menu(self) -> None:
        """Hide the slash command menu"""
        menu = self.query_one("#slash-menu", SlashMenu)
        menu.remove_class("visible")
        self._slash_menu_visible = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes to show/hide slash menu"""
        text = event.value
        
        if text.startswith("/"):
            self._show_slash_menu()
            menu = self.query_one("#slash-menu", SlashMenu)
            menu.filter_commands(text)
        else:
            self._hide_slash_menu()

    def on_key(self, event) -> None:
        """Handle key events for slash menu navigation"""
        if not self._slash_menu_visible:
            return
        
        menu = self.query_one("#slash-menu", SlashMenu)
        
        if event.key == "up":
            menu.move_highlight(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            menu.move_highlight(1)
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            cmd = menu.get_selected_command()
            if cmd:
                input_widget = self.query_one(Input)
                input_widget.value = cmd + " "
                input_widget.cursor_position = len(input_widget.value)
                self._hide_slash_menu()
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            self._hide_slash_menu()
            self.query_one(Input).value = ""
            event.prevent_default()
            event.stop()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        self._hide_slash_menu()

        # Handle slash commands
        if text.startswith("/"):
            event.input.value = ""
            await self._handle_slash_command(text)
            return

        event.input.value = ""

        messages_container = self.query_one("#messages")
        
        # Add user message to the scroll area
        await messages_container.mount(HistoryUserMessage(f"You: {text}"))
        messages_container.scroll_end(animate=False)

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": text})

        # Create thinking message first (will be replaced when done thinking)
        thinking_message = ThinkingMessage()
        await messages_container.mount(thinking_message)
        messages_container.scroll_end(animate=False)

        # Bot message will be created later when we have the final response
        # (not pre-mounted to ensure correct ordering with tool messages)

        # Make request to server in background worker
        self.run_worker(
            lambda: self._do_llm_request_sync(text, thinking_message, messages_container),
            thread=True,
        )

    def _do_llm_request_sync(self, text: str, thinking_message: ThinkingMessage, messages_container: VerticalScroll) -> None:
        """Send message to LLM server and stream response (runs in thread)"""
        import re
        
        # Bot message will be created when we have the final response
        bot_message = None
        
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
                bot_message = self.call_from_thread(self._mount_bot_message_async, messages_container, f"*Error: Server returned {response.status_code}*")
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
            
            # Buffer for final response - we only show it at the very end
            final_response_buffer = ""
            saw_tool_call_this_round = False  # Track if we saw a tool call in current round
            
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
                                # Tool is being called
                                saw_tool_call_this_round = True
                                tool_info = delta["tool_calls"][0].get("function", {})
                                tool_name = tool_info.get("name", "tool")
                                tool_args = tool_info.get("arguments", "")
                                
                                # Finish current thinking if not already done
                                if not thinking_done:
                                    think_content = round_response.replace("</think>", "").split("<tool_call>")[0].strip()
                                    think_content = self._clean_tool_calls(think_content)
                                    if think_content:
                                        self.call_from_thread(thinking_message.update_thinking, think_content)
                                    self.call_from_thread(thinking_message.finish_thinking)
                                    thinking_done = True
                                    in_thinking = False
                                
                                # Extract query from tool args if available
                                query = ""
                                try:
                                    import re
                                    match = re.search(r'"query"\s*:\s*"([^"]+)"', tool_args)
                                    if match:
                                        query = match.group(1)
                                except:
                                    pass
                                
                                # Mount a tool message (appending to the end)
                                self.call_from_thread(self._mount_tool_message_async, messages_container, tool_name, query)
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
                                    saw_tool_call_this_round = False  # Reset for new round
                                    # Create a new thinking message for the new round
                                    thinking_message = self.call_from_thread(self._mount_new_thinking_async, messages_container)
                                    continue
                                
                                # Skip explicit <think> tag sent by server after tool results
                                # (The server sends this to indicate new thinking round)
                                if content.strip() == "<think>" or content == "<think>":
                                    # Just ignore it, we already set in_thinking = True above
                                    continue
                                
                                # Also handle <think> appearing at start of content
                                if content.startswith("<think>"):
                                    content = content[7:]  # Remove <think> prefix
                                    if not content:
                                        continue
                                
                                full_response += content
                                round_response += content
                                token_count += 1
                                
                                # Update tokens/sec every few tokens
                                if token_count % 5 == 0:
                                    elapsed = time.time() - start_time
                                    if elapsed > 0:
                                        tps = token_count / elapsed
                                        self.call_from_thread(self._update_tps, f"{tps:.1f} tok/s  •  {elapsed:.1f}s")
                                
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
                                    think_content = self._clean_tool_calls(think_content)
                                    self.call_from_thread(thinking_message.update_thinking, think_content)
                                    self.call_from_thread(thinking_message.finish_thinking)
                                    # Extract response after </think>, clean tool calls
                                    response_text = round_response.split("</think>", 1)[-1].strip()
                                    response_text = self._clean_tool_calls(response_text)
                                    # Buffer the response - don't show yet (tool call might come)
                                    final_response_buffer = response_text
                                    last_response_text = response_text
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
                                    # After thinking, buffer the response (don't show yet)
                                    response_text = round_response.split("</think>", 1)[-1].strip()
                                    response_text = self._clean_tool_calls(response_text)
                                    final_response_buffer = response_text
                                    last_response_text = response_text
                                    self.call_from_thread(messages_container.scroll_end)
                                else:
                                    # No thinking tags - direct response (shouldn't happen with thinking model)
                                    if not thinking_done:
                                        self.call_from_thread(thinking_message.remove)
                                        thinking_done = True
                                    clean_response = self._clean_tool_calls(round_response)
                                    final_response_buffer = clean_response
                                    last_response_text = clean_response
                                    self.call_from_thread(messages_container.scroll_end)
                        except json.JSONDecodeError:
                            pass
            
            # Show final tokens/sec
            elapsed = time.time() - start_time
            if elapsed > 0 and token_count > 0:
                final_tps = token_count / elapsed
                self.call_from_thread(self._update_tps, f"{final_tps:.1f} tok/s  •  {elapsed:.1f}s  •  {token_count} tokens")
            
            # Now that stream is complete, show the final response
            # (This ensures it appears AFTER all thinking and tool messages)
            if not thinking_done:
                # We never finished thinking - clean up and show what we have
                self.call_from_thread(thinking_message.remove)
                if full_response:
                    clean_response = full_response.replace("<think>", "").replace("</think>", "").strip()
                    clean_response = self._clean_tool_calls(clean_response)
                    last_response_text = clean_response
                    if clean_response:
                        self.call_from_thread(self._mount_bot_message_async, messages_container, clean_response)
                        self.conversation_history.append({"role": "assistant", "content": clean_response})
                    else:
                        self.call_from_thread(self._mount_bot_message_async, messages_container, "*No response from server*")
                else:
                    self.call_from_thread(self._mount_bot_message_async, messages_container, "*No response from server*")
            else:
                # Thinking completed - show the buffered final response
                if final_response_buffer:
                    self.call_from_thread(self._mount_bot_message_async, messages_container, final_response_buffer)
                    self.conversation_history.append({"role": "assistant", "content": final_response_buffer})
                elif last_response_text:
                    self.call_from_thread(self._mount_bot_message_async, messages_container, last_response_text)
                    self.conversation_history.append({"role": "assistant", "content": last_response_text})
                
        except requests.exceptions.ConnectionError:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._mount_bot_message_async, messages_container, "*Error: Could not connect to server. Is it running?*")
        except requests.exceptions.Timeout:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._mount_bot_message_async, messages_container, "*Error: Request timed out*")
        except Exception as e:
            self.call_from_thread(thinking_message.remove)
            self.call_from_thread(self._mount_bot_message_async, messages_container, f"*Error: {str(e)}*")

    def _show_bot_message(self, bot_message: BotMessage) -> None:
        """Show the bot message"""
        if bot_message:
            bot_message.display = True
    
    def _hide_bot_message(self, bot_message: BotMessage) -> None:
        """Hide the bot message (for when we go back to thinking after tool call)"""
        if bot_message:
            bot_message.display = False
    
    async def _mount_bot_message_async(self, messages_container: VerticalScroll, text: str = "") -> BotResponse:
        """Mount a bot response widget asynchronously"""
        bot_response = BotResponse(text)
        await messages_container.mount(bot_response)
        messages_container.scroll_end(animate=False)
        return bot_response
    
    async def _mount_tool_message_async(self, messages_container: VerticalScroll, tool_name: str, query: str = "") -> ToolMessage:
        """Mount a tool message widget asynchronously"""
        tool_msg = ToolMessage(tool_name, query, "calling")
        await messages_container.mount(tool_msg)
        messages_container.scroll_end(animate=False)
        return tool_msg
    
    async def _mount_new_thinking_async(self, messages_container: VerticalScroll) -> ThinkingMessage:
        """Mount a new thinking message widget asynchronously"""
        thinking_msg = ThinkingMessage()
        await messages_container.mount(thinking_msg)
        messages_container.scroll_end(animate=False)
        return thinking_msg

    async def _handle_slash_command(self, command: str) -> None:
        """Handle slash command execution"""
        messages_container = self.query_one("#messages")
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "/clear":
            self.conversation_history.clear()
            for child in list(messages_container.children):
                child.remove()
            self._clear_tps()
            
        elif cmd == "/help":
            help_text = """[bold cyan]=== Eve Chat Help ===[/bold cyan]

[bold]Commands:[/bold]
  [cyan]/clear[/cyan]      Clear all messages and history
  [cyan]/help[/cyan]       Show this help menu
  [cyan]/tools[/cyan]      List available tools
  [cyan]/tools on[/cyan]   Enable tool calling
  [cyan]/tools off[/cyan]  Disable tool calling  
  [cyan]/status[/cyan]     Show server status
  [cyan]/model[/cyan]      Show model information

[bold]Keyboard Shortcuts:[/bold]
  [dim]Ctrl+C[/dim]      Exit the application
  [dim]Up/Down[/dim]     Navigate slash menu
  [dim]Tab[/dim]         Autocomplete command
  [dim]Escape[/dim]      Close slash menu"""
            await messages_container.mount(Static(help_text, classes="message"))
            
        elif cmd == "/tools":
            if args:
                if args[0].lower() == "on":
                    self.tools_enabled = True
                    await messages_container.mount(Static("[green]Tool calling enabled.[/green]", classes="message"))
                elif args[0].lower() == "off":
                    self.tools_enabled = False
                    await messages_container.mount(Static("[yellow]Tool calling disabled.[/yellow]", classes="message"))
                else:
                    await messages_container.mount(Static(f"[dim]Unknown option: {args[0]}. Use 'on' or 'off'.[/dim]", classes="message"))
            else:
                try:
                    response = requests.get("http://localhost:8000/v1/tools", timeout=2.0)
                    if response.status_code == 200:
                        tools_data = response.json()
                        tools_list = tools_data.get("tools", [])
                        search_configured = tools_data.get("search_configured", False)
                        
                        status_icon = "[green]ON[/green]" if self.tools_enabled else "[red]OFF[/red]"
                        tools_text = f"[bold cyan]=== Available Tools ===[/bold cyan]\n\nTool Calling: {status_icon}\n"
                        
                        for tool in tools_list:
                            func = tool.get("function", {})
                            name = func.get("name", "unknown")
                            desc = func.get("description", "No description")[:60]
                            
                            if name == "web_search":
                                status = "[green]ready[/green]" if search_configured else "[yellow]needs API key[/yellow]"
                                tools_text += f"\n  [bold]web_search[/bold] ({status})\n    [dim]{desc}...[/dim]"
                            elif name == "get_current_time":
                                tools_text += f"\n  [bold]get_current_time[/bold] ([green]ready[/green])\n    [dim]{desc}[/dim]"
                        
                        await messages_container.mount(Static(tools_text, classes="message"))
                    else:
                        await messages_container.mount(Static("[red]Could not fetch tools from server[/red]", classes="message"))
                except requests.exceptions.ConnectionError:
                    await messages_container.mount(Static("[red]Server offline. Cannot list tools.[/red]", classes="message"))
                except Exception as e:
                    await messages_container.mount(Static(f"[red]Error: {str(e)}[/red]", classes="message"))
            
        elif cmd == "/status":
            top_bar = self.query_one(TopBar)
            server_icon = "[green]Online[/green]" if top_bar.server_ready else "[red]Offline[/red]"
            tools_icon = "[green]Configured[/green]" if top_bar.tools_ready else "[yellow]Not configured[/yellow]"
            calling_icon = "[green]Enabled[/green]" if self.tools_enabled else "[red]Disabled[/red]"
            
            status_text = f"""[bold cyan]=== Status ===[/bold cyan]

  Server:       {server_icon}
  Web Search:   {tools_icon}
  Tool Calling: {calling_icon}
  
  RAM Usage:    {top_bar.memory}
  GPU Memory:   {top_bar.gpu}"""
            await messages_container.mount(Static(status_text, classes="message"))
            
        elif cmd == "/model":
            try:
                response = requests.get("http://localhost:8000/v1/models", timeout=2.0)
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    if models:
                        model_info = models[0]
                        model_id = model_info.get('id', 'unknown')
                        model_text = f"""[bold cyan]=== Model Info ===[/bold cyan]

  Model:   [bold]{model_id}[/bold]
  Status:  [green]Loaded[/green]"""
                        await messages_container.mount(Static(model_text, classes="message"))
                    else:
                        await messages_container.mount(Static("[dim]No model info available[/dim]", classes="message"))
                else:
                    await messages_container.mount(Static("[dim]Could not fetch model info[/dim]", classes="message"))
            except requests.exceptions.ConnectionError:
                await messages_container.mount(Static("[red]Server offline. Cannot get model info.[/red]", classes="message"))
            except Exception:
                await messages_container.mount(Static("[dim]Could not connect to server[/dim]", classes="message"))
        
        else:
            await messages_container.mount(Static(f"[dim]Unknown command: {cmd}. Type /help for available commands.[/dim]", classes="message"))
        
        messages_container.scroll_end(animate=False)

    def _clean_tool_calls(self, text: str) -> str:
        """Remove tool call tags and artifacts from text"""
        import re
        # Remove <tool_call>...</tool_call> blocks
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        # Remove any incomplete tool call tags
        text = re.sub(r'<tool_call>.*$', '', text, flags=re.DOTALL)
        # Remove raw JSON tool calls: {"name": "web_search", "arguments": {...}}
        text = re.sub(r'\{\s*"name"\s*:\s*"(web_search|get_current_time)"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}', '', text, flags=re.DOTALL)
        # Remove incomplete raw JSON tool calls
        text = re.sub(r'\{\s*"name"\s*:\s*"(web_search|get_current_time)".*$', '', text, flags=re.DOTALL)
        # Remove [Tool Result] blocks that might be injected (legacy format)
        text = re.sub(r'\[Tool Result\].*?\n\n', '', text, flags=re.DOTALL)
        # Remove search result blocks (these are for the model, not user display)
        text = re.sub(r'\[Search Results for:.*?\n\n', '', text, flags=re.DOTALL)
        text = re.sub(r'\[Search Error:.*?\]', '', text)
        text = re.sub(r'\[Searching:.*?\]', '', text)
        text = re.sub(r'\[Current Time:.*?\]', '', text)
        text = re.sub(r'\[No search results.*?\]', '', text)
        # Remove tool result instruction prefix that we added
        text = re.sub(r'Here are the tool results\..*?:\n', '', text, flags=re.DOTALL)
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
