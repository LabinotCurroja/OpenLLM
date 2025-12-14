"""
Tool Calling Support for Qwen3-4B
==================================
Provides web search and other tool capabilities.
Uses Brave Search API (free tier: 2,000 queries/month).
"""

import os
import re
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the project root (parent of server directory)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    # Also try server directory as fallback
    env_path_server = Path(__file__).parent / ".env"
    load_dotenv(env_path_server)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# ============================================================================
# Tool Definitions (OpenAI-compatible format)
# ============================================================================

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when you need up-to-date information, news, facts you're unsure about, or anything that might have changed recently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ============================================================================
# Brave Search API
# ============================================================================

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def web_search(query: str, count: int = 5) -> Dict[str, Any]:
    """
    Search the web using Brave Search API.
    
    Free tier: 2,000 queries/month, no credit card required.
    Sign up at: https://brave.com/search/api/
    
    Args:
        query: Search query
        count: Number of results (max 20)
    
    Returns:
        Dict with search results or error
    """
    if not BRAVE_API_KEY:
        return {
            "error": "BRAVE_API_KEY not set. Get a free API key at https://brave.com/search/api/",
            "results": []
        }
    
    try:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": min(count, 20),
            "text_decorations": False,
            "search_lang": "en"
        }
        
        response = requests.get(
            BRAVE_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=10
        )
        
        if response.status_code == 401:
            return {"error": "Invalid BRAVE_API_KEY", "results": []}
        elif response.status_code == 429:
            return {"error": "Rate limit exceeded. Free tier is 2,000 queries/month.", "results": []}
        elif response.status_code != 200:
            return {"error": f"Search failed with status {response.status_code}", "results": []}
        
        data = response.json()
        
        results = []
        
        # Extract web results
        web_results = data.get("web", {}).get("results", [])
        for item in web_results[:count]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            })
        
        # Include any instant answer / infobox if available
        infobox = data.get("infobox")
        if infobox:
            results.insert(0, {
                "title": infobox.get("title", "Quick Answer"),
                "url": infobox.get("url", ""),
                "description": infobox.get("description", infobox.get("long_desc", "")),
                "type": "infobox"
            })
        
        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }
        
    except requests.exceptions.Timeout:
        return {"error": "Search request timed out", "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}


def get_current_time() -> Dict[str, Any]:
    """Get current date and time."""
    now = datetime.now()
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "timezone": "local"
    }


# ============================================================================
# Tool Execution
# ============================================================================

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "get_current_time": get_current_time,
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given arguments."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {name}"}
    
    try:
        func = TOOL_FUNCTIONS[name]
        return func(**arguments)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


# ============================================================================
# Tool Call Detection & Parsing
# ============================================================================

# Pattern to detect tool calls in model output
# Format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL
)

# Alternative format the model might use
FUNCTION_CALL_PATTERN = re.compile(
    r'```tool_call\s*\n(\{.*?\})\s*\n```',
    re.DOTALL
)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from model output.
    
    Returns list of tool calls: [{"name": str, "arguments": dict}, ...]
    """
    tool_calls = []
    
    # Try primary pattern
    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            call_data = json.loads(match.group(1))
            if "name" in call_data:
                tool_calls.append({
                    "name": call_data["name"],
                    "arguments": call_data.get("arguments", {})
                })
        except json.JSONDecodeError:
            continue
    
    # Try alternative pattern
    if not tool_calls:
        for match in FUNCTION_CALL_PATTERN.finditer(text):
            try:
                call_data = json.loads(match.group(1))
                if "name" in call_data:
                    tool_calls.append({
                        "name": call_data["name"],
                        "arguments": call_data.get("arguments", {})
                    })
            except json.JSONDecodeError:
                continue
    
    return tool_calls


def has_tool_calls(text: str) -> bool:
    """Check if text contains any tool calls."""
    return bool(TOOL_CALL_PATTERN.search(text) or FUNCTION_CALL_PATTERN.search(text))


def remove_tool_calls(text: str) -> str:
    """Remove tool call tags from text."""
    text = TOOL_CALL_PATTERN.sub("", text)
    text = FUNCTION_CALL_PATTERN.sub("", text)
    return text.strip()


# ============================================================================
# System Prompt Enhancement
# ============================================================================

def get_tools_system_prompt() -> str:
    """
    Generate system prompt instructions for tool usage.
    """
    tools_desc = []
    for tool in AVAILABLE_TOOLS:
        func = tool["function"]
        params = func["parameters"].get("properties", {})
        param_desc = ", ".join([f'{k}: {v.get("description", "")}' for k, v in params.items()])
        tools_desc.append(f"- {func['name']}: {func['description']}")
        if param_desc:
            tools_desc.append(f"  Parameters: {param_desc}")
    
    return f"""
You have access to the following tools:

{chr(10).join(tools_desc)}

To use a tool, output a tool call in this exact format:
<tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>

Important guidelines:
- Use web_search when you need current information, recent events, facts you're uncertain about, or anything that changes over time
- When you make a tool call, you will receive the results in the next message
- After receiving tool results, provide a complete and helpful response that incorporates the information
- You can make multiple tool calls if needed
- Always cite or reference the information you found when answering
- If a search returns no results or an error, acknowledge this and provide what information you can from your knowledge

Example tool call:
<tool_call>{{"name": "web_search", "arguments": {{"query": "latest news about AI"}}}}</tool_call>
"""


def format_tool_result(name: str, result: Dict[str, Any]) -> str:
    """Format tool result for injection into conversation."""
    if name == "web_search":
        if result.get("error"):
            return f"\n[Search Error: {result['error']}]\n"
        
        if not result.get("results"):
            return f"\n[No search results found for: {result.get('query', 'unknown query')}]\n"
        
        formatted = f"\n[Search Results for: {result.get('query', '')}]\n"
        for i, r in enumerate(result["results"], 1):
            formatted += f"\n{i}. {r['title']}\n"
            formatted += f"   URL: {r['url']}\n"
            if r.get("description"):
                formatted += f"   {r['description']}\n"
        formatted += "\n"
        return formatted
    
    elif name == "get_current_time":
        return f"\n[Current Time: {result['date']}, {result['time']}]\n"
    
    else:
        return f"\n[Tool Result: {json.dumps(result, indent=2)}]\n"


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test tool parsing
    test_text = '''
    Let me search for that.
    <tool_call>{"name": "web_search", "arguments": {"query": "Python 3.12 new features"}}</tool_call>
    '''
    
    calls = parse_tool_calls(test_text)
    print("Parsed tool calls:", calls)
    
    # Test time tool
    print("\nCurrent time:", get_current_time())
    
    # Test search (only if API key is set)
    if BRAVE_API_KEY:
        print("\nTesting search...")
        result = web_search("Python programming")
        print(json.dumps(result, indent=2))
    else:
        print("\nSet BRAVE_API_KEY to test search functionality")
        print("Get free API key at: https://brave.com/search/api/")
