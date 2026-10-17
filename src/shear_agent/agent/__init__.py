"""Agent layer (C): tactile tools, an LLM client, and the tool-use loop."""
from .llm import LLMClient, LLMConfigError, LLMError, LLMReply, MockLLM
from .loop import SYSTEM_PROMPT, AgentResult, run_agent
from .tools import SKILLS, TOOL_NAMES, TOOLS, MockBackend, dispatch, validate

__all__ = [
    "AgentResult",
    "LLMClient",
    "LLMConfigError",
    "LLMError",
    "LLMReply",
    "MockBackend",
    "MockLLM",
    "SKILLS",
    "SYSTEM_PROMPT",
    "TOOLS",
    "TOOL_NAMES",
    "dispatch",
    "run_agent",
    "validate",
]
