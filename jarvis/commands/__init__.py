# jarvis/commands/__init__.py

from typing import Optional, Tuple

from ..memory import ConversationMemory
from ..config import LLM_ENABLED
from . import system, dev, fun

# Lazy init LLM so Jarvis still works without it if you want
_llm_agent: Optional['LocalLLMAgent'] = None


def _get_llm_agent():
    """Lazy-load LLM agent (singleton)."""
    global _llm_agent
    if _llm_agent is None:
        from ..llm_agent import LocalLLMAgent
        _llm_agent = LocalLLMAgent()
    return _llm_agent


def handle_command(text: str, memory: ConversationMemory) -> Tuple[str, Optional[str]]:
    """
    Dispatch command text to appropriate handler.
    Returns:
        (response_to_speak, debug_module_name_or_None)
    """
    t = text.lower().strip()
    if not t:
        return ("I did not hear anything useful.", None)

    # 1) Try explicit command modules first
    for module in (system, dev, fun):
        handled, reply = module.try_handle(t, memory)
        if handled:
            return (reply, module.__name__)

    # 2) If no explicit handler and LLM is enabled, ask the model
    if LLM_ENABLED:
        try:
            llm = _get_llm_agent()
            reply = llm.generate_reply(memory, text)
            return (reply, "llm_agent")
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            return (f"You said: {text}", None)

    # 3) Fallback: simple echo
    return (f"You said: {text}", None)
