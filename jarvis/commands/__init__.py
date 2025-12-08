# jarvis/commands/__init__.py

from typing import Optional, Tuple

from . import system, dev, fun


def handle_command(text: str) -> Tuple[str, Optional[str]]:
    """
    Dispatch command text to appropriate handler.
    Returns:
        (response_to_speak, debug_info_or_None)
    """
    t = text.lower().strip()
    if not t:
        return ("I did not hear anything useful.", None)

    # Order matters; system-level commands checked first
    for module in (system, dev, fun):
        handled, reply = module.try_handle(t)
        if handled:
            return (reply, module.__name__)

    # default fallback
    return (f"You said: {text}", None)
