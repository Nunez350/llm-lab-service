# jarvis/commands/system.py

import subprocess
from typing import Tuple

from ..memory import ConversationMemory


def try_handle(text: str, memory: ConversationMemory) -> Tuple[bool, str]:
    """
    Return (handled, reply)
    """
    if "what time is it" in text or "time" in text:
        from datetime import datetime
        now = datetime.now().strftime("%I:%M %p")
        return True, f"It is {now}."

    if "what's the date" in text or "date" in text:
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")
        return True, f"Today's date is {today}."

    if "my name is " in text:
        # Extract name
        name = text.split("my name is", 1)[1].strip(" .")
        if name:
            memory.remember_fact("user_name", name)
            return True, f"Nice to meet you, {name}."
        return True, "I didn't catch your name."

    if "what's my name" in text or "what is my name" in text:
        name = memory.get_fact("user_name")
        if name:
            return True, f"Your name is {name}."
        else:
            return True, "I do not know your name yet."

    if "open firefox" in text:
        try:
            subprocess.Popen(["firefox"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, "Opening Firefox."
        except Exception:
            return True, "I tried to open Firefox, but something went wrong."

    if "how many messages" in text and "today" in text:
        # Example of using history length
        turns = len(memory.history)
        return True, f"So far we have exchanged {turns} messages this session."

    if "shutdown" in text and "computer" in text:
        # You probably want to add confirmation in a real system.
        return True, "I will not shut down the computer without explicit confirmation."

    return False, ""
