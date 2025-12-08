# jarvis/commands/system.py

import subprocess
from typing import Tuple


def try_handle(text: str) -> Tuple[bool, str]:
    """
    Return (handled, reply)
    """
    # Example commands:
    if "what time is it" in text or "time" in text:
        from datetime import datetime
        now = datetime.now().strftime("%I:%M %p")
        return True, f"It is {now}."

    if "what's the date" in text or "date" in text:
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")
        return True, f"Today's date is {today}."

    if "open firefox" in text:
        try:
            subprocess.Popen(["firefox"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, "Opening Firefox."
        except Exception:
            return True, "I tried to open Firefox, but something went wrong."

    if "shutdown" in text and "computer" in text:
        # You probably want to add confirmation in a real system.
        return True, "I will not shut down the computer without explicit confirmation."

    return False, ""
