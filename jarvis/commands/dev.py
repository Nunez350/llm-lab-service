# jarvis/commands/dev.py

import subprocess
from typing import Tuple

from ..memory import ConversationMemory


def try_handle(text: str, memory: ConversationMemory) -> Tuple[bool, str]:
    if "run tests" in text:
        # Example: run pytest
        try:
            subprocess.Popen(["pytest"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, "Running tests."
        except FileNotFoundError:
            return True, "Pytest not found. Make sure it is installed."

    if "git status" in text:
        try:
            result = subprocess.run(
                ["git", "status", "-sb"],
                capture_output=True,
                text=True,
                timeout=5
            )
            summary = result.stdout.strip().splitlines()[0] if result.stdout else "No status."
            # Remember last git status
            memory.remember_fact("last_git_status", summary)
            return True, f"Git status: {summary}"
        except subprocess.TimeoutExpired:
            return True, "Git status command timed out."
        except Exception as e:
            return True, f"Error getting git status: {str(e)[:50]}"

    if "what was the last git status" in text:
        last = memory.get_fact("last_git_status")
        if last:
            return True, f"The last git status I saw was: {last}"
        else:
            return True, "I don't have any git status stored yet."

    if "git log" in text:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True,
                text=True,
                timeout=5
            )
            log_summary = result.stdout.strip()
            return True, f"Recent commits: {log_summary[:200]}"
        except Exception:
            return True, "Could not retrieve git log."

    return False, ""
