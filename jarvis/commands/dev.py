# jarvis/commands/dev.py

import subprocess
from typing import Tuple


def try_handle(text: str) -> Tuple[bool, str]:
    if "run tests" in text:
        # Example: run pytest
        try:
            subprocess.Popen(["pytest"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, "Running tests."
        except FileNotFoundError:
            return True, "Pytest not found. Make sure it is installed."

    if "git status" in text:
        # You might want to parse and summarize instead of printing raw
        try:
            result = subprocess.run(["git", "status", "-sb"], capture_output=True, text=True, timeout=5)
            status_summary = result.stdout[:200].strip()
            return True, f"Here is the git status: {status_summary}"
        except subprocess.TimeoutExpired:
            return True, "Git status command timed out."
        except Exception as e:
            return True, f"Error getting git status: {str(e)[:50]}"

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
