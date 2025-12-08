# jarvis/memory.py

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    text: str


@dataclass
class ConversationMemory:
    max_turns: int = 20
    history: Deque[Turn] = field(default_factory=lambda: deque(maxlen=20))
    facts: Dict[str, str] = field(default_factory=dict)

    def add_turn(self, role: str, text: str):
        self.history.append(Turn(role=role, text=text))

    def get_recent_context(self, n: int = 5) -> List[Turn]:
        """Return up to last n turns, oldest first."""
        return list(self.history)[-n:]

    def remember_fact(self, key: str, value: str):
        self.facts[key] = value

    def get_fact(self, key: str, default=None):
        return self.facts.get(key, default)

    def debug_summary(self) -> str:
        """Short text summary useful for logging."""
        turns = len(self.history)
        facts = ", ".join(f"{k}={v}" for k, v in self.facts.items()) or "none"
        return f"{turns} turns, facts: {facts}"
