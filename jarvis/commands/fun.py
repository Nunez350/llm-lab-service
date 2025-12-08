# jarvis/commands/fun.py

from typing import Tuple
import random


def try_handle(text: str) -> Tuple[bool, str]:
    if "tell me a joke" in text or "joke" in text:
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are only ten types of people in the world: those who understand binary and those who do not.",
            "How do you comfort a JavaScript bug? You console it.",
            "Why did the Python programmer not respond? They were stuck in an infinite loop.",
            "What do you call a programmer from Finland? Nerdic.",
        ]
        return True, random.choice(jokes)

    if "how are you" in text:
        responses = [
            "I am operating within expected parameters.",
            "All systems nominal.",
            "I am functioning correctly.",
            "Everything is working as designed.",
        ]
        return True, random.choice(responses)

    if "hello" in text or "hi" in text:
        return True, "Hello! How can I help you?"

    if "goodbye" in text or "bye" in text:
        return True, "Goodbye! Have a great day."

    return False, ""
