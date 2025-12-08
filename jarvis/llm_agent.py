# jarvis/llm_agent.py

from typing import List
import json
import urllib.request
import urllib.error

from .config import (
    LLM_API_URL,
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_CONTEXT_TURNS,
    LLM_SYSTEM_PROMPT,
    DEBUG,
)
from .memory import ConversationMemory, Turn


class LocalLLMAgent:
    """
    LLM agent that uses an OpenAI-compatible API endpoint (e.g., vLLM server on port 9020).
    This avoids loading the model locally and prevents GPU memory issues.
    """
    
    def __init__(self):
        self.api_url = LLM_API_URL
        self.model_name = LLM_MODEL_NAME
        
        # Test connection
        try:
            models_url = self.api_url.replace("/v1/chat/completions", "/v1/models")
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                pass  # Just check if it connects
            print(f"🧠 Using LLM API: {self.api_url} (model: {self.model_name})")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to LLM API at {self.api_url}: {e}\n"
                f"Make sure the API server is running on port 9020."
            )

    # -------- prompt packing using ConversationMemory -------- #

    def _build_chat_messages(self, memory: ConversationMemory, user_text: str):
        """
        Build chat-style messages list: [{role, content}, ...]
        including system, facts, recent turns, and current user message.
        """
        messages = []

        # System message
        messages.append({"role": "system", "content": LLM_SYSTEM_PROMPT})

        # Facts block
        if memory.facts:
            facts_str = "\n".join(f"- {k}: {v}" for k, v in memory.facts.items())
            messages.append(
                {
                    "role": "system",
                    "content": f"Here are some known facts about the user and environment:\n{facts_str}",
                }
            )

        # Recent conversation
        recent_turns: List[Turn] = memory.get_recent_context(LLM_CONTEXT_TURNS)
        for turn in recent_turns:
            # map our roles to chat roles
            role = "user" if turn.role == "user" else "assistant"
            messages.append({"role": role, "content": turn.text})

        # Current user message (latest turn)
        messages.append({"role": "user", "content": user_text})

        return messages

    # -------- main public API -------- #

    def generate_reply(self, memory: ConversationMemory, user_text: str) -> str:
        """
        Use LLM API to generate a reply conditioned on conversation memory.
        """
        messages = self._build_chat_messages(memory, user_text)
        
        # Prepare API request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                if "choices" in result and len(result["choices"]) > 0:
                    reply = result["choices"][0]["message"]["content"]
                    return reply.strip()
                else:
                    if DEBUG:
                        print(f"⚠️ Unexpected API response: {result}")
                    return "I'm having trouble generating a response right now."
                    
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"LLM API error ({e.code}): {error_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to call LLM API: {e}")

