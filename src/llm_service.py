"""
Handles communication with the local Qwen model through Ollama.
"""

from typing import List, Dict, Optional
from ollama import Client

class LLMService:
    """
    A wrapper around the Ollama client for interacting with
    the local Qwen language model.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        # model: str = "qwen3:14b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
    ):
        """
        Initialize the LLM service.

        Args:
            model: Ollama model name.
            host: Ollama server URL.
            temperature: Generation temperature.
        """

        self.model = model
        self.temperature = temperature

        # Connect to the local Ollama server
        self.client = Client(host=host)

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict]:
        """
        Build the conversation messages.
        """

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful research assistant.",
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Send a prompt to the local model.

        Args:
            user_prompt: User question.
            system_prompt: System instruction.
            history: Previous conversation messages (optional).

        Returns:
            Model response.
        """
        messages = []

        if history:
            messages.extend(history)
        else:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        try:
            response = self.client.chat(
            model=self.model,
            messages=messages,
            think=False,
            options={
                "temperature": self.temperature,
            },
)
            return response["message"]["content"].strip()
            

        except Exception as e:

            print(e)

    def health_check(self) -> bool:
        """
        Verify that the model is available.

        Returns:
            True if the model responds successfully.
        """

        try:

            reply = self.chat("Hello!")
            return len(reply) > 0

        except Exception:
            return False