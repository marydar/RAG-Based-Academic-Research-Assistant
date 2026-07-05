# from src.llm_service import LLMService


# def main():

#     llm = LLMService()

#     print("Checking Ollama connection...")

#     if llm.health_check():
#         print("Connection successful!\n")
#     else:
#         print("Failed to connect.")
#         return

#     answer = llm.chat(
#         "Explain Retrieval-Augmented Generation in one paragraph."
#     )

#     print("Model Response:\n")
#     print(answer)


# if __name__ == "__main__":
#     main()
    
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_service import LLMService


def main():

    llm = LLMService()

    print("=== Testing Conversation History ===\n")

    # First user message
    history = [
        {
            "role": "user",
            "content": "My favorite programming language is Python."
        }
    ]

    # Assistant reply
    assistant_reply = llm.chat(
        user_prompt=history[-1]["content"]
    )

    print("Assistant:")
    print(assistant_reply)

    # Save assistant response
    history.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )

    # Second user message
    second_question = "What is my favorite programming language?"

    print("\nUser:")
    print(second_question)

    answer = llm.chat(
        user_prompt=second_question,
        history=history
    )

    print("\nAssistant:")
    print(answer)


if __name__ == "__main__":
    main()