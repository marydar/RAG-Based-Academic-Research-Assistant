from pathlib import Path

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.voice.speech_to_text import SpeechRecognizer


recognizer = SpeechRecognizer()

result = recognizer.transcribe(
    Path("data/temp/test.wav")
)

print()

print("Language :", result["language"])
print("Confidence :", result["probability"])
print()

print(result["text"])