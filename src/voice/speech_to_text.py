"""
Speech recognition using Faster Whisper.

Supports:
- English
- Persian

"""

from pathlib import Path

from faster_whisper import WhisperModel

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMP_DIR = (
    PROJECT_ROOT
    / "data"
    / "temp"
)


class SpeechRecognizer:

    def __init__(
        self,
        model_size: str = "medium",
    ):
        """
        Initialize Whisper model.

        Available models:

        tiny
        base
        small
        medium
        large-v3
        """

        TEMP_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(
        self,
        audio_path,
    ):
        """
        Convert speech to text.

        Returns:
            dict
        """

        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
        )

        text_parts = []

        for segment in segments:

            if segment.avg_logprob < -1.0:
                continue

            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()

        BAD_TEXTS = {
            "",
            ".",
            "...",
            "thanks for watching",
            "thank you for watching",
            "thanks for watching.",
            "thank you for watching.",
            "bye",
            "goodbye",
        }

        if (
            len(text) < 3
            or text.lower() in BAD_TEXTS
        ):
            text = ""

        return {
            "text": text,
            "language": info.language,
            "probability": round(
                info.language_probability,
                3,
            ),
        }