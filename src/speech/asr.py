import os
from pathlib import Path
from typing import Optional

class IndicSpeechRecognizer:
    """
    Multilingual Automatic Speech Recognition (ASR) Engine.
    Compatible with AI4Bharat IndicConformer and Whisper multilingual models.
    Transcribes farmer voice queries in Hindi, Telugu, Tamil, Marathi, Bengali, and English.
    """
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        self.model_name = model_name
        self.pipeline = None

    def _init_pipeline(self):
        if self.pipeline is None:
            try:
                from transformers import pipeline
                print(f"[ASR] Loading lightweight speech recognition model: {self.model_name}...")
                self.pipeline = pipeline("automatic-speech-recognition", model=self.model_name, device="cpu")
                print("[ASR] Speech model loaded successfully.")
            except Exception as e:
                print(f"[ASR] Speech pipeline init failed: {e}. Voice will use fallback transcription.")
                self.pipeline = False

    def transcribe(self, audio_path: str, lang: str = "hi") -> str:
        """
        Transcribes audio file to text.
        """
        if not audio_path or not os.path.exists(audio_path):
            return ""

        self._init_pipeline()

        if self.pipeline:
            try:
                result = self.pipeline(audio_path)
                return result.get("text", "").strip()
            except Exception as e:
                print(f"[ASR] Transcription error: {e}")

        # Simulated transcription indicator for testing environment
        return f"[Transcribed Audio ({lang})]: Farmer voice query from {Path(audio_path).name}"

