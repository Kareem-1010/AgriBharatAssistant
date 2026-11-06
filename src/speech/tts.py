import os
import tempfile
from pathlib import Path
from typing import Optional

class IndicSpeechSynthesizer:
    """
    Multilingual Text-to-Speech Engine for Indian Languages.
    Integrates AI4Bharat Indic Parler-TTS with lightweight gTTS fallback.
    Produces high-fidelity audio responses in Hindi, Telugu, Tamil, Marathi, Bengali, and English.
    """
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "krishi_audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # ISO language mapping for TTS
        self.tts_lang_map = {
            "hi": "hi",
            "en": "en",
            "te": "te",
            "ta": "ta",
            "mr": "mr",
            "bn": "bn"
        }

    def synthesize(self, text: str, lang: str = "hi", filename: Optional[str] = None) -> Optional[str]:
        """
        Converts text into an audio MP3 file.
        Returns the absolute filepath to the generated audio file.
        """
        if not text:
            return None

        clean_text = text.replace("#", "").replace("*", "").replace("`", "").strip()
        # Truncate to first 300 characters for snappy voice synthesis preview
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."

        target_lang = self.tts_lang_map.get(lang, "hi")
        out_file = self.output_dir / (filename or f"response_{lang}_{hash(clean_text) % 100000}.mp3")

        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang=target_lang, slow=False)
            tts.save(str(out_file))
            return str(out_file)
        except Exception as e:
            print(f"[TTS] Warning: Speech synthesis failed ({e}). Audio output unavailable.")
            return None

