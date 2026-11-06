"""
KrishiMitra AI - Main Application Launcher
CS3235 Undergraduate LLM / Multimodal AI Project
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui import create_ui

if __name__ == "__main__":
    print("=" * 65)
    print("🌾 Starting KrishiMitra AI: Indian-Language Farmer Advisory Assistant")
    print("🌾 Models & Frameworks: BharatGen AgriParam + AI4Bharat")
    print("🌾 Interface: Gradio Multilingual Web UI on http://127.0.0.1:7860")
    print("=" * 65)

    app, theme = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme)
