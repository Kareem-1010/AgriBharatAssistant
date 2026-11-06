"""
KrishiMitra AI - Enterprise Launcher
CS3235 Undergraduate LLM / Multimodal AI Project
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui import create_ui

if __name__ == "__main__":
    print("=" * 70)
    print("🌾 Launching KrishiMitra AI: National Multilingual Farmer Advisory System")
    print("🌾 Models & Frameworks: BharatGen AgriParam + AI4Bharat")
    print("🌾 Frontend: Enterprise Gradio 6.0 Web Application")
    print("🌾 URL: http://127.0.0.1:7860")
    print("=" * 70)

    app, css = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, css=css)
