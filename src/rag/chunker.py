import re
from pathlib import Path
from typing import List, Dict, Any

class AgriculturalChunker:
    """
    Agricultural Document Chunker with domain metadata extraction.
    Extracts crop names, topics (pests, fertilizer, irrigation, schemes),
    and creates semantically coherent chunks.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def detect_category(self, text: str, filename: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["fertilizer", "npk", "urea", "dap", "potash", "zinc"]):
            return "fertilizer_npk"
        elif any(w in text_lower for w in ["pm-kisan", "pmfby", "scheme", "subsidy", "kcc", "insurance"]):
            return "govt_schemes"
        elif any(w in text_lower for w in ["irrigation", "water", "awd", "cri stage"]):
            return "irrigation"
        elif any(w in text_lower for w in ["weed", "phalaris", "sulfosulfuron", "clodinafop"]):
            return "weed_management"
        elif any(w in text_lower for w in ["pest", "disease", "blight", "rust", "bollworm", "insect"]):
            return "pest_disease"
        return "general_agronomy"

    def detect_crops(self, text: str) -> List[str]:
        crops = []
        text_lower = text.lower()
        crop_keywords = {
            "Rice": ["rice", "paddy", "धान", "వరి", "நெல்", "भात"],
            "Wheat": ["wheat", "गेहूं", "గోధుమ", "கோதுமை", "गहू"],
            "Cotton": ["cotton", "कपास", "పత్తి", "பருத்தி", "कापूस"],
            "Sugarcane": ["sugarcane", "गन्ना", "చెరకు", "கரும்பு", "ऊस"],
            "Tomato": ["tomato", "टमाटर", "టమాట", "தக்காளி", "टोमॅटो"],
            "Mustard": ["mustard", "सरसों", "ఆవాలు", "கடுகு", "मोहरी"],
            "Chickpea/Gram": ["gram", "chickpea", "चना", "శనగలు", "கொண்டைக்கடலை", "हरभरा"]
        }
        for crop, keys in crop_keywords.items():
            if any(k in text_lower for k in keys):
                crops.append(crop)
        return crops or ["General/Multiple"]

    def chunk_document(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        filename = file_path.name
        # Split by markdown sections (## or ###)
        sections = re.split(r'\n(?=##?\s)', content)
        chunks = []
        chunk_idx = 0

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            lines = sec.split("\n")
            header = lines[0].replace("#", "").strip()
            body = "\n".join(lines[1:]).strip()

            if not body:
                body = header

            # If section fits within chunk_size, keep whole
            words = body.split()
            if len(words) <= self.chunk_size:
                chunk_text = f"[{filename} | {header}]\n{body}"
                chunks.append({
                    "id": f"{filename}_{chunk_idx}",
                    "source": filename,
                    "section": header,
                    "text": chunk_text,
                    "category": self.detect_category(sec, filename),
                    "crops": self.detect_crops(sec),
                    "word_count": len(words)
                })
                chunk_idx += 1
            else:
                # Sliding window chunking for long sections
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, len(words), step):
                    window = " ".join(words[i:i + self.chunk_size])
                    chunk_text = f"[{filename} | {header} (Part {i//step + 1})]\n{window}"
                    chunks.append({
                        "id": f"{filename}_{chunk_idx}",
                        "source": filename,
                        "section": f"{header} (Part {i//step + 1})",
                        "text": chunk_text,
                        "category": self.detect_category(window, filename),
                        "crops": self.detect_crops(window),
                        "word_count": len(window.split())
                    })
                    chunk_idx += 1

        return chunks

    def process_all_documents(self, raw_docs_dir: Path) -> List[Dict[str, Any]]:
        all_chunks = []
        for file_path in raw_docs_dir.glob("*.md"):
            chunks = self.chunk_document(file_path)
            all_chunks.extend(chunks)
        return all_chunks
