import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

class MultilingualEmbeddings:
    """
    Multilingual Dense Semantic Embedding Manager.
    Supports SentenceTransformers multilingual models with disk caching
    and graceful fallback if offline.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", cache_dir: Path = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path(__file__).resolve().parents[2] / "data" / "processed"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embeddings] Loading dense multilingual embedding model: {self.model_name}...")
            # Load with cpu to stay lightweight and avoid OOM
            self.model = SentenceTransformer(self.model_name, device="cpu")
            print("[Embeddings] Multilingual model successfully loaded on CPU.")
        except Exception as e:
            print(f"[Embeddings] Warning: Could not load SentenceTransformer ({e}). Falling back to statistical n-gram vectorizer.")
            self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        else:
            # Fallback semantic hash representation
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        # 384-dimensional deterministic feature hashing for offline robustness
        dim = 384
        vectors = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                idx = hash(w) % dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors)

    def compute_and_cache_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        cache_path = self.cache_dir / "chunk_embeddings.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    if cached_data.get("model") == self.model_name and len(cached_data.get("chunks", [])) == len(chunks):
                        print("[Embeddings] Using cached dense chunk embeddings.")
                        return cached_data["embeddings"]
            except Exception as e:
                print(f"[Embeddings] Cache read error: {e}, recomputing.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.encode(texts)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "model": self.model_name,
                    "chunks": [c["id"] for c in chunks],
                    "embeddings": embeddings
                }, f)
            print(f"[Embeddings] Computed and saved {len(embeddings)} chunk embeddings.")
        except Exception as e:
            print(f"[Embeddings] Error saving cache: {e}")

        return embeddings

    def similarity_search(self, query: str, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        query_vec = self.encode([query])[0]
        # Dot product with normalized embeddings equals cosine similarity
        scores = np.dot(chunk_embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((chunks[idx], float(scores[idx])))
        return results
