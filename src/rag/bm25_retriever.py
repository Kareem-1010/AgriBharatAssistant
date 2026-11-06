import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

class BM25AgriculturalRetriever:
    """
    BM25 sparse keyword retriever tuned for Indian agricultural vocabulary,
    crop common names, chemical active ingredients, and transliterated queries.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, clean punctuation while preserving Indic Unicode codepoints and alphanumeric
        cleaned = re.sub(r'[^\w\s\u0900-\u0D7F]', ' ', text.lower())
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
        return tokens

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                results.append((self.chunks[idx], float(scores[idx])))
        return results
