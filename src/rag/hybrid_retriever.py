from typing import List, Dict, Any, Tuple
import numpy as np
from src.rag.bm25_retriever import BM25AgriculturalRetriever
from src.rag.embeddings import MultilingualEmbeddings

class HybridAgriculturalRetriever:
    """
    Hybrid Retriever utilizing Reciprocal Rank Fusion (RRF) between
    BM25 (lexical exact keyword match) and Multilingual Dense Semantic Search.
    """
    def __init__(self, chunks: List[Dict[str, Any]], embeddings_manager: MultilingualEmbeddings = None, rrf_k: int = 60):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25_retriever = BM25AgriculturalRetriever(chunks)
        self.embeddings_manager = embeddings_manager or MultilingualEmbeddings()
        self.chunk_embeddings = self.embeddings_manager.compute_and_cache_chunk_embeddings(chunks)

    def retrieve(self, query: str, top_k: int = 3, dense_weight: float = 1.0, bm25_weight: float = 1.0) -> List[Dict[str, Any]]:
        # 1. BM25 Search
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 3)
        # 2. Dense Semantic Search
        dense_results = self.embeddings_manager.similarity_search(query, self.chunks, self.chunk_embeddings, top_k=top_k * 3)

        # 3. Reciprocal Rank Fusion
        rrf_scores = {}
        provenance = {}

        for rank, (chunk, score) in enumerate(bm25_results):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (bm25_weight / (self.rrf_k + rank + 1))
            if cid not in provenance:
                provenance[cid] = {"chunk": chunk, "bm25_rank": rank + 1, "dense_rank": None, "bm25_score": score, "dense_score": 0.0}
            else:
                provenance[cid]["bm25_rank"] = rank + 1
                provenance[cid]["bm25_score"] = score

        for rank, (chunk, score) in enumerate(dense_results):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (dense_weight / (self.rrf_k + rank + 1))
            if cid not in provenance:
                provenance[cid] = {"chunk": chunk, "bm25_rank": None, "dense_rank": rank + 1, "bm25_score": 0.0, "dense_score": score}
            else:
                provenance[cid]["dense_rank"] = rank + 1
                provenance[cid]["dense_score"] = score

        # Sort by final RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        final_chunks = []
        for cid in sorted_cids:
            item = provenance[cid]["chunk"].copy()
            item["rrf_score"] = float(rrf_scores[cid])
            item["bm25_rank"] = provenance[cid]["bm25_rank"]
            item["dense_rank"] = provenance[cid]["dense_rank"]
            item["dense_score"] = provenance[cid]["dense_score"]
            final_chunks.append(item)

        return final_chunks
